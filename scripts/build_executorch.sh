#!/usr/bin/env bash
# Builds ExecuTorch (+ its pinned PyTorch/torchvision versions) from source
# into a fresh virtualenv, reproducing the exact recipe used to create the
# working build this project's app/python/vendor/ was populated from.
#
# This is a LONG-RUNNING, HEAVY operation — expect anywhere from tens of
# minutes to a few hours depending on hardware, since it compiles ExecuTorch's
# C++ backends (XNNPACK included) from source via CMake, on top of downloading
# a full PyTorch wheel and this repo's git submodules. Only run it when you
# actually need to (re)build the environment: disaster recovery on a fresh
# board, or deliberately moving to a newer executorch version.
#
# Recipe, reverse-engineered from the currently-working install (see
# docs/executorch-integration.md for how this was confirmed):
#   - executorch is installed via `pip install <checkout>` (non-editable, the
#     default) from a local git checkout at tag v1.3.1 — this is what
#     produces the "+<short-sha>" local version suffix.
#   - torch/torchvision are NOT built from source — install_executorch.py
#     fetches them as prebuilt wheels. Running it with its default flags
#     (no --use-pt-pinned-commit) pulls the exact pinned "torch==2.12.0" from
#     PyTorch's test/RC channel, matching the currently-installed version.
#   - No --minimal flag was used: the resulting venv includes the full
#     example-script dependency set (matplotlib, torchaudio, etc.), confirmed
#     by torchaudio==2.11.0 matching install_requirements.py's non-minimal pin.
#
# Usage:
#   scripts/build_executorch.sh [options]
#
# Options:
#   --repo PATH    Path to the executorch git checkout.
#                  Default: /home/arduino/executorch
#                  Cloned fresh (with submodules) if it doesn't exist yet.
#   --venv PATH    Target virtualenv path to build into.
#                  Default: /home/arduino/.venv
#   --ref REF      Git tag/branch/commit to build. Default: v1.3.1
#   --force        Allow reusing/overwriting an existing venv at --venv.
#                  Refused by default so this script can't silently clobber
#                  a known-working environment.
#   -h, --help     Show this help.

set -euo pipefail

REPO_PATH="/home/arduino/executorch"
VENV_PATH="/home/arduino/.venv"
REF="v1.3.1"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_PATH="$2"; shift 2 ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -e "$VENV_PATH" && "$FORCE" -ne 1 ]]; then
  echo "Refusing to touch existing venv at $VENV_PATH (pass --force to reuse/overwrite it)." >&2
  echo "If it's already working, you probably want scripts/vendor_executorch.py instead." >&2
  exit 1
fi

echo "== ExecuTorch source build =="
echo "  repo: $REPO_PATH"
echo "  venv: $VENV_PATH"
echo "  ref:  $REF"
echo

if [[ ! -d "$REPO_PATH/.git" ]]; then
  echo "== Cloning executorch (with submodules) into $REPO_PATH =="
  git clone --branch "$REF" --recursive https://github.com/pytorch/executorch.git "$REPO_PATH"
else
  echo "== Using existing checkout at $REPO_PATH =="
  if [[ -n "$(git -C "$REPO_PATH" status --porcelain)" ]]; then
    echo "WARNING: checkout has local modifications — leaving them as-is." >&2
  fi
  git -C "$REPO_PATH" fetch --tags
  git -C "$REPO_PATH" checkout "$REF"
  echo "== Syncing submodules =="
  git -C "$REPO_PATH" submodule sync --recursive
  git -C "$REPO_PATH" submodule update --init --recursive
fi

echo
echo "== Creating virtualenv at $VENV_PATH =="
python3 -m venv "$VENV_PATH"

echo
echo "== Building executorch from source — this is the long part =="
cd "$REPO_PATH"
"$VENV_PATH/bin/python" install_executorch.py

echo
echo "== Build complete. Installed versions: =="
"$VENV_PATH/bin/python" -m pip show torch torchvision executorch 2>/dev/null | grep -E "^(Name|Version):"

echo
echo "Next step: re-run scripts/vendor_executorch.py to copy the built"
echo "packages into app/python/vendor/ and verify the app still works."

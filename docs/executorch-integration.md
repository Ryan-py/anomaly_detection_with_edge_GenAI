# Getting ExecuTorch Running Inside an Arduino App Lab App

This documents how local ExecuTorch/PyTorch inference was made to work inside
the `app/` Arduino App Lab application, why the obvious approaches don't work
on this platform, and how to reproduce or update the solution later.

## TL;DR

Arduino App Lab apps run inside a Docker container that only has access to
their own app folder — it cannot see or `pip install` the `torch`/`executorch`
build that already exists (and was built from source) in the host venv at
`/home/arduino/.venv`, and the container has no compiler to rebuild
them itself. The fix: **copy the exact already-built packages into the app
folder** (`app/python/vendor/`, git-ignored) and **prepend that directory to
`sys.path`** at the top of `app/python/inference_engine.py`, before importing
`torch`/`torchvision`/`executorch`. No `pip install`, no rebuild, no relaunch
trick — the container's own Python process just imports the vendored copies
directly.

## The problem

### 1. Apps run in an isolated container

`arduino-app-cli` runs every app inside a Docker container
(`ghcr.io/arduino/app-bricks/python-apps-base:0.12.0`) whose entrypoint
(`/run.sh`) bind-mounts **only the app's own folder** to `/app` inside the
container. Nothing else on the host filesystem — including
`/home/arduino/.venv`, where a working `torch`/`executorch` build
already exists — is reachable from inside that container.

`run.sh` also builds a per-app virtualenv at `/app/.cache/.venv` via
`uv venv --system-site-packages`, then runs `uv pip install -r
python/requirements.txt` if that file exists. This is the *only* supported
way to add extra Python dependencies to an app.

### 2. A fresh `pip install` isn't safe here

Two problems rule out just adding `torch`/`torchvision`/`executorch` to
`python/requirements.txt` and letting `uv pip install` fetch them fresh inside
the container:

- **The existing build was compiled from source**, not installed from a
  standard PyPI wheel — there's a full ExecuTorch git checkout at
  `/home/arduino/executorch/` confirming this, and it's what
  produced the exact working `.pte` runtime behavior the app depends on.
  A fresh `pip install` might resolve to a different build entirely (or fail
  to find a matching wheel for this board's exact aarch64/Python-3.13
  combination at all).
- **The container has no compiler** (`gcc`, `cmake`, etc. are all absent from
  the base image — confirmed directly). If no prebuilt wheel is available,
  `pip install` would have nothing to fall back to.

### 3. Relaunching into the host venv doesn't work either

The original design called for `main.py` to detect it isn't running inside
`/home/arduino/model_conversion/.venv/bin/python` and `os.execl()` itself
into it. Beyond that exact path not existing, this approach is fundamentally
incompatible with the container model: `execl`-ing into a venv that lives
outside `/app` isn't possible (the container can't see it), and even if it
were, that host venv doesn't have `arduino.app_utils`, `Bridge`, `WebUI`, or
`Camera` — those only exist inside the container's own Python environment.
Relaunching into the host venv would trade "no torch" for "no Arduino APIs."

## The solution: vendoring

Since the container **can** see anything placed inside the app folder itself,
the fix is to copy the already-built packages from the host venv into
`app/python/vendor/` (git-ignored — see `app/.gitignore`), and make the
container's Python process import from there directly via `sys.path`,
bypassing `pip`/`uv` entirely for these specific packages.

`app/python/inference_engine.py` does this at the very top of the file,
before any of the heavy imports:

```python
import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from executorch.extension.pybindings.portable_lib import _load_for_executorch
```

No other file needs to know about this — `main.py` just does a normal
`import inference_engine`. `python/requirements.txt` only lists `reportlab`
(a pure-Python wheel with no compiler needs); `torch`/`torchvision`/
`executorch` are deliberately **not** listed there.

## How the vendor list was derived

Copying the *entire* host venv's `site-packages` wasn't necessary or
desirable — a lot of it (pandas, scikit-learn, coremltools, matplotlib,
hydra-core, etc.) belongs to ExecuTorch's export/quantization tooling
(used by `model_convert/convert.py`, not by on-device inference) or is
unrelated to this project entirely.

The actual list was derived **empirically**, not guessed: a real inference
run — load the `.pte` model, run `.forward()` on `model_convert/rust_3.png`,
exactly what `inference_engine.run_inspection()` does — was executed in the
host venv, and every module that ended up in `sys.modules` afterward was
captured (`before`/`after` diff against a fresh interpreter, then reduced to
top-level package names). That gave a precise, verified list of what the real
code path actually touches, rather than everything `pip` happened to install
as a dependency.

The current vendored set (`app/python/vendor/`, ~950MB):

```
torch          torchvision     executorch      torchao
functorch      torchgen        numpy           numpy.libs
PIL            pillow.libs     torchvision.libs
sympy          networkx        setuptools      mpmath
fsspec         jinja2          markupsafe      yaml (PyYAML)
packaging      ruamel          filelock        flatbuffers
tabulate       dill            tqdm            typing_extensions.py
```

Each package's own `*.dist-info` directory is copied alongside it (some
internal `torch`/`executorch` code paths call `importlib.metadata` to check
installed versions).

### Two gotchas the first pass missed

The empirical `sys.modules` method isn't foolproof on its own — two real gaps
turned up only by actually running the vendored copy end-to-end in a fresh
process (see **Verification** below), both worth knowing about if this list
is ever regenerated:

1. **`*.libs` sibling directories.** `numpy`, `pillow`, and `torchvision` each
   ship a sibling `<name>.libs/` folder alongside the importable package,
   holding bundled shared libraries their compiled extensions link against at
   *import* time (e.g. `numpy`'s bundled OpenBLAS). These aren't Python
   packages themselves, so they don't show up as an importable module in
   `sys.modules` — but without them, `import numpy` fails with
   `ImportError: libscipy_openblas64_-....so: cannot open shared object
   file`. They must be copied even though nothing "imports" them by name.
2. **`functorch` / `torchgen`.** These are separate top-level packages
   bundled inside the same `torch` wheel (alongside `torch/` itself), pulled
   in transitively by `torch.nn.modules` → `torch.utils._python_dispatch` →
   `import torchgen`. They weren't caught by an earlier, cruder detection
   pass that only checked `importlib.metadata.packages_distributions()`
   (which doesn't map every importable top-level package back to its owning
   distribution) — the fix was to diff the *raw* `sys.modules` keys before
   and after a real run against `site-packages`' actual directory listing,
   not rely on distribution-metadata mapping alone.

If `torch`/`executorch` are ever rebuilt and this vendor directory needs
regenerating, re-run that same before/after `sys.modules` diff rather than
copying by intuition — both gotchas above were invisible until doing so.

## Verification

Two levels, both actually executed (not just read/assumed):

1. **Standalone, outside any venv.** Plain system `python3` (not the host
   venv that originally built these packages, and not the container) with
   only `sys.path` pointing at `app/python/vendor/` successfully imported
   `torch`, `torchvision`, and `executorch`, loaded the real `.pte` model,
   and ran inference on `model_convert/rust_3.png`, producing
   **6.807108968496323%** reconstruction error — an exact match to the
   reference run in the original host venv. This confirms the vendored
   copies are self-contained and don't secretly depend on anything from the
   venv they were copied out of.
2. **Inside the actual container.** After `arduino-app-cli app start`, the
   container logs showed `torch` importing cleanly (the same benign
   `torch.compile`/`KernelPreference` warning seen in the standalone test —
   a useful fingerprint that it's genuinely the same build running), and a
   live inspection via the API reproduced the identical
   `6.807108968496323%` result. This was the one real open question going
   in — whether the vendor copy's Python 3.13.5 build tag would be ABI
   -compatible with the container's Python 3.13.14 — and it was confirmed
   working, not just assumed.

## Disk footprint

`app/python/vendor/` is ~950MB. It lives under `/home/arduino`, which is its
own filesystem separate from `/` (root has much less headroom — this
distinction matters if checking `df -h /` instead of `df -h /home/arduino`
gives a misleadingly tight picture). It's listed in `app/.gitignore`
(`python/vendor/`), so it's never committed — only reproducible from the host
venv via the process above.

## If this ever needs to be redone

1. Confirm the working build still lives at
   `/home/arduino/.venv/lib/python3.13/site-packages/` (or wherever
   it's rebuilt to).
2. Run the real inference path in that venv and diff `sys.modules` before/after
   (see **How the vendor list was derived**) to get an authoritative package
   list — don't just copy the list in this doc verbatim if the underlying
   `torch`/`executorch` versions have changed, since transitive imports can
   shift between versions.
3. For each package in the resulting list, copy both the importable
   module/package and its `*.dist-info` folder, **plus** any sibling
   `*.libs/` directory (`ls site-packages | grep '\.libs$'` and copy the ones
   matching a vendored package).
4. Smoke-test with plain `python3` (not the container, not the source venv) —
   `sys.path.insert(0, ".../vendor")`, import the four top-level packages,
   load the `.pte`, run one real inference — before trusting it in the
   container.
5. `arduino-app-cli app start`, check `app logs --follow` for a clean
   `import torch` (no `ModuleNotFoundError`/`ImportError`), then re-run the
   same inference via the API and confirm the score matches the standalone
   test.

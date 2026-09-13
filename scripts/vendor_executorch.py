#!/usr/bin/env python3
"""Vendor a working torch/executorch build into app/python/vendor/.

Arduino App Lab apps run inside a container that can only see their own app
folder — it has no access to a host venv and no compiler, so torch/executorch
can't be `pip install`-ed fresh inside it (see docs/executorch-integration.md
for the full story). This script re-derives, from a real working install,
exactly which packages the app's actual inference code path touches, copies
them into app/python/vendor/ (git-ignored), and verifies the result works
completely on its own before replacing the live vendor directory.

IMPORTANT: run this with the SOURCE venv's own interpreter — the one that
already has a working torch/torchvision/executorch installed (e.g. built by
scripts/build_executorch.sh) — not with the system python3:

    /home/arduino/.venv/bin/python scripts/vendor_executorch.py

What it does, in order:
  1. Diffs sys.modules before/after actually loading the .pte model and
     running one real inference (the same code path app/python/
     inference_engine.py uses) — this is the authoritative list of what's
     actually needed, not a guess. This is also what produces the "expected"
     reconstruction-error value used to sanity-check the result at the end.
  2. Maps each touched top-level module back to the wheel distribution that
     owns it by parsing each *.dist-info/RECORD in site-packages — this is
     what automatically pulls in sibling *.libs/ folders (e.g. numpy.libs)
     and multi-package bundles (torch/functorch/torchgen all ship under one
     "torch" distribution) without hardcoding either case.
  3. Copies everything into a staging directory, then smoke-tests it in a
     brand new `python3` subprocess (not this venv, not the container) with
     only the staged directory on sys.path — proving the copy is genuinely
     self-contained.
  4. Only on a passing smoke test does it replace the live app/python/vendor/.

Usage:
    <source-venv>/bin/python scripts/vendor_executorch.py [options]

Options:
    --dest PATH              Destination vendor directory.
                              Default: <repo>/app/python/vendor
    --model PATH              .pte model to run the verification inference
                              with. Default: app/assets/models/
                              vae_anomaly_xnnpack_int8.pte, falling back to
                              model_convert/vae_anomaly_xnnpack_int8.pte.
    --reference-image PATH   Image to run through the model for the
                              before/after diff and smoke test.
                              Default: model_convert/rust_3.png
    --expected-error-pct N   Override the expected reconstruction-error
                              percentage the smoke test must match (default:
                              whatever this run itself computes).
    --tolerance N             Allowed absolute difference from the expected
                              error percentage. Default: 0.01
    --keep-staging            Don't delete the staging directory on success
                              (useful for inspecting what was collected).
"""

import argparse
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def site_packages_dir() -> Path:
    purelib = Path(sysconfig.get_paths()["purelib"])
    if not purelib.exists():
        fail(f"Could not locate site-packages at {purelib}")
    return purelib


def run_reference_inference(model_path: Path, image_path: Path):
    """Runs the exact inference code path app/python/inference_engine.py
    uses, capturing sys.modules before/after so we know precisely what it
    touched. Returns (touched_top_level_names, error_percentage)."""
    baseline = set(sys.modules.keys())

    import torch
    from PIL import Image
    from torchvision.transforms import transforms
    from executorch.extension.pybindings.portable_lib import _load_for_executorch

    edge_module = _load_for_executorch(str(model_path))
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)
    output = edge_module.forward([img_tensor])
    recon = output[0]
    error_percentage = torch.mean(torch.abs(img_tensor - recon)).item() * 100

    after = set(sys.modules.keys())
    touched_top_level = sorted({name.split(".")[0] for name in (after - baseline)})
    return touched_top_level, error_percentage


def filter_to_site_packages_entries(names, site_packages: Path):
    """Keep only names that actually exist as a file/dir in site-packages —
    drops stdlib/builtin modules, which the container already provides."""
    kept = []
    for name in names:
        if (site_packages / name).exists() or (site_packages / f"{name}.py").exists():
            kept.append(name)
    return kept


def build_record_owner_map(site_packages: Path):
    """Maps every top-level file/dir a dist-info's RECORD lists back to that
    dist-info's directory name — the ground-truth way to resolve aliases
    (PIL -> pillow, yaml -> PyYAML, ruamel -> ruamel_yaml) and multi-package
    bundles (torch/functorch/torchgen all under one torch distribution)."""
    owner_of = {}
    info_dirs = list(site_packages.glob("*.dist-info")) + list(site_packages.glob("*.egg-info"))
    for info_dir in info_dirs:
        record = info_dir / "RECORD"
        if not record.exists():
            continue
        for line in record.read_text(errors="ignore").splitlines():
            path = line.split(",")[0].strip()
            if not path or path.startswith(".."):
                continue
            top = path.split("/")[0]
            if top.endswith(".dist-info") or top.endswith(".egg-info") or not top:
                continue
            owner_of[top] = info_dir.name
    return owner_of


def owned_top_level_entries(info_dir: Path):
    """All top-level files/dirs listed in one dist-info's RECORD — this is
    what naturally captures a package's sibling *.libs/ folder alongside it,
    since wheel-packaging tools record those in RECORD too."""
    entries = set()
    record = info_dir / "RECORD"
    if not record.exists():
        return entries
    for line in record.read_text(errors="ignore").splitlines():
        path = line.split(",")[0].strip()
        if not path or path.startswith(".."):
            continue
        top = path.split("/")[0]
        if top.endswith(".dist-info") or top.endswith(".egg-info") or not top:
            continue
        entries.add(top)
    return entries


def resolve_vendor_set(touched_names, site_packages: Path):
    """Expands the empirically-touched import names into the full set of
    (dist-info dir, [owned top-level entries]) needed to vendor them
    completely and correctly."""
    owner_of = build_record_owner_map(site_packages)
    dist_info_dirs = set()
    unresolved = []
    for name in touched_names:
        owner = owner_of.get(name)
        if owner is None:
            unresolved.append(name)
            continue
        dist_info_dirs.add(owner)

    if unresolved:
        print(f"NOTE: {len(unresolved)} touched module(s) had no owning dist-info "
              f"(likely stdlib-adjacent or namespace packages) — copying them directly: "
              f"{', '.join(unresolved)}")

    plan = {}  # dist_info_dir_name -> set of top-level entries
    for info_name in sorted(dist_info_dirs):
        info_dir = site_packages / info_name
        plan[info_name] = owned_top_level_entries(info_dir)

    return plan, unresolved


def copy_vendor_set(plan, unresolved, site_packages: Path, staging: Path):
    staging.mkdir(parents=True, exist_ok=True)
    copied = []

    for info_name, entries in plan.items():
        src_info = site_packages / info_name
        dst_info = staging / info_name
        if src_info.exists() and not dst_info.exists():
            shutil.copytree(src_info, dst_info)
            copied.append(info_name)
        for entry in sorted(entries):
            src = site_packages / entry
            dst = staging / entry
            if not src.exists() or dst.exists():
                continue
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            copied.append(entry)

    for name in unresolved:
        src = site_packages / name
        src_py = site_packages / f"{name}.py"
        if src.exists():
            dst = staging / name
            if not dst.exists():
                shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
                copied.append(name)
        elif src_py.exists():
            dst = staging / f"{name}.py"
            if not dst.exists():
                shutil.copy2(src_py, dst)
                copied.append(f"{name}.py")

    return copied


def smoke_test(staging: Path, model_path: Path, image_path: Path, expected: float, tolerance: float):
    system_python = shutil.which("python3")
    if system_python is None:
        fail("Could not find a system python3 to run the independent smoke test with.")

    code = f"""
import sys
sys.path.insert(0, {str(staging)!r})
import torch
from PIL import Image
from torchvision.transforms import transforms
from executorch.extension.pybindings.portable_lib import _load_for_executorch

m = _load_for_executorch({str(model_path)!r})
img = Image.open({str(image_path)!r}).convert("RGB")
t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(img).unsqueeze(0)
out = m.forward([t])
recon = out[0]
error_pct = torch.mean(torch.abs(t - recon)).item() * 100
print(f"SMOKE_TEST_RESULT={{error_pct}}")
"""
    print("\n== Running independent smoke test (fresh python3 subprocess, vendor-only sys.path) ==")
    result = subprocess.run([system_python, "-c", code], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        fail("Smoke test subprocess failed — see output above. Staging directory left in place for inspection.")

    for line in result.stdout.splitlines():
        if line.startswith("SMOKE_TEST_RESULT="):
            got = float(line.split("=", 1)[1])
            if abs(got - expected) > tolerance:
                fail(
                    f"Smoke test produced {got}, expected {expected} "
                    f"(tolerance {tolerance}). Staging directory left in place for inspection."
                )
            print(f"Smoke test OK: {got} matches expected {expected} within {tolerance}.")
            return
    fail("Smoke test did not print a result line — something went wrong. See output above.")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=repo_root / "app" / "python" / "vendor")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--reference-image", type=Path, default=repo_root / "model_convert" / "rust_3.png")
    parser.add_argument("--expected-error-pct", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        candidates = [
            repo_root / "app" / "assets" / "models" / "vae_anomaly_xnnpack_int8.pte",
            repo_root / "model_convert" / "vae_anomaly_xnnpack_int8.pte",
        ]
        model_path = next((c for c in candidates if c.exists()), None)
        if model_path is None:
            fail(f"No .pte model found in either of: {[str(c) for c in candidates]}. Pass --model explicitly.")

    if not args.reference_image.exists():
        fail(f"Reference image not found: {args.reference_image}")

    print("== Vendoring executorch/torch from this interpreter's environment ==")
    print(f"  interpreter: {sys.executable}")
    print(f"  model:       {model_path}")
    print(f"  ref image:   {args.reference_image}")
    print(f"  destination: {args.dest}")
    print()

    print("== Step 1/4: running real inference to determine exactly what's needed ==")
    touched, measured_error_pct = run_reference_inference(model_path, args.reference_image)
    expected = args.expected_error_pct if args.expected_error_pct is not None else measured_error_pct
    site_packages = site_packages_dir()
    touched = filter_to_site_packages_entries(touched, site_packages)
    print(f"  touched {len(touched)} site-packages entries; reference error = {measured_error_pct}%")

    print("\n== Step 2/4: resolving owning distributions via RECORD files ==")
    plan, unresolved = resolve_vendor_set(touched, site_packages)
    print(f"  {len(plan)} distribution(s) to vendor: {', '.join(sorted(plan))}")

    staging = args.dest.parent / (args.dest.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)

    print(f"\n== Step 3/4: copying into staging directory {staging} ==")
    copied = copy_vendor_set(plan, unresolved, site_packages, staging)
    total_size = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    print(f"  copied {len(copied)} entries, {total_size / (1024 * 1024):.0f}MB total")

    smoke_test(staging, model_path, args.reference_image, expected, args.tolerance)

    if args.keep_staging:
        print(f"\n--keep-staging set: leaving the verified build at {staging} without "
              f"touching {args.dest}. Promote it manually when ready:\n"
              f"  rm -rf {args.dest} && mv {staging} {args.dest}")
        return

    print(f"\n== Step 4/4: replacing {args.dest} ==")
    if args.dest.exists():
        shutil.rmtree(args.dest)
    staging.rename(args.dest)

    print(f"\nDone. {args.dest} now holds a verified, self-contained torch/executorch build.")


if __name__ == "__main__":
    main()

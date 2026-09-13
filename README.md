# Edge Anomaly Detection Dashboard

An [Arduino App Lab](https://docs.arduino.cc/software/app-lab/) app for the
**Arduino Uno Q** that tracks physical degradation of industrial components 
cables, grids, metal nuts, screws, transistors  over time. Inspection images
(from the board's USB camera or an upload) are run through a local INT8
ExecuTorch VAE anomaly detector; results are logged per component instance,
charted over time, exportable as PDF reports, and drive an 8×13 LED-matrix
alert on the board itself when a reading crosses a configurable critical
threshold. 


## First-time setup

Bringing this project up on a board that doesn't already have it set up 
each step depends on files the previous one produces, so follow the order.

### 1. Clone the repo onto the board

```bash
git clone <this-repo-url> /home/arduino/anomaly_detection_with_edge_GenAI
cd /home/arduino/anomaly_detection_with_edge_GenAI
```

Cloning to exactly this path matters: `scripts/build_executorch.sh` and
`scripts/vendor_executorch.py` both default to
`/home/arduino/.venv` and `/home/arduino/executorch`.
Cloning elsewhere just means passing `--venv`/`--repo` explicitly in step 2.

### 2. Build ExecuTorch from source, then vendor it into the app

```bash
sudo ./scripts/build_executorch.sh
/home/arduino/.venv/bin/python scripts/vendor_executorch.py
```

The first command is the long one (tens of minutes to a few hours — see
[`docs/executorch-integration.md`](docs/executorch-integration.md) for why
this can't just be a `pip install`) and **requires `sudo`** — it installs
system build dependencies (`python3-venv`, `python3-dev`/`python3.13-dev`,
`build-essential`, `libzstd-dev`, `pkg-config`) via `apt-get` before building,
so expect a sudo password prompt unless it's already cached or passwordless
for this user. It builds a venv at
`/home/arduino/.venv`. The second command copies exactly what
`app/python/inference_engine.py` needs out of that venv into
`app/python/vendor/`, and verifies the copy works on its own before
finishing.

### 3. Bring in the trained model and convert it

From your training machine, copy the trained weights and calibration tensor
to the board, into `model_convert/`, named **exactly** as `convert.py`
expects — `vae_edge_weights.pth` and `calibration_data.pt` (rename them on
the way over if your local filenames differ):

```bash
scp vae_edge_weights.pth calibration_data.pt \
    arduino@<board-ip>:/home/arduino/anomaly_detection_with_edge_GenAI/model_convert/
```

Then run the conversion **using the venv step 2 just built** — `convert.py`
needs the same `torch`/`torchao`/`executorch` stack that lives there, not the
app's own (much smaller) environment:

```bash
cd /home/arduino/anomaly_detection_with_edge_GenAI/model_convert
/home/arduino/.venv/bin/python convert.py
```

This produces `vae_anomaly_xnnpack_int8.pte` right there in `model_convert/`.

### 4. Copy the converted model into the app

```bash
cd /home/arduino/anomaly_detection_with_edge_GenAI
mkdir -p app/assets/models
cp model_convert/vae_anomaly_xnnpack_int8.pte app/assets/models/
```

### 5. Make the app visible to Arduino App Lab

The Arduino App Lab GUI only lists projects under `~/ArduinoApps/`. A plain
`cp -r` works, but note it duplicates the ~900MB `app/python/vendor/`
directory along with everything else — a symlink avoids that and stays in
sync with the repo automatically:

```bash
ln -s /home/arduino/anomaly_detection_with_edge_GenAI/app \
      ~/ArduinoApps/edge-anomaly-dashboard
```

(Use `cp -r app ~/ArduinoApps/edge-anomaly-dashboard` instead if you
specifically want an independent copy rather than a link back to the repo.)
Open **Arduino App Lab** and start "Edge Anomaly Detection Dashboard" from
there — or skip the GUI entirely and use the CLI directly, see **Running the
app** below.

## Running the app

Once the steps above have been done at least once, there are two ways to
start/stop/watch it day-to-day — pick whichever fits how you're working.

**Arduino App Lab (GUI)** — requires the app to actually be under
`~/ArduinoApps/`, i.e. the symlink or copy from step 5. Open Arduino App Lab
and start "Edge Anomaly Detection Dashboard" from there.

**`arduino-app-cli` (terminal)** — works by pointing directly at the `app/`
folder wherever it lives, so it doesn't require step 5 at all (the symlink
into `~/ArduinoApps/` is only needed for the GUI to see it):

```bash
arduino-app-cli app start /path/to/this/repo/app
arduino-app-cli app logs  /path/to/this/repo/app --follow
```

Either way, `app start`/starting it from the GUI stops whatever app is
currently running on the board — check before running it. Once started, the
dashboard is at `http://<board-ip>:7000/` (or `http://localhost:7000/` from
the board itself).

First launch installs `reportlab` into the app's own venv (a few seconds —
it's a small pure-Python wheel) and imports `torch`/`executorch` from the
vendored copy in `app/python/vendor/`, which takes a little longer. Both are
one-time costs per container rebuild, not per restart.

The app starts without `app/assets/models/*.pte` or a populated
`app/python/vendor/`, but inspections will fail until both are in place —
see **First-time setup** above if either is missing.

## Model conversion (`model_convert/`)

`convert.py` trains/exports the VAE, quantizes it (PT2E, per-channel
symmetric INT8), and lowers it to `.pte` via the XNNPACK partitioner. This
step runs in a full PyTorch environment — not the app's own minimal
container — and its output (`vae_anomaly_xnnpack_int8.pte`) is what
`app/assets/models/` needs a copy of. `test_inference.py` is the reference
implementation `app/python/inference_engine.py`'s math is kept in sync with
(same preprocessing, same MAE-based error calculation).

Large artifacts here (`*.pth`, `*.pt`, `*.pte`, `edge_result*`) are
git-ignored intentionally — they're multi-hundred-MB build outputs, not
source.

## Database

No setup needed — `app/python/database.py` is plain stdlib `sqlite3`, no
server or separate service involved. `main.py` calls `database.init_db()` on
every startup, which creates `app/assets/static/db/anomaly_dashboard.db` (and
its parent directories) if it doesn't exist yet, runs `CREATE TABLE IF NOT
EXISTS` for all four tables, and seeds the 5 tracked component types and the
default 15% critical threshold via `INSERT OR IGNORE` — safe to run on every
restart, won't duplicate or reset existing data, but will pick up newly added
seed rows.

`assets/static/` is git-ignored (it's runtime data, not source), so a fresh
clone has no database file at all until the first `arduino-app-cli app
start` — at which point it appears fully created and seeded automatically.

## ExecuTorch: how inference works inside the app at all

Arduino App Lab apps run inside a Docker container that only has access to
their own app folder, with no compiler and no access to any host virtualenv.
That's a problem for this app specifically, because its `torch`/`executorch`
build was compiled from source (not a generic PyPI wheel) and can't simply be
`pip install`-ed fresh inside the container.

The fix — and the full reasoning behind it — is documented in
**[`docs/executorch-integration.md`](docs/executorch-integration.md)**. The
short version: the working build is copied ("vendored") into
`app/python/vendor/` and imported via `sys.path`, not `pip`. Two scripts
automate this:

- **`scripts/build_executorch.sh`** — (re)builds ExecuTorch from source into a
  fresh venv. Long-running (tens of minutes to a few hours) and heavy — only
  needed for disaster recovery on a fresh board or when deliberately moving
  to a newer ExecuTorch version. **Requires `sudo`** (installs system build
  dependencies via `apt-get` first). Refuses to touch an existing venv unless
  you pass `--force`.
- **`scripts/vendor_executorch.py`** — copies a working build into
  `app/python/vendor/`. Run it with the *source* venv's own Python
  interpreter:

  ```bash
  /home/arduino/.venv/bin/python scripts/vendor_executorch.py
  ```

  It determines exactly which packages are needed by actually running a real
  inference and diffing `sys.modules`, then verifies the result in an
  isolated subprocess before touching the live `app/python/vendor/`. Pass
  `--keep-staging` to build and verify without replacing the live vendor
  directory, if you want to inspect the result first.

Both are read-only with respect to `app/python/vendor/` until their final
step, and both print exactly what they're about to do before doing it.

## License

See [`LICENSE`](LICENSE).

# Contributing to CameraKit

Thanks for contributing. This guide documents architecture, workflow, and roadmap boundaries.

## Project Structure

Core command modules:

- `camerakit/cli.py`: unified CLI dispatch.
- `camerakit/devices.py`: camera discovery command.
- `camerakit/init.py`: project scaffold command.
- `camerakit/calibration.py`: calibration command entry.
- `camerakit/capture.py`: synchronized capture command.
- `camerakit/report.py`: calibration summary command.

Core runtime modules:

- `camerakit/utils/enumerator.py`: camera metadata + synchronizable filtering.
- `camerakit/utils/common.py`: logging setup, OpenCV logging control, hardware probing helpers.
- `camerakit/utils/calibration.py`: calibration pipeline internals.
- `camerakit/utils/sync.py`: threaded multi-camera frame reader.
- `camerakit/core/calibration_file.py`: calibration TOML reader/wrapper.
- `camerakit/core/calibration_data.py`: per-camera data model.

## Development Workflow

After cloning, switch to the `dev` branch first:

```bash
git checkout dev
git pull origin dev
```

All contributions must be based on `dev`, and pull requests should target `dev`.

1. Checkout and update `dev`.
2. Create your working branch from `dev` (or work directly on `dev` if your workflow requires it).
3. Make focused changes.
4. Run local checks.
5. Update docs/release notes for behavior changes.
6. Open a pull request targeting `dev`.

Typical local checks:

```bash
python3 -m compileall camerakit
```

If you have test tooling installed:

```bash
pytest -q
```

## Logging Conventions

- Use the package logger: `get_logger()` from `camerakit.utils.common`.
- Initialize per-command/session logging with `setup_logging()`.
- Avoid configuring or mutating the root logger in feature modules.

## Documentation Workflow

CameraKit docs are Sphinx-based in `docs/`.

Build docs locally:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

Open:

- `docs/_build/html/index.html`

When changing user-facing behavior:

- Update `README.md`
- Update `docs/`
- Update `RELEASE_NOTES.md`

## Roadmap Boundary

Stable scope today:

- Camera discovery
- Project initialization
- Checkerboard-based calibration workflow
- Synchronized software capture
- Calibration reporting

Experimental/in-progress scope:

- Third-party calibration conversion paths (Qualisys/Vicon/OpenCap/EasyMocap/bioCV)

Treat experimental paths as roadmap until they are stabilized, versioned, and fully documented.

## Pull Request Checklist

- [ ] Changes are based on `dev` and PR target is `dev`.
- [ ] Code compiles/runs locally.
- [ ] Logging behavior is consistent with package logger conventions.
- [ ] Documentation is updated.
- [ ] Release notes are updated for user-visible changes.

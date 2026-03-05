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

1. Create a branch.
2. Make focused changes.
3. Run local checks.
4. Update docs/release notes for behavior changes.
5. Open a pull request.

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

- [ ] Code compiles/runs locally.
- [ ] Logging behavior is consistent with package logger conventions.
- [ ] Documentation is updated.
- [ ] Release notes are updated for user-visible changes.

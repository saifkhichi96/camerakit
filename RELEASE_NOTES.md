# CameraKit Release Notes

## v2.0.0

This release established the v2 CLI and calibration output format.

### Highlights

- Added full Sphinx documentation site under `docs/` using the Furo theme.
- Added a shared SVG package header used in both README and docs home.
- Unified CLI under a single entry point:
  - `camerakit devices`
  - `camerakit init`
  - `camerakit calibrate`
  - `camerakit capture`
  - `camerakit report`
- Calibration outputs include separate error fields per camera and in metadata:
  - `intrinsics_error_px`
  - `extrinsics_error_px`
- Added `camerakit report` for quick calibration summaries.
- Logger usage standardized through the package logger.

### Breaking changes (v1 -> v2)

- Legacy `ck-*` command entrypoints were removed.
- Calibration TOML legacy key `error` was superseded by
  `intrinsics_error_px` and `extrinsics_error_px`.

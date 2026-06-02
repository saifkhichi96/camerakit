# CameraKit Release Notes

## v2.0.1

- Improved capture workflow for better codec compatibility.
- Removed the `--codec` CLI flag from `camerakit capture`.
- Added three new CLI flags to `camerakit capture` for finer control over capture settings:
  - `--capture-profile` (default: `balanced`): Preset capture settings for common use cases. Choices include `raw`, `balanced`, and `compressed`.
  - `--output-format` (default: `mp4`): Output video format. Choices include `mp4`, `avi`, and `mkv`.
  - `--output-profile` (default: `browser`): Preset output encoding settings. Choices include `browser`, `archive`, and `capture`.

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

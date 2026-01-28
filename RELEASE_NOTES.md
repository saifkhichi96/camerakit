# CameraKit Release Notes

## v2.0.0

This release establishes the v2 standard and aligns the CLI, documentation, and calibration outputs.

Highlights:
- Unified CLI: a single `camerakit` entry point with `devices`, `calibrate`, `capture`, and `report` subcommands.
- Removed legacy `ck-*` entrypoints.
- Calibration outputs now store **separate** reprojection errors for intrinsics and extrinsics (`intrinsics_error_px`, `extrinsics_error_px`) per camera and in metadata.
- Fisheye support removed (out of scope).
- Conversion/keypoint routes removed (out of roadmap).
- Added device listing and calibration summary helpers as subcommands.
- Logging standardized on the CameraKit logger; OpenCV noise silenced for device discovery.
- Documentation refreshed and CONTRIBUTING added with architecture + roadmap.

Breaking changes:
- CLI entrypoints changed to `camerakit <subcommand>`.
- Calibration TOML error key `error` replaced by `intrinsics_error_px` / `extrinsics_error_px`.
- Fisheye flags are no longer accepted or written.

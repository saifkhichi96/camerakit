Release Notes
=============

v2.0.1
------

- Improved capture workflow for better codec compatibility.
- Removed the `--codec` CLI flag from `camerakit capture`.
- Added three new CLI flags to `camerakit capture` for finer control over capture settings:
  - `--capture-profile` (default: `balanced`): Preset capture settings for common use cases. Choices include `raw`, `balanced`, and `compressed`.
  - `--output-format` (default: `mp4`): Output video format. Choices include `mp4`, `avi`, and `mkv`.
  - `--output-profile` (default: `browser`): Preset output encoding settings. Choices include `browser`, `archive`, and `capture`.


v2.0.0
------

- Added full Sphinx documentation (Furo theme).
- Added shared SVG header asset for README/docs landing pages.
- Unified CLI with ``camerakit`` entry point and subcommands:

  - ``devices``
  - ``init``
  - ``calibrate``
  - ``capture``
  - ``report``

- Calibration TOML now stores separate per-camera errors:

  - ``intrinsics_error_px``
  - ``extrinsics_error_px``

- Added calibration summary command (``camerakit report``).
- Logging standardized through the package logger.

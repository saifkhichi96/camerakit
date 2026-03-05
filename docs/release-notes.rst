Release Notes
=============

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

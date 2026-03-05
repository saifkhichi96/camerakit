Calibration
===========

Overview
--------

The stable calibration workflow is checkerboard-based and writes a ``Calib_*.toml`` file.

- Intrinsics are estimated per camera from multiple checkerboard views.
- Extrinsics are optionally estimated from one frame/clip per camera.
- Per-camera error fields are persisted as:

  - ``intrinsics_error_px``
  - ``extrinsics_error_px``

Configuration structure
-----------------------

The default ``Config.toml`` created by ``camerakit init`` uses this structure:

.. code-block:: toml

   [calibration]
   calibration_type = "calculate"

   [calibration.calculate.intrinsics]
   overwrite_intrinsics = false
   show_detection_intrinsics = true
   intrinsics_extension = "mp4"
   extract_every_N_sec = 1
   intrinsics_corners_nb = [9, 6]
   intrinsics_square_size = 25
   fisheye = false

   [calibration.calculate.extrinsics]
   calculate_extrinsics = true
   extrinsics_method = "board_outer"

   [calibration.calculate.extrinsics.board]
   extrinsics_corners_nb = [9, 6]
   extrinsics_square_size = 25
   extrinsics_extension = "png"
   show_reprojection_error = true

Extrinsics methods
------------------

- ``board``: use all detected checkerboard corners.
- ``board_outer``: use only 4 outer corners.
- ``scene``: manually click 2D points and match to configured 3D coordinates.

Fisheye mode
------------

Set ``calibration.calculate.intrinsics.fisheye = true`` to use OpenCV fisheye intrinsics.
Extrinsics then use fisheye undistortion before PnP.

Current limitations
-------------------

- Extrinsics are solved from a single frame per camera in this workflow.
- Manual annotation UX still has TODO hardening in edge cases.

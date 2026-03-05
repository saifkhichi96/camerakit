Calibration File Format
=======================

CameraKit writes TOML calibration files named ``Calib_<method>.toml``.

Per-camera section
------------------

Each camera section contains:

- ``name``
- ``size`` (``[width, height]``)
- ``matrix`` (3x3 intrinsics)
- ``distortions`` (4 coefficients)
- ``rotation`` (Rodrigues vector)
- ``translation``
- ``intrinsics_error_px`` (optional)
- ``extrinsics_error_px`` (optional)

Metadata section
----------------

The ``[metadata]`` section includes:

- ``adjusted`` (bool)
- ``intrinsics_error_px`` (list)
- ``extrinsics_error_px`` (list)

Example
-------

.. code-block:: toml

   [cam_00]
   name = "cam_00"
   size = [1920, 1080]
   matrix = [[1450.1, 0.0, 960.0], [0.0, 1450.1, 540.0], [0.0, 0.0, 1.0]]
   distortions = [0.02, -0.11, 0.0005, 0.0002]
   rotation = [0.01, -0.02, 0.03]
   translation = [0.0, 0.0, 0.0]
   intrinsics_error_px = 0.41
   extrinsics_error_px = 0.88

   [metadata]
   adjusted = false
   intrinsics_error_px = [0.41]
   extrinsics_error_px = [0.88]

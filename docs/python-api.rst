Python API
==========

High-level usage
----------------

Run calibration from Python:

.. code-block:: python

   from camerakit.calibration import run_calibration

   run_calibration("/path/to/project")

Discover cameras programmatically:

.. code-block:: python

   from camerakit.utils import CameraEnumerator

   cameras = CameraEnumerator(max_cameras=6).list()
   for cam in cameras:
       print(cam.id, cam.name, len(cam.settings))

Work with calibration files:

.. code-block:: python

   from camerakit.core import CalibrationFile

   calib = CalibrationFile("/path/to/Calib_board_outer.toml")
   print(len(calib))
   print(calib.metadata)

Stable exported symbols
-----------------------

From package root (``import camerakit``):

- ``run_calibration``
- ``find_cameras``
- ``SynchronizedVideoCapture``
- ``CalibrationFile``
- ``CalibrationData``
- ``CameraInfo``

From ``camerakit.utils``:

- ``CameraEnumerator``
- ``CameraMetadata``
- ``CaptureSettings``
- ``find_cameras``
- ``SynchronizedVideoCapture``

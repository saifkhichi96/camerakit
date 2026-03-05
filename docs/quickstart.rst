Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install camerakit

For capture features, install ``opencv-python`` (not ``opencv-python-headless``).

Project setup
-------------

.. code-block:: bash

   camerakit init --path /path/to/project --cameras 2

This creates:

.. code-block:: text

   project/
     Config.toml
     calibration/
       intrinsics/
         cam_00/
         cam_01/
       extrinsics/
         cam_00/
         cam_01/

Add calibration media
---------------------

- Put intrinsics clips/images in ``calibration/intrinsics/<cam_xx>/``.
- Put one extrinsics frame/clip per camera in ``calibration/extrinsics/<cam_xx>/``.

Run calibration
---------------

.. code-block:: bash

   camerakit calibrate --config /path/to/project

Inspect output
--------------

.. code-block:: bash

   camerakit report --input /path/to/project/calibration/Calib_board_outer.toml

Record synchronized videos
--------------------------

.. code-block:: bash

   camerakit capture --data-dir data --max-cameras 6

Use keys in the preview window:

- ``r`` start recording
- ``s`` stop and save trial
- ``q`` quit

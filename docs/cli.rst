CLI Reference
=============

CameraKit provides one entry point with subcommands:

.. code-block:: bash

   camerakit <command> [options]

Top-level commands
------------------

``devices``
~~~~~~~~~~~ 
List discoverable cameras and probed settings.

.. code-block:: bash

   camerakit devices --max-cameras 6 --codec MJPG,YUYV

Options:

- ``--max-cameras``: upper bound for discovery probing.
- ``--codec``: optional comma-separated FourCC list for probing.

``init``
~~~~~~~~
Create project folders and a default ``Config.toml``.

.. code-block:: bash

   camerakit init --path /path/to/project --cameras 2

Options:

- ``--path``: destination directory.
- ``--cameras``: number of camera subfolders to create.
- ``--force``: overwrite existing ``Config.toml``.

``calibrate``
~~~~~~~~~~~~~
Run calibration using ``Config.toml`` and calibration assets.

.. code-block:: bash

   camerakit calibrate --config /path/to/project

Options:

- ``--config``: config directory (defaults to current directory).

``capture``
~~~~~~~~~~~
Run synchronized live preview and per-trial recording.

.. code-block:: bash

   camerakit capture --data-dir data --required-fps 60 --max-cameras 6

Options:

- ``--required-fps``: optional output FPS target.
- ``--codec``: recording codec (default ``mp4v``).
- ``--max-cameras``: discovery limit (Linux/Windows probing).
- ``--data-dir``: output root for session/trial recordings.

``report``
~~~~~~~~~~
Print a calibration TOML summary.

.. code-block:: bash

   camerakit report --input /path/to/Calib_board_outer.toml

Options:

- ``--input``: calibration TOML path.

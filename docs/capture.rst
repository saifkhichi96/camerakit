Capture
=======

Overview
--------

``camerakit capture`` provides software-synchronized multi-camera recording:

- Probes available devices and settings.
- Filters selected cameras to common settings.
- Shows a live stacked preview.
- Records trial videos under ``data/Session_*/Trial_*``.

Requirements
------------

- GUI-enabled OpenCV build (``opencv-python``).
- If ``opencv-python-headless`` is installed, capture is disabled intentionally.

Controls
--------

In the preview window:

- ``r`` start a trial recording
- ``s`` stop and finalize current trial
- ``q`` exit

Output behavior
---------------

- Raw files are written as ``<cam_id>_raw.mp4`` while recording.
- On stop, files may be re-encoded to ``--required-fps`` when needed.
- Final outputs are named ``<cam_id>.mp4`` per trial folder.

Synchronization note
--------------------

Synchronization is software-based (threaded camera reads) and should be considered best-effort,
not hardware-trigger-accurate.

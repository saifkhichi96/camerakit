Development
===========

Architecture
------------

Primary modules:

- ``camerakit/cli.py``: unified CLI dispatch.
- ``camerakit/calibration.py``: calibration command orchestration.
- ``camerakit/capture.py``: synchronized capture command.
- ``camerakit/devices.py``: device listing command.
- ``camerakit/report.py``: calibration summary command.
- ``camerakit/utils/calibration.py``: calibration pipeline internals.
- ``camerakit/utils/enumerator.py``: camera enumeration classes.
- ``camerakit/utils/sync.py``: threaded synchronized frame reader.
- ``camerakit/core/calibration_file.py``: calibration TOML reader.

Logging
-------

Use the package logger via ``camerakit.utils.common.get_logger()``.
Session-level logging should be initialized with ``setup_logging()``.

Documentation workflow
----------------------

Build docs locally:

.. code-block:: bash

   pip install -r docs/requirements.txt
   make -C docs html

Generated output:

- ``docs/_build/html/index.html``

Contribution expectations
-------------------------

- Base all contribution work on ``dev``:

  .. code-block:: bash

     git checkout dev
     git pull origin dev

- Open pull requests targeting ``dev``.
- Keep TOML config/output compatibility when possible.
- Update docs and release notes when CLI behavior changes.
- Mark experimental features clearly (see roadmap).

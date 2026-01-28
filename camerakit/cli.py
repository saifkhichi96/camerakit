import argparse
import sys

from . import calibration, capture, devices, report
from .version import __version__


def _dispatch(module_main, passthrough_args):
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + passthrough_args
    try:
        module_main()
    finally:
        sys.argv = original_argv


def _add_passthrough(subparsers, name, help_text, module_main):
    subparser = subparsers.add_parser(name, help=help_text, add_help=False)
    subparser.add_argument("args", nargs=argparse.REMAINDER)
    subparser.set_defaults(_func=lambda args: _dispatch(module_main, args.args))


def main():
    parser = argparse.ArgumentParser(
        description="CameraKit unified CLI. Use a subcommand to access a tool.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"camerakit {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_passthrough(subparsers, "devices", "List available cameras.", devices.main)
    _add_passthrough(
        subparsers, "calibrate", "Run calibration from a Config.toml.", calibration.main
    )
    _add_passthrough(subparsers, "capture", "Record synchronized video.", capture.main)
    _add_passthrough(subparsers, "report", "Summarize a calibration TOML.", report.main)

    args = parser.parse_args()
    args._func(args)


if __name__ == "__main__":
    main()

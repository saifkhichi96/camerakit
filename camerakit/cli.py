import argparse
import sys

from . import calibration, capture, devices, report
from . import init as init_project
from .version import __version__

COMMANDS = {
    "devices": ("List available cameras.", devices.main),
    "init": ("Create a calibration project layout.", init_project.main),
    "calibrate": ("Run calibration from a Config.toml.", calibration.main),
    "capture": ("Record synchronized video.", capture.main),
    "report": ("Summarize a calibration TOML.", report.main),
}


def _dispatch(module_main, passthrough_args):
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + passthrough_args
    try:
        module_main()
    finally:
        sys.argv = original_argv


def _print_help():
    print("CameraKit unified CLI. Use a subcommand to access a tool.\n")
    print("Usage:")
    print("  camerakit [-V|--version] <command> [args]\n")
    print("Commands:")
    for name, (help_text, _) in COMMANDS.items():
        print(f"  {name:<10} {help_text}")
    print("\nTip: Use 'camerakit <command> --help' for command-specific options.")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument(
        "-V", "--version", action="version", version=f"camerakit {__version__}"
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS.keys())

    args, remainder = parser.parse_known_args()
    if args.help or not args.command:
        _print_help()
        return

    _, module_main = COMMANDS[args.command]
    _dispatch(module_main, remainder)


if __name__ == "__main__":
    main()

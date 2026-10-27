"""Build the FORGE-ready USD of a Robotiq robot profile (what forge_server.py does by itself at startup, as a command).

    python scripts/isaaclab/build_robot_usd.py --robot robotiq_2f85_tsf          # -> assets/robots/robotiq_2f85_tsf.usda
    python scripts/isaaclab/build_robot_usd.py --robot robotiq_2f85 --force      # rebuild even if up to date

The server builds the USD itself the first time a YAML asks for `robot: robotiq_2f85` (or `robotiq_2f85_tsf`) and rebuilds
it when an input changes, so this script is only needed to build and check it BEFORE a server run. It prints every check
of the result (articulation, FORGE's links, joints, taxel zones). The NVIDIA files (Robotiq USD, TSF_85.usd, the tactile
repository) are only read, from expo_ft/third_party/robotiq/; the output goes to the gitignored assets/robots/.

Runs in .venv-isaaclab (Isaac Sim is started headless to get the pxr library) or in any Python that has `usd-core`.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pxr  # noqa: F401  (usd-core, or an Isaac Sim already started)
    AppLauncher = None
except ImportError:
    from isaaclab.app import AppLauncher

from expo_ft.env.isaaclab.robots import PROFILES  # noqa: E402

parser = argparse.ArgumentParser(description="Build assets/robots/<robot>.usda for a Robotiq profile.")
parser.add_argument("--robot", default="robotiq_2f85_tsf", choices=sorted(n for n, p in PROFILES.items() if p.build_usd))
parser.add_argument("--force", action="store_true", help="Rebuild even if the USD is up to date.")
parser.add_argument("--exclude-zone-joints", action="store_true",
                    help="Tactile profile only: mark the 56 zone joints physics:excludeFromArticulation (fallback if PhysX "
                         "refuses the number of links; then the server re-syncs the zones after each reset).")
if AppLauncher is not None:
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
args = parser.parse_args()
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app if AppLauncher is not None else None

from expo_ft.env.isaaclab.robot_assets import ensure_robot_usd  # noqa: E402
from expo_ft.env.isaaclab.robots import get_profile  # noqa: E402

try:
    out = ensure_robot_usd(get_profile(args.robot), force=args.force, exclude_zone_joints=args.exclude_zone_joints)
    print(f"\nUSD ready: {out}")
    print(f"next: python scripts/isaaclab/inspect_robot_usd.py --usd {out.relative_to(REPO_ROOT) if REPO_ROOT in out.parents else out} --spawn")
finally:
    sys.stdout.flush()
    if app is not None:
        os._exit(0)          # app.close() can hang (seen with inspect_robot_usd.py)

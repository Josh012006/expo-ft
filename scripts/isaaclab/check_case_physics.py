"""What is the TSF-85 case (tsf85_case) in the derived Robotiq USD: a rigid body, an ACTIVE collider, or only visual?

Context: PhysX logs "Rigid Body of .../right_inner_finger/tsf85_case missing xformstack reset when child of another
enabled rigid body". This prints, for every tsf85_case prim, its physics attributes and the colliders below it.
Isaac Sim is started headless only to get the pxr library (pxr is not importable in a plain Python of the venv).

    python scripts/isaaclab/check_case_physics.py --usd assets/robots/robotiq_2f85_tsf.usda
"""
import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="assets/robots/robotiq_2f85_tsf.usda")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics  # noqa: E402

try:
    stage = Usd.Stage.Open(args.usd)
    found = 0
    for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if p.GetName() != "tsf85_case":
            continue
        found += 1
        enabled = p.GetAttribute("physics:rigidBodyEnabled")
        cols = [c for c in Usd.PrimRange(p, Usd.TraverseInstanceProxies()) if c.HasAPI(UsdPhysics.CollisionAPI)]
        live = [c for c in cols if UsdPhysics.CollisionAPI(c).GetCollisionEnabledAttr().Get() is not False]
        print(f"\n{p.GetPath()}")
        print(f"  RigidBodyAPI: {p.HasAPI(UsdPhysics.RigidBodyAPI)} | rigidBodyEnabled: {enabled.Get() if enabled else None}"
              f" | colliders below: {len(cols)} (enabled: {len(live)})")
        for c in cols[:5]:
            print(f"    collider: {c.GetPath()} | collisionEnabled: {UsdPhysics.CollisionAPI(c).GetCollisionEnabledAttr().Get()}")
        if live:
            print("  -> the case has ENABLED collision geometry")
        elif p.HasAPI(UsdPhysics.RigidBodyAPI):
            print("  -> visual only, but still carries a (disabled) RigidBodyAPI: PhysX logs an error for it")
        else:
            print("  -> visual only, no rigid body, no active collider: nothing for PhysX to complain about")
    print(f"\n{found} tsf85_case prim(s) found")
finally:
    sys.stdout.flush()
    os._exit(0)          # app.close() can hang

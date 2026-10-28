"""List the joints of a robot USD that are NOT part of the articulation (loop closures) and the joints with a mimic constraint.

Runs in .venv-isaaclab (Isaac Sim is started headless, no environment is built, no server is needed):

    python scripts/isaaclab/usd_joint_flags.py --usd expo_ft/third_party/robotiq/robotiq_usd/franka_robotiq_2f_85_flattened.usd | tee /tmp/usd_joint_flags.txt

Why: the Robotiq 2F-85 is a closed linkage (the inner finger closes a loop with the knuckle). PhysX cannot put a closed loop in
an articulation (a tree): one joint of the loop is authored with physics:excludeFromArticulation = true and solved as an ordinary
constraint between two rigid bodies, outside the articulation. The Franka Hand has no loop. scripts/isaaclab/reset_history.py
showed that the Franka's reset forgets the previous episode and the Robotiq's does not: this tells whether the Robotiq has such
constraints outside the articulation (their solver state is not what a joint-state write resets).
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Joints outside the articulation and mimic joints of a robot USD.")
parser.add_argument("--usd", required=True, help="Path of the robot .usd file.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
sys.stdout.reconfigure(line_buffering=True)
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics  # noqa: E402


def targets(rel):
    return [str(t).rsplit("/", 1)[-1] for t in rel.GetTargets()]


stage = Usd.Stage.Open(args.usd)
rows = []
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(prim)
    exclude_attr = prim.GetAttribute("physics:excludeFromArticulation")
    excluded = bool(exclude_attr.Get()) if exclude_attr and exclude_attr.HasValue() else False
    mimic = [s for s in prim.GetAppliedSchemas() if "Mimic" in s]
    rows.append((prim.GetName(), prim.GetTypeName(), targets(j.GetBody0Rel()), targets(j.GetBody1Rel()), excluded, mimic))

print(f"{len(rows)} joints in {args.usd}\n")
print("OUTSIDE the articulation (physics:excludeFromArticulation = true):")
out = [r for r in rows if r[4]]
for name, typ, b0, b1, _, _ in out:
    print(f"  {name:40s} {typ:22s} {b0} -> {b1}")
print(f"  -> {len(out)} joint(s)\n")
print("with a mimic constraint:")
mim = [r for r in rows if r[5]]
for name, typ, b0, b1, _, schemas in mim:
    print(f"  {name:40s} {typ:22s} {schemas}")
print(f"  -> {len(mim)} joint(s)")
app.close()

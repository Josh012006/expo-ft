"""Inspect a robot USD (for example franka_robotiq_2f_85_flattened.usd) before plugging it into FORGE.

Runs in .venv-isaaclab (Isaac Sim is started headless, no environment is built, no server is needed):

    python scripts/isaaclab/inspect_robot_usd.py --usd assets/robots/franka_robotiq_2f_85_flattened.usd | tee robot_usd_report.txt
    python scripts/isaaclab/inspect_robot_usd.py --usd assets/robots/franka_robotiq_2f_85_flattened.usd --spawn | tee robot_usd_report.txt

Part 1 (always), read from the USD file itself:
    stage units / up axis, articulation roots, every joint (type, bodies, limits, drive gains, MIMIC joints),
    every rigid body (mass, number of collision shapes), cameras, and the names FORGE relies on
    (panda_joint1..7, panda_hand, panda_link0, panda_fingertip_centered, panda_leftfinger, panda_rightfinger).
Part 2 (--spawn), the robot loaded as an Isaac Lab Articulation in a bare scene:
    the joint order and body order as Isaac Lab sees them (FORGE assumes [0:7] = arm and [7:9] = fingers),
    default joint positions, joint limits, and the position of every body in the root frame (to find the TCP offset
    and the finger pads).
Paste the output back: it tells what to change in forge_env_patched.py / forge_server.py.
"""
import argparse
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect a robot USD (joints, bodies, mimic joints, names FORGE needs).")
parser.add_argument("--usd", required=True, help="Path of the robot .usd file.")
parser.add_argument("--spawn", action="store_true",
                    help="Also load the robot as an Isaac Lab Articulation (joint order, limits, body poses).")
parser.add_argument("--match", default=r"hand|finger|pad|tip|tcp|robotiq|tactile|sensor|gripper|flange|ee_|camera",
                    help="Regex: bodies / prims whose name matches are listed (case-insensitive).")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
sys.stdout.reconfigure(line_buffering=True)   # with "| tee" Python buffers stdout: the report would only appear at exit
app = AppLauncher(args).app

import math  # noqa: E402

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

EXPECTED = (["panda_joint%d" % i for i in range(1, 8)]
            + ["panda_link0", "panda_hand", "panda_fingertip_centered", "panda_leftfinger", "panda_rightfinger",
               "panda_finger_joint1", "panda_finger_joint2"])


def f(v, nd=4):
    return "-" if v is None else f"{float(v):.{nd}f}"


def inspect_stage(path):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SystemExit(f"cannot open {path}")
    print("=" * 100)
    print(f"USD: {path}")
    print(f"defaultPrim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}   "
          f"metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)}   upAxis: {UsdGeom.GetStageUpAxis(stage)}   "
          f"kgPerUnit: {UsdPhysics.GetStageKilogramsPerUnit(stage)}")
    prims = [p for p in stage.Traverse()]
    print(f"{len(prims)} prims")

    roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print("\n-- Articulation roots:")
    for p in roots:
        print(f"   {p.GetPath()}")
    if not roots:
        print("   NONE (no PhysicsArticulationRootAPI: Isaac Lab cannot load it as an Articulation as is)")

    # ---- joints
    joints = [p for p in prims if p.IsA(UsdPhysics.Joint)]
    moving, fixed = [], []
    for p in joints:
        (fixed if p.GetTypeName() == "PhysicsFixedJoint" else moving).append(p)
    print(f"\n-- Joints: {len(moving)} movable, {len(fixed)} fixed")
    print(f"   {'name':28s} {'type':22s} {'body0 -> body1':50s} {'limits':22s} drive (stiffness / damping / maxForce)")
    for p in moving:
        j = UsdPhysics.Joint(p)
        b0 = [t.name for t in j.GetBody0Rel().GetTargets()]
        b1 = [t.name for t in j.GetBody1Rel().GetTargets()]
        typ = p.GetTypeName().replace("Physics", "")
        lim = "-"
        drive_kind = None
        if p.IsA(UsdPhysics.RevoluteJoint):
            r = UsdPhysics.RevoluteJoint(p)
            lo, hi = r.GetLowerLimitAttr().Get(), r.GetUpperLimitAttr().Get()          # degrees in USD
            lim = f"[{f(math.radians(lo), 3) if lo is not None else '-'}, {f(math.radians(hi), 3) if hi is not None else '-'}] rad"
            drive_kind = "angular"
        elif p.IsA(UsdPhysics.PrismaticJoint):
            r = UsdPhysics.PrismaticJoint(p)
            lo, hi = r.GetLowerLimitAttr().Get(), r.GetUpperLimitAttr().Get()
            lim = f"[{f(lo, 4)}, {f(hi, 4)}] m"
            drive_kind = "linear"
        drive = "no drive"
        if drive_kind:
            d = UsdPhysics.DriveAPI.Get(p, drive_kind)
            if d:
                drive = (f"{f(d.GetStiffnessAttr().Get(), 1)} / {f(d.GetDampingAttr().Get(), 1)} / "
                         f"{f(d.GetMaxForceAttr().Get(), 1)}")
        print(f"   {p.GetName():28s} {typ:22s} {','.join(b0) + ' -> ' + ','.join(b1):50s} {lim:22s} {drive}")
        for a in p.GetAttributes():                                                    # mimic joints (PhysX schema)
            if "mimic" in a.GetName().lower() and a.HasValue():
                print(f"         mimic: {a.GetName()} = {a.Get()}")
        for rel in p.GetRelationships():
            if "mimic" in rel.GetName().lower():
                print(f"         mimic: {rel.GetName()} -> {[str(t) for t in rel.GetTargets()]}")

    # ---- rigid bodies
    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    coll = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    counts = {}
    for c in coll:
        a = c.GetParent()
        while a and a.IsValid() and not a.HasAPI(UsdPhysics.RigidBodyAPI):
            a = a.GetParent()
        key = str(a.GetPath()) if a and a.IsValid() else "(no rigid body above)"
        counts[key] = counts.get(key, 0) + 1
    print(f"\n-- Rigid bodies: {len(bodies)}   (collision shapes: {len(coll)})")
    print(f"   {'name':34s} {'mass (kg)':>10s}  collision shapes")
    for p in bodies:
        m = UsdPhysics.MassAPI(p).GetMassAttr().Get() if p.HasAPI(UsdPhysics.MassAPI) else None
        print(f"   {p.GetName():34s} {f(m, 4):>10s}  {counts.get(str(p.GetPath()), 0)}")

    # ---- cameras, names of interest, names FORGE needs
    cams = [p for p in prims if p.IsA(UsdGeom.Camera)]
    print(f"\n-- Cameras in the file: {[str(p.GetPath()) for p in cams] or 'none'}")
    rx = re.compile(args.match, re.I)
    hits = [p for p in prims if rx.search(p.GetName()) and not p.IsA(UsdGeom.Mesh) and not p.IsA(UsdGeom.Scope)]
    print(f"\n-- Prims whose name matches /{args.match}/ ({len(hits)}):")
    for p in hits[:80]:
        print(f"   {str(p.GetPath()):80s} {p.GetTypeName()}")
    if len(hits) > 80:
        print(f"   ... {len(hits) - 80} more (narrow with --match)")

    names = {p.GetName() for p in prims}
    print("\n-- Names the current FORGE code relies on:")
    for n in EXPECTED:
        print(f"   {n:28s} {'found' if n in names else 'MISSING'}")
    return [p.GetName() for p in moving]


def inspect_spawn(path):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    # Default joint positions: FORGE's own start pose (factory_env_cfg.py). All zeros is refused by Isaac Lab:
    # panda_joint4 only goes from -3.072 to -0.070 rad.
    forge_pose = {"panda_joint1": 0.00871, "panda_joint2": -0.10368, "panda_joint3": -0.00794,
                  "panda_joint4": -1.49139, "panda_joint5": -0.00083, "panda_joint6": 1.38774, "panda_joint7": 0.0}
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(Path(path).resolve())),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=forge_pose, pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )
    robot = Articulation(cfg)
    sim.reset()
    robot.update(0.0)
    d = robot.data
    print("\n" + "=" * 100)
    print("Isaac Lab Articulation (what FORGE's code indexes):")
    print(f"   num joints {robot.num_joints}, num bodies {robot.num_bodies}")
    lim = d.joint_pos_limits[0].cpu().numpy()
    dflt = d.default_joint_pos[0].cpu().numpy()
    print(f"\n   {'idx':>3s}  {'joint':28s} {'default':>9s}  {'lower':>9s}  {'upper':>9s}")
    for i, n in enumerate(robot.joint_names):
        print(f"   {i:3d}  {n:28s} {dflt[i]:9.4f}  {lim[i, 0]:9.4f}  {lim[i, 1]:9.4f}")
    root = d.root_pos_w[0].cpu().numpy()
    pos = d.body_pos_w[0].cpu().numpy()
    quat = d.body_quat_w[0].cpu().numpy()                                   # (w, x, y, z), world frame
    try:
        masses = robot.root_physx_view.get_masses()[0].cpu().numpy()       # mass PhysX uses for each body
    except Exception:
        masses = None
    print(f"\n   {'idx':>3s}  {'body':34s} {'position in the root frame (x, y, z) m':40s} {'quat (w, x, y, z)':32s} mass kg")
    for i, n in enumerate(robot.body_names):
        p = pos[i] - root
        q = quat[i]
        m = "-" if masses is None else f"{masses[i]:.4f}"
        print(f"   {i:3d}  {n:34s} {p[0]:+8.4f} {p[1]:+8.4f} {p[2]:+8.4f}{'':16s} "
              f"{q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f} {q[3]:+.3f}   {m}")
    print("\n   FORGE assumes joints [0:7] = arm, [7:9] = fingers. If the order or the number of gripper joints "
          "differs, forge_env_patched.py (target[:, 7:9]) and forge_server.py (finger width) must change.")


try:
    movable = inspect_stage(args.usd)
    if args.spawn:
        inspect_spawn(args.usd)
except Exception:
    import traceback
    traceback.print_exc()
    print("\nThe inspection failed (traceback above): paste it back.", file=sys.stderr)
finally:
    # app.close() sometimes hangs after an Articulation was created: the report is printed, so leave right away.
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)

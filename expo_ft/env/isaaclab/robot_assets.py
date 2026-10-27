"""Builds the FORGE-ready USD of the Robotiq robots (needs pxr: runs inside Isaac Sim / Kit, or with usd-core).

    python scripts/isaaclab/build_robot_usd.py --robot robotiq_2f85_tsf          (what to run by hand)
    forge_server.py calls ensure_robot_usd() itself when the YAML says `robot: robotiq_2f85[_tsf]`.

Output: assets/robots/<robot>.usda (+ <robot>.json stamp, + <robot>_zones.usda for the taxel variant). All gitignored.
NVIDIA's files are never modified nor copied: the output only REFERENCES them (franka_robotiq_2f_85_flattened.usd,
TSF_85.usd) and the tactile repository's own builder (zoned_tactile.usd_builder) is imported and called as it is.

What FORGE needs from the robot, and that NVIDIA's USD (Franka arm + Robotiq, 18 bodies) does not have:
    panda_fingertip_centered   the tool frame: FORGE controls it (IK, hand_init_orn, where the peg sits in the hand).
                               Placed at the BOTTOM of the finger pads on the gripper axis, with the orientation of
                               panda_link8: the fingers then open along its y axis exactly like the Franka Hand's
                               fingertip frame, so FORGE's yaw conventions and the peg-in-hand pose stay valid.
    panda_leftfinger / panda_rightfinger   FORGE averages their Jacobians to get the fingertip Jacobian. Placed at the
                               bottom-centre of each pad (so their mean is the tool frame), fixed to the inner fingers.
    force_sensor               a link in series between panda_link8 and the gripper: FORGE reads the joint force that
                               enters it (get_link_incoming_joint_force) as the wrist force / torque.
All four are tiny rigid bodies joined by fixed joints (no new degree of freedom). panda_hand_joint (link8 -> base_link) is
re-pointed to start from force_sensor, which sits exactly on panda_link8.

Topology of the output (default prim /World, so Isaac Lab spawns it as .../Robot/):
    /World/robot          NVIDIA's articulation (panda_link0..8, Gripper/Robotiq_2F_85/..) + the 4 bodies above
    /World/TactileZones   (taxel variant only) 2 x 28 zone rigid bodies, fixed-jointed to the inner fingers, 4 mm boxes
                          that REPLACE the stock fingertip pads as the contact surface (NVIDIA's builder disables them)
NVIDIA's TSF-85 cases (visual only: body and collider both disabled) lose their leftover RigidBodyAPI in our layer
(drop_disabled_rigid_bodies), which PhysX reports as an error when it is nested in the finger's rigid body.
If PhysX refuses the articulation because of its number of links, rebuild with exclude_zone_joints=True
(--exclude-zone-joints / YAML `tactile_exclude_zone_joints: true`): the zone joints then leave the articulation.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from expo_ft.env.isaaclab.robots import (
    ASSETS_DIR, ROBOTIQ_SOURCE_USD, TACTILE_REPO, TSF85_USD, RobotProfile,
)

BUILDER_VERSION = 2
ROBOT = "/World/robot"
REQUIRED_BODIES = ["panda_link%d" % i for i in range(9)] + [
    "base_link", "force_sensor", "panda_fingertip_centered", "panda_leftfinger", "panda_rightfinger"]
OPEN_GAP_RANGE = (0.070, 0.095)       # m: inner-face distance of the pads of the OPEN Robotiq (83 mm measured)

FORCE_SENSOR_MASS, FORCE_SENSOR_INERTIA = 1.0e-2, 1.0e-6
ALIAS_MASS, ALIAS_INERTIA = 1.0e-3, 1.0e-7


# ---- small geometry helpers (column-vector 4x4 matrices) ----------------------------------------------------------
def _mat(xform_cache, prim):
    """Local-to-world transform of a prim, as a column-vector matrix (USD stores row-vector matrices)."""
    return np.array(xform_cache.GetLocalToWorldTransform(prim)).T


def _gf_matrix(m):
    return Gf.Matrix4d(np.asarray(m, dtype=float).T.tolist())


def _quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4)
        q[1 + i] = s / 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _find(stage, root, name):
    """The one prim called `name` below `root`."""
    found = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.GetName() == name]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one prim called {name!r} below {root}, found {len(found)}")
    return found[0]


def _mesh_points_world(xform_cache, prim):
    chunks = []
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if pts is None or len(pts) == 0:
                continue
            P = np.asarray(pts, dtype=float)
            chunks.append((np.c_[P, np.ones(len(P))] @ _mat(xform_cache, p).T)[:, :3])
    if not chunks:
        raise RuntimeError(f"no mesh below {prim.GetPath()}")
    return np.concatenate(chunks)


def _rigid_body(stage, path, world_matrix, parent_world, mass, inertia):
    """A tiny rigid body (Xform) whose authored pose is `world_matrix`, so that it starts exactly where its joint wants it."""
    xf = UsdGeom.Xform.Define(stage, path)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(_gf_matrix(np.linalg.inv(parent_world) @ world_matrix))
    prim = xf.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(float(inertia), float(inertia), float(inertia)))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
    return prim


def _fixed_joint(stage, path, body0, body1, T0=None, T1=None):
    """Fixed joint; T0 / T1 = pose of the joint frame in body0 / body1 (column-vector 4x4, identity by default)."""
    T0 = np.eye(4) if T0 is None else T0
    T1 = np.eye(4) if T1 is None else T1
    j = UsdPhysics.FixedJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(str(body0))])
    j.CreateBody1Rel().SetTargets([Sdf.Path(str(body1))])
    for suffix, T in (("0", T0), ("1", T1)):
        getattr(j, f"CreateLocalPos{suffix}Attr")().Set(Gf.Vec3f(*[float(v) for v in T[:3, 3]]))
        q = _quat(T[:3, :3])
        getattr(j, f"CreateLocalRot{suffix}Attr")().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*[float(v) for v in q[1:]])))
    return j


def _rigid_body_enabled(prim) -> bool:
    a = prim.GetAttribute("physics:rigidBodyEnabled")
    return not (a and a.HasValue() and a.Get() is False)


def drop_disabled_rigid_bodies(stage, log=print) -> int:
    """Remove the RigidBodyAPI of the prims NVIDIA left with `rigidBodyEnabled = False` (the TSF-85 case, one per finger).

    They are not bodies (disabled) and their collider is disabled too, so they only carry the visual mesh; but the API is
    still there, nested in the enabled rigid body of the inner finger, and PhysX logs an error for it ("missing xformstack
    reset when child of another enabled rigid body ... unpredicted results"). Removing the API in OUR layer (an override,
    NVIDIA's file is untouched) leaves the mesh as a plain visual child of the finger and silences the error.
    A prim is left alone, with a WARNING, if a collider below it is enabled: removing its body would then change the physics."""
    n = 0
    for p in list(Usd.PrimRange(stage.GetPrimAtPath(ROBOT))):
        if not p.HasAPI(UsdPhysics.RigidBodyAPI) or _rigid_body_enabled(p):
            continue
        live = [c.GetPath() for c in Usd.PrimRange(p) if c.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(c).GetCollisionEnabledAttr().Get() is not False]
        if live:
            log(f"[robot_assets] WARNING {p.GetPath()}: disabled rigid body with enabled colliders {live}, left as it is")
            continue
        p.RemoveAPI(UsdPhysics.RigidBodyAPI)
        n += 1
    if n:
        log(f"[robot_assets] {n} disabled rigid bodies (visual-only TSF-85 cases) turned into plain visual prims")
    return n


# ---- the four bodies FORGE looks for ----------------------------------------------------------------------------------
def add_forge_links(stage, log=print):
    """Author force_sensor, panda_fingertip_centered, panda_leftfinger, panda_rightfinger on /World/robot.
    Returns the measured geometry (dict, metres, frame of panda_link8)."""
    xc = UsdGeom.XformCache()
    link8 = _find(stage, ROBOT, "panda_link8")
    base = _find(stage, ROBOT, "base_link")
    fingers = {"left": _find(stage, ROBOT, "left_inner_finger"), "right": _find(stage, ROBOT, "right_inner_finger")}
    parent_world = _mat(xc, stage.GetPrimAtPath(ROBOT))
    W8, WB = _mat(xc, link8), _mat(xc, base)
    W8_inv = np.linalg.inv(W8)

    # -- the pads: bounding box of the stock fingertip meshes, in the frame of panda_link8 (z = along the gripper axis)
    pads = {}
    for side, finger in fingers.items():
        tips = [p for p in Usd.PrimRange(finger) if "fingertips" in p.GetName().lower() and not p.IsA(UsdGeom.Mesh)]
        if len(tips) != 1:
            raise RuntimeError(f"expected one 'fingertips' pad prim below {finger.GetPath()}, found {len(tips)}")
        world = _mesh_points_world(xc, tips[0])
        P = (np.c_[world, np.ones(len(world))] @ W8_inv.T)[:, :3]
        pads[side] = (P.min(0), P.max(0))
    (lmin, lmax), (rmin, rmax) = pads["left"], pads["right"]
    inner = sorted([lmin[1], lmax[1], rmin[1], rmax[1]])
    gap = inner[2] - inner[1]                       # distance between the two inner faces (the middle two y values)
    if not (OPEN_GAP_RANGE[0] <= gap <= OPEN_GAP_RANGE[1]):
        raise RuntimeError(f"the pads are {gap * 1000:.1f} mm apart: the authored gripper is not the open Robotiq "
                           f"(expected {OPEN_GAP_RANGE[0] * 1000:.0f}-{OPEN_GAP_RANGE[1] * 1000:.0f} mm)")
    z_bottom = 0.5 * (lmax[2] + rmax[2])            # bottom end of the pads (link8 z points to the fingertips)
    centre = lambda lo, hi: 0.5 * (lo + hi)
    pt_left = np.array([centre(lmin[0], lmax[0]), centre(lmin[1], lmax[1]), z_bottom])
    pt_right = np.array([centre(rmin[0], rmax[0]), centre(rmin[1], rmax[1]), z_bottom])
    tcp = 0.5 * (pt_left + pt_right)

    def pose_in_link8(p):                           # orientation of panda_link8, position p (in link8)
        T = np.eye(4)
        T[:3, 3] = p
        return W8 @ T

    # -- panda_hand_joint: link8 -> base_link becomes force_sensor -> base_link
    hand_joint = _find(stage, ROBOT, "panda_hand_joint")
    old_b0 = [str(t) for t in UsdPhysics.Joint(hand_joint).GetBody0Rel().GetTargets()]
    if not old_b0 or old_b0[0] != str(link8.GetPath()):
        raise RuntimeError(f"panda_hand_joint starts from {old_b0}, not from panda_link8: unexpected asset")
    fs_path, ft_path = f"{ROBOT}/force_sensor", f"{ROBOT}/panda_fingertip_centered"
    lf_path, rf_path = f"{ROBOT}/panda_leftfinger", f"{ROBOT}/panda_rightfinger"

    _rigid_body(stage, fs_path, W8, parent_world, FORCE_SENSOR_MASS, FORCE_SENSOR_INERTIA)
    _fixed_joint(stage, f"{fs_path}/fixed_joint_link8", link8.GetPath(), fs_path)
    UsdPhysics.Joint(hand_joint).GetBody0Rel().SetTargets([Sdf.Path(fs_path)])   # same pose as link8: local frames stay valid

    W_ft = pose_in_link8(tcp)
    _rigid_body(stage, ft_path, W_ft, parent_world, ALIAS_MASS, ALIAS_INERTIA)
    _fixed_joint(stage, f"{ft_path}/fixed_joint_base", base.GetPath(), ft_path, T0=np.linalg.inv(WB) @ W_ft)
    for side, path, pt in (("left", lf_path, pt_left), ("right", rf_path, pt_right)):
        W_pad = pose_in_link8(pt)
        _rigid_body(stage, path, W_pad, parent_world, ALIAS_MASS, ALIAS_INERTIA)
        _fixed_joint(stage, f"{path}/fixed_joint_finger", fingers[side].GetPath(), path,
                     T0=np.linalg.inv(_mat(xc, fingers[side])) @ W_pad)

    geom = {
        "tcp_in_link8_m": [round(float(v), 5) for v in tcp],
        "pad_bottom_from_link8_m": round(float(z_bottom), 5),
        "pad_height_m": round(float(0.5 * (lmax[2] - lmin[2] + rmax[2] - rmin[2])), 5),
        "pad_inner_gap_open_mm": round(float(gap) * 1000, 2),
        "left_pad_point_in_link8_m": [round(float(v), 5) for v in pt_left],
        "right_pad_point_in_link8_m": [round(float(v), 5) for v in pt_right],
        "base_link_in_link8_m": [round(float(v), 5) for v in (W8_inv @ WB)[:3, 3]],
    }
    log(f"[robot_assets] tool frame (panda_fingertip_centered) at {geom['tcp_in_link8_m']} m from panda_link8 "
        f"(Franka Hand: 0.1034 m), pads open {geom['pad_inner_gap_open_mm']} mm apart, {geom['pad_height_m'] * 1000:.0f} mm tall")
    return geom


# ---- the taxel variant: NVIDIA's own builder, called as it is -----------------------------------------------------
def _build_zones(zones_usd: Path, log=print):
    for p, what in ((ROBOTIQ_SOURCE_USD, "NVIDIA's Robotiq USD"), (TSF85_USD, "TSF_85.usd"),
                    (TACTILE_REPO / "src" / "zoned_tactile" / "usd_builder.py", "the tactile repository")):
        _check_file(p, what)
    sys.path.insert(0, str(TACTILE_REPO / "src"))
    try:
        from zoned_tactile import usd_builder
    finally:
        sys.path.pop(0)
    if zones_usd.exists():
        zones_usd.unlink()
    log("[robot_assets] NVIDIA tactile builder (usd_builder.build) ...")
    usd_builder.build(robot_usd=str(ROBOTIQ_SOURCE_USD.resolve()), output_usd=str(zones_usd.resolve()),
                      tsf85_usd=str(TSF85_USD.resolve()), with_tsf85_case=True)


def _check_file(path: Path, what: str):
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}\nPut NVIDIA's original files in expo_ft/third_party/robotiq/ "
                                "(gitignored): robotiq_usd/{franka_robotiq_2f_85_flattened.usd,TSF_85.usd} and "
                                "robotiq-tsf85-contact-twin-main/")
    if path.suffix in (".usd", ".usda", ".usdc") and path.stat().st_size < 2000:
        raise RuntimeError(f"{path} is only {path.stat().st_size} bytes: a Git LFS pointer, not the real file. "
                           "Use the original from NVIDIA's Archive.zip.")


# ---- build / verify / cache ----------------------------------------------------------------------------------------
def build_robot_usd(profile: RobotProfile, out_usd: Path, exclude_zone_joints=False, log=print) -> dict:
    out_usd = Path(out_usd)
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    _check_file(ROBOTIQ_SOURCE_USD, "NVIDIA's Robotiq USD")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    if profile.tactile:
        zones_usd = out_usd.with_name(out_usd.stem + "_zones.usda")
        _build_zones(zones_usd, log)
        world.GetPrim().GetReferences().AddReference(str(zones_usd.resolve()), "/World")
        if exclude_zone_joints:
            n = 0
            for p in stage.Traverse():
                if p.IsA(UsdPhysics.FixedJoint) and "/TactileZones/" in str(p.GetPath()):
                    p.CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(True)
                    n += 1
            log(f"[robot_assets] {n} tactile zone joints excluded from the articulation")
    else:
        stage.DefinePrim(ROBOT).GetReferences().AddReference(str(ROBOTIQ_SOURCE_USD.resolve()))

    drop_disabled_rigid_bodies(stage, log)
    geom = add_forge_links(stage, log)
    stage.GetRootLayer().Export(str(out_usd))
    summary = verify_robot_usd(out_usd, tactile=profile.tactile, log=log)
    summary["geometry"] = geom
    return summary


def verify_robot_usd(path, tactile, log=print) -> dict:
    """Open the written USD and check what FORGE and PhysX need: the bodies, joints that point to real bodies, and one
    connected tree from panda_link0 for everything under /World/robot."""
    stage = Usd.Stage.Open(str(path))
    robot = stage.GetPrimAtPath(ROBOT)
    if not robot.IsValid():
        raise RuntimeError(f"{path}: no {ROBOT}")
    roots = [p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"{path}: {len(roots)} articulation roots below {ROBOT}, expected 1")
    enabled = _rigid_body_enabled
    bodies = {str(p.GetPath()): p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI) and enabled(p)}
    disabled = sorted(str(p.GetPath()) for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI) and not enabled(p))
    if disabled:                        # PhysX would log "missing xformstack reset ..." for each of them
        log(f"[robot_assets] WARNING {len(disabled)} disabled rigid bodies remain under {ROBOT}: {disabled}")
    names = {p.GetName() for p in bodies.values()}
    missing = [n for n in REQUIRED_BODIES if n not in names]
    if missing:
        raise RuntimeError(f"{path}: rigid bodies missing: {missing}")
    parent = {b: b for b in bodies}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_joints = 0
    for p in Usd.PrimRange(robot):
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        t0, t1 = [str(t) for t in j.GetBody0Rel().GetTargets()], [str(t) for t in j.GetBody1Rel().GetTargets()]
        for t in t0 + t1:
            if t not in bodies:
                raise RuntimeError(f"joint {p.GetPath()} points to {t}, which is not a rigid body")
        if t0 and t1:
            parent[find(t0[0])] = find(t1[0])
        n_joints += 1
    link0 = next(b for b, p in bodies.items() if p.GetName() == "panda_link0")
    loose = sorted(p.GetName() for b, p in bodies.items()
                   if find(b) != find(link0) and p.GetName() not in ("banana", "bowl"))
    if loose:
        raise RuntimeError(f"{path}: bodies not connected to panda_link0 by a joint: {loose}")
    out = {"bodies_under_robot": len(bodies), "joints_under_robot": n_joints}
    if tactile:
        zones = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/TactileZones")) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(zones) != 56:
            raise RuntimeError(f"{path}: {len(zones)} tactile zone bodies, expected 56")
        out["tactile_zones"] = len(zones)
    log(f"[robot_assets] verified {path}: {out}")
    return out


def _digest(profile: RobotProfile, exclude_zone_joints: bool) -> dict:
    files = [ROBOTIQ_SOURCE_USD] + ([TSF85_USD, TACTILE_REPO / "src" / "zoned_tactile" / "usd_builder.py"]
                                    if profile.tactile else [])
    sizes = {str(f): (f.stat().st_size if f.is_file() else None) for f in files}
    own = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()
    zb = TACTILE_REPO / "src" / "zoned_tactile" / "usd_builder.py"
    zb_hash = hashlib.sha1(zb.read_bytes()).hexdigest() if (profile.tactile and zb.is_file()) else None
    return {"builder_version": BUILDER_VERSION, "this_file": own, "tactile_builder": zb_hash, "input_sizes": sizes,
            "exclude_zone_joints": bool(exclude_zone_joints)}


def ensure_robot_usd(profile: RobotProfile, force=False, exclude_zone_joints=False, log=print) -> Path:
    """Path of the USD to spawn for `profile`; builds it when missing or when an input / this file changed."""
    if profile.robot_usd:
        p = Path(profile.robot_usd)
        if not p.is_file():
            raise FileNotFoundError(f"robot_usd {p} does not exist")
        return p.resolve()
    if not profile.build_usd:
        raise ValueError(f"profile {profile.name} has no USD to build")
    out = ASSETS_DIR / f"{profile.name}.usda"
    stamp = out.with_suffix(".json")
    want = _digest(profile, exclude_zone_joints)
    if not force and out.is_file() and stamp.is_file():
        try:
            if json.loads(stamp.read_text()).get("digest") == want:
                log(f"[robot_assets] {out} is up to date")
                return out.resolve()
        except (OSError, ValueError):
            pass
    t0 = time.time()
    summary = build_robot_usd(profile, out, exclude_zone_joints, log)
    stamp.write_text(json.dumps({"digest": want, "summary": summary}, indent=1))
    log(f"[robot_assets] built {out} in {time.time() - t0:.1f} s")
    return out.resolve()

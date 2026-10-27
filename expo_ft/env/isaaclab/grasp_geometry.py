"""Where are the robot's bodies relative to the held peg? (used by the debug op `grasp_geometry` of forge_server.py)

The question: when the peg is pushed into the hole, do the FINGERS (not the peg) touch the socket top before the peg is
fully inserted? That depends on how far the peg sticks out below the lowest point of the fingers (the "protrusion").

Two steps, both here so that they can be tested without Isaac Sim:
  collect_body_points(stage, root_path, body_names)   once, from the USD stage: the vertices of every robot body, in the
                                                      body's own frame. Collision meshes if the body has some, else the
                                                      visible meshes. Needs pxr (it runs inside the server).
  lowest_per_body(points, names, pos, quat, ...)      at each call, numpy only: put the vertices in the world with the
                                                      simulated pose of each body and measure their height along the peg
                                                      axis, from the peg base (0) towards the gripper (+).
The USD transforms of moving bodies are NOT read at run time (with Fabric they are not updated): only the fixed
relation mesh -> body, which does not change.
"""
import numpy as np


def quat_to_matrix(q):
    """Rotation matrix of a quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def collect_body_points(stage, root_path, body_names):
    """{body name: (N, 3) vertices in the body frame}, {body name: "collision" | "visual"} for the bodies found."""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"no prim at {root_path}")
    wanted = set(body_names)
    bodies = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        name = prim.GetName()
        if name in wanted and name not in bodies and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[name] = prim

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    def to_world(prim):                      # pxr convention: world point = local point @ matrix (row vectors)
        return np.array(xc.GetLocalToWorldTransform(prim), dtype=np.float64)

    def is_collider(prim, body):
        p = prim
        while p.IsValid():
            if p.HasAPI(UsdPhysics.CollisionAPI):
                attr = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr()
                return attr.Get() is not False
            if p.GetPath() == body.GetPath():
                return False
            p = p.GetParent()
        return False

    points, source = {}, {}
    for name, body in bodies.items():
        mb = to_world(body)
        rot = mb[:3, :3] / np.linalg.norm(mb[:3, :3], axis=1, keepdims=True)     # rigid part (a body scale is dropped)
        origin = mb[3, :3]
        coll, vis = [], []
        it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        for p in it:
            if p.GetPath() != body.GetPath() and p.HasAPI(UsdPhysics.RigidBodyAPI):
                it.PruneChildren()           # another body (nested link): its meshes are not ours
                continue
            if not p.IsA(UsdGeom.Mesh):
                continue
            raw = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if raw is None or len(raw) == 0:
                continue
            mp = to_world(p)
            world = np.asarray(raw, dtype=np.float64).reshape(-1, 3) @ mp[:3, :3] + mp[3, :3]   # at the authored pose
            (coll if is_collider(p, body) else vis).append((world - origin) @ rot.T)             # in the body's frame
        if coll:
            points[name], source[name] = np.concatenate(coll), "collision"
        elif vis:
            points[name], source[name] = np.concatenate(vis), "visual"
    return points, source


def lowest_per_body(points, names, pos, quat, origin, axis, lateral_min=0.0):
    """For every body with vertices: its lowest vertex along `axis` from `origin`, among the vertices at least
    `lateral_min` (m) away from the axis line (the ones that cannot go into the hole).

    names / pos (B, 3) / quat (B, 4, w x y z): the simulated bodies, all in the same frame as origin and axis.
    Returns [(name, height m, lateral distance m of that vertex)] sorted by height."""
    origin, axis = np.asarray(origin, dtype=np.float64), np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    out = []
    for i, name in enumerate(names):
        pts = points.get(name)
        if pts is None:
            continue
        world = np.asarray(pos[i], dtype=np.float64) + pts @ quat_to_matrix(quat[i]).T
        d = world - origin
        h = d @ axis
        lat = np.linalg.norm(d - np.outer(h, axis), axis=1)
        ok = lat >= lateral_min
        if not ok.any():
            continue
        k = np.where(ok)[0][np.argmin(h[ok])]
        out.append((name, float(h[k]), float(lat[k])))
    return sorted(out, key=lambda r: r[1])

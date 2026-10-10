"""Convert the waypoint demos (.npz from collect_waypoint_demos.py) into a LeRobot dataset for openpi SFT.

    python scripts/isaaclab/convert_waypoint_demos_to_lerobot.py \
        --demos-dir demos/isaaclab/ForgePegInsert \
        --output-dir demos/isaaclab/lerobot \
        --repo-name expo_ft/forge_peg_insert

Run it in the main env (.venv: it needs `lerobot`). Both folders are given explicitly:
    --demos-dir   folder with the ep_*.npz files (input)
    --output-dir  LeRobot home: the dataset is written to <output-dir>/<repo-name>
                  (the script sets HF_LEROBOT_HOME to it). The SFT job must use the SAME folder as HF_LEROBOT_HOME
                  (lerobot_home in configs/task/isaaclab/peg_insert_forge_sft.yaml, default demos/isaaclab/lerobot).

Layout (same keys as convert_maniskill_to_lerobot.py, so the openpi config LeRobotDROIDDataConfig reads it as is):
    exterior_image_1_left  (224, 224, 3) uint8   exterior camera
    exterior_image_2_left  (224, 224, 3) uint8   black (unused by pi05, the repack transform wants the key)
    wrist_image_left       (224, 224, 3) uint8
    joint_position         (7,)  float32         measured joint angles (rad)
    gripper_position       (1,)  float32         exactly what the server sent as observation (gripper_obs mode)
    actions                (8,)  float32         7 ABSOLUTE joint targets + gripper command (1 = closed)
    task                   str                   the language instruction
Every frame holds the observation BEFORE its action, as saved by collect_waypoint_demos.py.

The actions stay ABSOLUTE here. The openpi config expo_pi05_droid_lora_finetune_sft_joint_state_delta turns the
7 joints into offsets from the chunk-start state at training time (DeltaActions), like the pi05_droid_jointpos weights.

--dry-run only reads and checks the .npz files and prints the report (no lerobot needed).
The report shows the 16-step chunk offsets the model will be trained on and, when the DROID norm_stats.json is
found, the same numbers after openpi's quantile normalization ((x - q01) / (q99 - q01) * 2 - 1).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = (224, 224)
ACTION_HORIZON = 16     # pi05 action chunk, as in the openpi config
DEFAULT_NORM_STATS = ("/network/scratch/m/monganj/cache/openpi/openpi-assets/checkpoints/"
                      "pi05_droid_jointpos/assets/droid/norm_stats.json")
REQUIRED = ("exterior_image", "wrist_image", "joint_position", "gripper_position", "actions")


def load_episode(path):
    """Read one .npz, check shapes / dtypes / values and return clean arrays."""
    d = np.load(path)
    missing = [k for k in REQUIRED if k not in d.files]
    if missing:
        raise ValueError(f"{path.name}: missing arrays {missing}")
    ext, wrist = d["exterior_image"], d["wrist_image"]
    q = d["joint_position"].astype(np.float32)
    grip = d["gripper_position"].astype(np.float32).reshape(-1, 1)
    act = d["actions"].astype(np.float32)
    T = len(act)
    if not (len(ext) == len(wrist) == len(q) == len(grip) == T):
        raise ValueError(f"{path.name}: lengths differ  images {len(ext)}/{len(wrist)}  q {len(q)}  "
                         f"gripper {len(grip)}  actions {T}")
    if ext.shape[1:] != (*IMAGE_SIZE, 3) or wrist.shape[1:] != (*IMAGE_SIZE, 3):
        raise ValueError(f"{path.name}: image shape {ext.shape[1:]} / {wrist.shape[1:]}, expected {(*IMAGE_SIZE, 3)}")
    if ext.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise ValueError(f"{path.name}: images must be uint8, got {ext.dtype} / {wrist.dtype}")
    if q.shape != (T, 7) or act.shape != (T, 8):
        raise ValueError(f"{path.name}: joint_position {q.shape} / actions {act.shape}, expected (T,7) / (T,8)")
    for name, arr in (("joint_position", q), ("gripper_position", grip), ("actions", act)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{path.name}: non-finite values in {name}")
    prompt = str(d["prompt"]) if "prompt" in d.files else None
    return dict(ext=ext, wrist=wrist, q=q, grip=grip, act=act, prompt=prompt)


def chunk_offsets(q, act, horizon=ACTION_HORIZON):
    """What the trainer builds for every frame t: actions[t : t+horizon] (the last action repeated past the end of
    the episode, as LeRobot does), joints minus the joint state at t, gripper kept absolute. Shape (T, horizon, 8)."""
    T = len(q)
    idx = np.minimum(np.arange(T)[:, None] + np.arange(horizon)[None, :], T - 1)
    chunks = act[idx].copy()
    chunks[..., :7] -= q[:, None, :]
    return chunks


def quantile_normalize(x, stats):
    q01, q99 = np.asarray(stats["q01"])[: x.shape[-1]], np.asarray(stats["q99"])[: x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def report(episodes, norm_stats_path):
    q = np.concatenate([e["q"] for e in episodes])
    grip = np.concatenate([e["grip"] for e in episodes])
    off = np.concatenate([chunk_offsets(e["q"], e["act"]) for e in episodes])      # (N, H, 8)
    flat = off.reshape(-1, 8)
    lengths = [len(e["q"]) for e in episodes]
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print(f"\n{len(episodes)} episodes, {sum(lengths)} frames, length min/mean/max = "
          f"{min(lengths)}/{np.mean(lengths):.0f}/{max(lengths)}")
    print(f"gripper observation  min/max = {grip.min():.3f} / {grip.max():.3f}    "
          f"gripper command  min/max = {flat[:, 7].min():.2f} / {flat[:, 7].max():.2f}")
    print("\nchunk offsets (rad, joints 1-7), what the model is trained to output")
    print("  std :", flat[:, :7].std(0))
    print("  q01 :", np.quantile(flat[:, :7], 0.01, axis=0))
    print("  q99 :", np.quantile(flat[:, :7], 0.99, axis=0))

    path = Path(norm_stats_path) if norm_stats_path else None
    if path is None or not path.exists():
        print(f"\n(DROID norm_stats.json not found at {norm_stats_path}: normalized report skipped, "
              "pass --norm-stats PATH to get it)")
        return
    import json
    ns = json.load(open(path))["norm_stats"]
    n_off = quantile_normalize(flat, ns["actions"])
    state = np.concatenate([q, grip], axis=1)
    n_state = quantile_normalize(state, ns["state"])
    print(f"\nafter DROID quantile normalization ({path.name}), what the loss sees")
    print("  actions  std     :", n_off[:, :7].std(0))
    print("  actions  min/max :", n_off[:, :7].min(), "/", n_off[:, :7].max(),
          "   (full scale would be -1 / +1)")
    print("  gripper command  :", n_off[:, 7].min(), "/", n_off[:, 7].max(), "  (closed = 1.0 -> about +1, as in DROID)")
    out = (np.abs(n_state) > 1.0).mean(0)
    print("  state    min     :", n_state.min(0))
    print("  state    max     :", n_state.max(0))
    print("  share of state values outside [-1, 1] (saturate in pi05's 256-bin state tokens):", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos-dir", required=True,
                   help="Input: folder with ep_*.npz (fail_*.npz and index.jsonl are ignored).")
    p.add_argument("--output-dir", required=True,
                   help="Output: LeRobot home. The dataset is written to <output-dir>/<repo-name>. "
                        "Relative paths are relative to the repo root.")
    p.add_argument("--repo-name", default="expo_ft/forge_peg_insert",
                   help="LeRobot repo id: sub-folder of --output-dir.")
    p.add_argument("--task", default=None, help="Language instruction. Default: the one stored in the .npz files.")
    p.add_argument("--fps", type=int, default=15, help="Control frequency (FORGE: 15 Hz, as pi05_droid).")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Replace an existing dataset of the same name.")
    p.add_argument("--dry-run", action="store_true", help="Check the files and print the report only.")
    p.add_argument("--norm-stats", default=DEFAULT_NORM_STATS, help="DROID norm_stats.json, only used by the report.")
    args = p.parse_args()

    demos_dir = Path(args.demos_dir)
    if not demos_dir.is_absolute():
        demos_dir = REPO_ROOT / demos_dir
    files = sorted(demos_dir.glob("ep_*.npz"))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        raise SystemExit(f"no ep_*.npz in {demos_dir}")

    print(f"reading {len(files)} demos from {demos_dir}")
    episodes = [load_episode(f) for f in files]
    prompts = {e["prompt"] for e in episodes} - {None}
    if args.task is None:
        if len(prompts) != 1:
            raise SystemExit(f"expected exactly one prompt in the files, found {sorted(prompts)}: pass --task")
        args.task = prompts.pop()
    print(f"task: {args.task!r}")
    report(episodes, args.norm_stats)
    if args.dry_run:
        print("\ndry run: nothing written.")
        return

    # lerobot reads HF_LEROBOT_HOME when it is imported: set it first.
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    os.environ["HF_LEROBOT_HOME"] = str(output_dir)

    from tqdm import tqdm
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    output_path = HF_LEROBOT_HOME / args.repo_name
    if Path(HF_LEROBOT_HOME).resolve() != output_dir.resolve():
        raise SystemExit(f"lerobot uses {HF_LEROBOT_HOME} instead of {output_dir}: HF_LEROBOT_HOME was already "
                         "imported with another value (unset it in your shell and retry).")
    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_path} already exists: pass --overwrite to replace it.")
        print(f"removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    img = {"dtype": "image", "shape": (*IMAGE_SIZE, 3), "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(
        repo_id=args.repo_name,
        robot_type="panda",
        fps=args.fps,
        features={
            "exterior_image_1_left": dict(img),
            "exterior_image_2_left": dict(img),
            "wrist_image_left": dict(img),
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
    )
    black = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    for e in tqdm(episodes, desc="episodes"):
        for t in range(len(e["act"])):
            dataset.add_frame({
                "exterior_image_1_left": e["ext"][t],
                "exterior_image_2_left": black,
                "wrist_image_left": e["wrist"][t],
                "joint_position": e["q"][t],
                "gripper_position": e["grip"][t],
                "actions": e["act"][t],
                "task": args.task,
            })
        dataset.save_episode()

    print(f"\ndone: {output_path}")
    print(f"train with HF_LEROBOT_HOME={output_dir}  (lerobot_home: {args.output_dir!r} in configs/task/isaaclab/peg_insert_forge_sft.yaml)")
    print(f"episodes: {dataset.num_episodes}, frames: {dataset.num_frames}  (expected {len(episodes)} / "
          f"{sum(len(e['act']) for e in episodes)})")
    if dataset.num_episodes != len(episodes):
        sys.exit("episode count mismatch")


if __name__ == "__main__":
    main()

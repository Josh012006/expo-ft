"""Pick task (configs/task/pick.py) driven in joint-position mode, for pi05_droid_jointpos.

Server side (robot machine):  python -m client.run_client --config-task-path configs/task/pick_jointpos.py
"""
from configs.task import pick, real_jointpos_base


def get_config():
    config = pick.get_config()                    # env class, bounds, reset pose, detector, prompt
    base = real_jointpos_base.get_config()
    for key in real_jointpos_base.JOINT_KEYS:
        config[key] = base[key]
    # pick.py: 80 steps at 10 Hz = 8 s of episode; same duration at 15 Hz.
    config.auto_reset_steps = 120
    return config

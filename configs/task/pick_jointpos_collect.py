"""Pick task, demonstration collection with the SpaceMouse, joint-position labels.

The robot is still TELEOPERATED in cartesian_velocity (the SpaceMouse produces 6D twists), but
DROID's create_action_dict also computes the matching absolute joint targets for every step
(franka/robot.py: action_dict["joint_position"], from the IK solver), and that is what gets
saved as the label (label_action_space below, read by scripts/convert_droid_data_to_lerobot.py).

    python client/collect_data.py --task_config configs/task/pick_jointpos_collect.py ...
"""
from configs.task import pick


def get_config():
    config = pick.get_config()                    # cartesian_velocity teleop, as before
    config.control_hz = 15                        # pi05_droid_jointpos nominal rate. NOTE: the
                                                  # SpaceMouse scaling (collect_max_*) was tuned at
                                                  # 10 Hz: retune if the motion feels too fast/slow.
    # Labels: absolute joint targets + gripper command (0 = open, 1 = closed).
    config.label_action_space = "joint_position"
    config.label_gripper_action_space = "position"
    config.use_cartesian_state = False
    # Saved state must use the same gripper convention as the labels (0 = open).
    config.invert_gripper_observation = True
    return config

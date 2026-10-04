"""CPU config-invariant tests for the Seeed Studio StackForce Mini port (fork-local)."""

import mujoco
import numpy as np

from mjlab.tasks.registry import list_tasks
from mjlab_microduck.robot.stackforce.stackforce_constants import get_stackforce_spec
from mjlab_microduck.tasks.stackforce_velocity_env_cfg import make_stackforce_velocity_env_cfg


def test_stackforce_spec_loads():
    model = get_stackforce_spec().compile()
    total = float(sum(model.body_mass))
    assert 0.52 <= total <= 0.56, f"StackForce total mass {total:.3f} kg off-spec (Seeed: 0.540 kg)"

    # 4 hip servos + 2 continuous wheel joints = 6 actuated hinges (+ 4 passive linkage knees = 10 hinges)
    hinges = [n for n in range(model.njnt) if model.jnt_type[n] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(hinges) == 10
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinges]
    assert {"left_front_hip", "left_rear_hip", "right_front_hip", "right_rear_hip",
            "left_front_knee", "left_rear_knee", "right_front_knee", "right_rear_knee",
            "left_wheel_joint", "right_wheel_joint"} == set(names)

    # 2 equality constraints for 5-bar linkage loop closure
    assert model.neq == 2

    # 4 position servos + 2 wheel motors = 6 actuators
    assert model.nu == 6

    # Sensors
    site_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(model.nsite)}
    assert {"imu", "left_wheel_center", "right_wheel_center"} <= site_names

    # 100 simulation steps: assert zero numerical drift or NaN in closed loop
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert not np.isnan(data.qpos).any(), "NaN in qpos during 100 step simulation"
    assert not np.isnan(data.qvel).any(), "NaN in qvel during 100 step simulation"


def test_stackforce_velocity_cfg_builds():
    cfg = make_stackforce_velocity_env_cfg()
    assert "joint_pos" in cfg.actions
    assert "wheel_effort" in cfg.actions

    action_pos = cfg.actions["joint_pos"]
    assert len(action_pos.actuator_names) == 4

    action_wheel = cfg.actions["wheel_effort"]
    assert len(action_wheel.actuator_names) == 2

    positive = ("pose", "upright", "track_linear_velocity", "track_angular_velocity")
    for name in positive:
        assert cfg.rewards[name].weight is not None and cfg.rewards[name].weight >= 0, name

    # Reward sign-convention rule: penalizers must be <= 0
    for name in ("action_rate_l2", "dof_pos_limits", "body_ang_vel"):
        w = cfg.rewards[name].weight
        assert w is None or w <= 0, f"penalty {name} has positive weight {w}"

    if hasattr(cfg.commands["twist"].ranges, "lin_vel_x"):
        lo, hi = cfg.commands["twist"].ranges.lin_vel_x
        assert hi >= 0.50 and lo <= -0.10


def test_stackforce_task_registered():
    assert "Mjlab-Velocity-Flat-StackForce" in list_tasks()

"""CPU config-invariant tests for the PU Robot port (fork-local)."""

import mujoco

from mjlab.tasks.registry import list_tasks
from mjlab_microduck.robot.pu.pu_constants import get_pu_spec
from mjlab_microduck.robot.xgo.xgo_constants import get_xgo_spec
from mjlab_microduck.tasks.pu_velocity_env_cfg import make_pu_velocity_env_cfg


def test_pu_spec_loads():
    model = get_pu_spec().compile()
    total = float(sum(model.body_mass))
    assert 0.30 <= total <= 0.36, f"PU total mass {total:.3f} kg off-spec (ELECFREAKS: 0.300 + 0.020 stick)"
    hinges = [n for n in range(model.njnt) if model.jnt_type[n] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(hinges) == 6
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinges]
    assert {"left_leg_stride", "right_leg_stride", "left_foot_tilt", "right_foot_tilt", "head_yaw", "head_pitch"} == set(names)
    geom_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)}
    assert {"left_foot_collision", "right_foot_collision"} <= geom_names
    site_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(model.nsite)}
    assert {"left_foot", "right_foot", "imu"} <= site_names
    assert model.nu == 6


def test_pu_velocity_cfg_builds():
    cfg = make_pu_velocity_env_cfg()
    action = cfg.actions["joint_pos"]
    assert len(action.actuator_names) == 4
    positive = ("pose", "upright", "track_linear_velocity", "track_angular_velocity")
    for name in positive:
        assert cfg.rewards[name].weight is not None and cfg.rewards[name].weight >= 0, name
    # Reward sign-convention rule (playbook 5.1): penalizers must be <= 0.
    for name in ("foot_slip", "action_rate_l2", "dof_pos_limits", "body_ang_vel"):
        w = cfg.rewards[name].weight
        assert w is None or w <= 0, f"penalty {name} has positive weight {w}"
    if hasattr(cfg.commands["twist"].ranges, "lin_vel_x"):
        lo, hi = cfg.commands["twist"].ranges.lin_vel_x
        assert hi <= 0.30 and lo >= -0.10


def test_pu_task_registered():
    assert "Mjlab-Velocity-Flat-PURobot" in list_tasks()


def test_xgo_spec_loads():
    model = get_xgo_spec().compile()
    total = float(sum(model.body_mass))
    assert 0.50 <= total <= 0.65, f"XGO total mass {total:.3f} kg off-spec (ELECFREAKS: 0.600)"

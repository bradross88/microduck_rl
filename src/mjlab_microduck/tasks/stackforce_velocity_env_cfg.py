"""Seeed Studio StackForce Mini velocity (wheeled-legged) environment.

Mjlab-Velocity-Flat-StackForce: forward/yaw velocity tracking on flat terrain,
hybrid 6-DOF action space:
  - 4x leg servos (JointPositionAction, scale 0.5) for height, roll, and pitch pose
  - 2x brushless wheel motors (JointEffortAction, scale 0.25) for balance and traction
Closed 5-bar linkage equality constraints solved natively in MuJoCo.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.robot.stackforce.stackforce_constants import (
    STACKFORCE_LEG_ACTUATOR_NAMES,
    STACKFORCE_ROBOT_CFG,
    STACKFORCE_WHEEL_ACTUATOR_NAMES,
)

STACKFORCE_LIN_VEL_RANGE = (-0.2, 0.8)   # m/s: fast wheeled rover
STACKFORCE_ANG_VEL_RANGE = (-2.0, 2.0)   # rad/s: responsive differential turning

HIP_STANDING_STD = {"^(left|right)_(front|rear)_hip$": 0.25}
HIP_WALKING_STD = {"^(left|right)_(front|rear)_hip$": 0.35}


def make_stackforce_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": STACKFORCE_ROBOT_CFG}

    wheels_ground_cfg = ContactSensorCfg(
        name="wheels_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^(left_wheel_collision|right_wheel_collision)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=False,
    )
    cfg.scene.sensors = (wheels_ground_cfg,)
    cfg.viewer.body_name = "trunk_base"

    # Action term 1: 4x 5-bar hip position servos
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.actuator_names = STACKFORCE_LEG_ACTUATOR_NAMES
    joint_pos_action.scale = 0.5
    joint_pos_action.preserve_order = True

    # Action term 2: 2x brushless wheel motors
    cfg.actions["wheel_effort"] = JointEffortActionCfg(
        entity_name="robot",
        actuator_names=STACKFORCE_WHEEL_ACTUATOR_NAMES,
        scale=0.25,
        preserve_order=True,
    )

    pose = cfg.rewards["pose"]
    pose.params["std_standing"] = HIP_STANDING_STD
    pose.params["std_walking"] = HIP_WALKING_STD
    pose.params["std_running"] = HIP_WALKING_STD
    pose.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r"^(left|right)_(front|rear)_hip$",))
    pose.weight = 1.0

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # Wheeled robot: remove foot clearance / slip terms intended for stepping feet
    cfg.rewards.pop("foot_clearance", None)
    cfg.rewards.pop("foot_slip", None)

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    cmd = cfg.commands["twist"]
    ranges = cmd.ranges
    for attr, val in {"lin_vel_x": STACKFORCE_LIN_VEL_RANGE, "lin_vel_y": (0.0, 0.0), "ang_vel_z": STACKFORCE_ANG_VEL_RANGE}.items():
        if hasattr(ranges, attr):
            setattr(ranges, attr, val)
    cmd.resampling_time_range = (2.0, 5.0)

    # Event tuning for 540g wheeled robot (standing height ~0.16 m)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.155, 0.165)
    if "ranges" in cfg.events["foot_friction"].params:
        cfg.events["foot_friction"].params["ranges"] = (0.8, 1.4)
    push = cfg.events.get("push_robot")
    if push is not None:
        push.params["velocity_range"] = {
            "x": (-0.25, 0.25), "y": (-0.25, 0.25), "z": (0.0, 0.0),
            "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.4, 0.4),
        }

    if play and "push_robot" in cfg.events:
        cfg.events["push_robot"].interval_range_s = (4.0, 8.0)

    return cfg

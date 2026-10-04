"""PU Robot velocity (walking) environment.

Mjlab-Velocity-Flat-PURobot: forward/yaw velocity tracking on the flat
terrain, legs-only policy (4 actuators), head centered as passive
counterweight. Recipe mirrors the Microduck velocity task at PU scale.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.robot.pu.pu_constants import PU_LEG_ACTUATOR_NAMES, PU_ROBOT_CFG

# PU walks much slower than the duck: top ~0.25 m/s at a scale that fits a
# 300 g biped with 1.5 kg.cm servos.
PU_LIN_VEL_RANGE = (-0.05, 0.25)
PU_ANG_VEL_RANGE = (-1.0, 1.0)

STANDING_STD = {"^(left|right)_leg_stride$": 0.20, "^(left|right)_foot_tilt$": 0.12}
WALKING_STD = {"^(left|right)_leg_stride$": 0.35, "^(left|right)_foot_tilt$": 0.20}

SITE_NAMES = ("left_foot", "right_foot")


def make_pu_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": PU_ROBOT_CFG}

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in SITE_NAMES),
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=2),
        ray_alignment="yaw",
        max_distance=0.5,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )
    cfg.scene.sensors = (feet_ground_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.actuator_names = PU_LEG_ACTUATOR_NAMES
    joint_pos_action.scale = 0.4
    joint_pos_action.preserve_order = True

    pose = cfg.rewards["pose"]
    pose.params["std_standing"] = STANDING_STD
    pose.params["std_walking"] = WALKING_STD
    pose.params["std_running"] = WALKING_STD
    pose.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r"^(?!head_).*",))
    pose.weight = 1.0

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = SITE_NAMES
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # Command ranges at PU scale.
    cmd = cfg.commands["twist"]
    ranges = cmd.ranges
    for attr, val in {"lin_vel_x": PU_LIN_VEL_RANGE, "lin_vel_y": (0.0, 0.0), "ang_vel_z": PU_ANG_VEL_RANGE}.items():
        if hasattr(ranges, attr):
            setattr(ranges, attr, val)
    cmd.resampling_time_range = (2.0, 5.0)

    # Event tuning: reset above the nominal STAND height, grippier footpads,
    # gentler pushes than the duck (PU's CoM margin is smaller).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.090, 0.098)
    if "ranges" in cfg.events["foot_friction"].params:
        cfg.events["foot_friction"].params["ranges"] = (0.6, 1.2)
    push = cfg.events.get("push_robot")
    if push is not None:
        push.params["velocity_range"] = {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (0.0, 0.0),
                                         "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3)}

    if play and "push_robot" in cfg.events:
        cfg.events["push_robot"].interval_range_s = (4.0, 8.0)

    return cfg

"""Seeed Studio StackForce Mini simulation constants.

Spec-anchored 540 g model with a closed 5-bar parallel linkage per leg.
Actuation: 4x hip position servos + 2x continuous brushless wheel motors.
Equality constraints close the 5-bar loop at the wheel axle.
"""

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_ROBOT_DIR = Path(__file__).parent
STACKFORCE_XML = _ROBOT_DIR / "stackforce.xml"
assert STACKFORCE_XML.exists(), f"XML not found: {STACKFORCE_XML}"


STACKFORCE_LEG_ACTUATOR_NAMES = (
    r"^left_front_hip_ctrl$",
    r"^left_rear_hip_ctrl$",
    r"^right_front_hip_ctrl$",
    r"^right_rear_hip_ctrl$",
)

STACKFORCE_WHEEL_ACTUATOR_NAMES = (
    r"^left_wheel_ctrl$",
    r"^right_wheel_ctrl$",
)

def get_stackforce_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(STACKFORCE_XML))


STACKFORCE_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

STACKFORCE_WHEELS_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_wheel_collision$",),
    condim=3,
    priority=1,
    friction=(1.2,),
)

_sf_actuator_kwargs = dict(
    target_names_expr=(r".*",),
    delay_min_lag=1,
    delay_max_lag=2,
)
stackforce_actuators = XmlActuatorCfg(**_sf_actuator_kwargs)

STACKFORCE_ROBOT_CFG = EntityCfg(
    spec_fn=get_stackforce_spec,
    init_state=STACKFORCE_HOME_FRAME,
    collisions=(STACKFORCE_WHEELS_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(stackforce_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


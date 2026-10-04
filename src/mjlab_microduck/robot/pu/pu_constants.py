"""PU Robot (ELECFREAKS EF08449, Fleet Device 8) simulation constants.

First-order spec-anchored model: geometry from ELECFREAKS sheets
(300 g, 120x110x160 mm, 6 DOF) and the NVS kinematic trims on the real
chassis. Actuators are ideal XML position drives with the 1.5 kg.cm servo
torque ceiling (forcerange +/-0.147 Nm) and 1-3 step command delay.
BENCH ACTUATOR ID is the open TODO before sim2real: these micro servos have
no published BAM model, so the XL330 BAM parameters do not transfer.
"""

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_ROBOT_DIR = Path(__file__).parent
PU_XML = _ROBOT_DIR / "pu.xml"
assert PU_XML.exists(), f"XML not found: {PU_XML}"


def get_pu_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(PU_XML))


PU_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

PU_FEET_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot_collision$",),
    condim=3,
    priority=1,
    friction=(1.0,),
)

# XML-driven gains (kp / dampratio / forcerange live in pu.xml), plus the
# I2C/Pa.Hub command latency band (pausable micro-stepping).
_pu_actuator_kwargs = dict(
    target_names_expr=(r".*",),
    delay_min_lag=1,
    delay_max_lag=3,
)
pu_actuators = XmlActuatorCfg(**_pu_actuator_kwargs)

PU_ROBOT_CFG = EntityCfg(
    spec_fn=get_pu_spec,
    init_state=PU_HOME_FRAME,
    collisions=(PU_FEET_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(pu_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Policy actuates the 4 leg servos only; head servos stay centered and act
# as passive counterweight until the head-counterweight gait term exists.
PU_LEG_ACTUATOR_NAMES = (r"^left_leg_stride_ctrl$", r"^left_foot_tilt_ctrl$",
                         r"^right_leg_stride_ctrl$", r"^right_foot_tilt_ctrl$")

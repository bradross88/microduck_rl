"""XGO-Rider simulation constants (kit deferred - BL-173).

Robot cfg mirrors the PU pattern; wheel motors are torque (motor) actuators.
No velocity task registered yet: RL work waits until the kit is in hand for
measurement, and the first integration is HIGH-LEVEL URCI (onboard ESP32
balances; we send 0x30/0x32/0x35/0x36 setpoints) which needs no policy.
"""

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_ROBOT_DIR = Path(__file__).parent
XGO_XML = _ROBOT_DIR / "xgo.xml"
assert XGO_XML.exists(), f"XML not found: {XGO_XML}"


def get_xgo_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(XGO_XML))


XGO_HOME_FRAME = EntityCfg.InitialStateCfg(joint_pos={r"^(?!left_wheel|right_wheel).*": 0.0}, joint_vel={".*": 0.0})

xgo_actuators = XmlActuatorCfg(target_names_expr=(r".*_ctrl$",), delay_min_lag=1, delay_max_lag=3)
xgo_wheel_motors = XmlActuatorCfg(target_names_expr=(r"^.*_wheel_motor$",), delay_min_lag=0, delay_max_lag=1)

XGO_ROBOT_CFG = EntityCfg(
    spec_fn=get_xgo_spec,
    init_state=XGO_HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(xgo_actuators, xgo_wheel_motors),
        soft_joint_pos_limit_factor=0.9,
    ),
)

"""Optional tip-space metrics through MuJoCo forward kinematics."""
import os
from pathlib import Path

import numpy as np

from trajlens.io import TrajError

DEFAULT_XML = "~/UR3_ROS2_PICK_AND_PLACE/ur_rl_training/models/ur3/ur3.xml"


def find_model(xml=None):
    """Model path from the argument, TRAJLENS_UR3_XML, a TRAJLENS_MODELS directory, or the default location."""
    cands = [xml, os.environ.get("TRAJLENS_UR3_XML")]
    if os.environ.get("TRAJLENS_MODELS"):
        cands.append(str(Path(os.environ["TRAJLENS_MODELS"]) / "ur3" / "ur3.xml"))
    cands.append(DEFAULT_XML)
    for c in cands:
        if c and Path(c).expanduser().is_file():
            return str(Path(c).expanduser())
    raise TrajError("MuJoCo model not found; pass --xml or set TRAJLENS_UR3_XML")


def tip_positions(tr, xml=None, site="attachment_site", tool_length=0.0):
    """(N,3) positions of the tool tip for every sample; joints are matched to the model by name."""
    try:
        import mujoco
    except ImportError:
        raise TrajError("tip-space metrics need mujoco (pip install trajlens[fk])")
    model = mujoco.MjModel.from_xml_path(find_model(xml))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        raise TrajError(f"site {site!r} not in the model")
    adr = []
    for n in tr.names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise TrajError(f"joint {n!r} not in the model")
        adr.append(model.jnt_qposadr[jid])
    out = np.zeros((len(tr.t), 3))
    for i, row in enumerate(tr.q):
        data.qpos[adr] = row
        mujoco.mj_kinematics(model, data)
        out[i] = data.site_xpos[sid] + data.site_xmat[sid].reshape(3, 3)[:, 2] * tool_length
    return out

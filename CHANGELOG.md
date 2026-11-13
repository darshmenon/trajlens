# Changelog

## 0.1.0

- First release.
- Per-joint max velocity, acceleration and jerk, normalized jerk, SPARC, limit margins from a URDF, time ratio against the velocity-limit bound, joint-space path length.
- Optional tip-space path length and speed through MuJoCo forward kinematics (UR3).
- Tracking error and lag between a planned and an executed trajectory.
- Findings with codes and hints: LIMIT_EXCEEDED, LIMIT_NEAR, VEL_OVER, ACC_OVER, JERK_OVER, JERK_SPIKE, ROUGH, STOP_START, SAMPLING_GAP, SLOW_PROFILE, FEW_SAMPLES, TRACKING_LAG, TRACKING_ERROR, TRACKING_FINAL, EXEC_ENDS_EARLY.
- Inputs: CSV, JSON joint_trajectory, JSON lines recordings, and rosbag2 when `rosbag2_py` is importable.
- CLI: `trajlens analyze`, `trajlens compare`, `trajlens track`, with PNG and JSON output.

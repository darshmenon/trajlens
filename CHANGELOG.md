# Changelog

## Unreleased
- Optional C++ core (`trajlens._fast`, built by `pip install .` when a compiler is present; pybind11 is a build requirement): the non-uniform finite-difference gradient behind velocity, acceleration, jerk and speed. On a 500,000 sample, 6 joint recording, `derivatives` took 0.69 s instead of 1.59 s (2.3x) and a full `analyze` 5.3 s instead of 8.0 s (1.5x). The rest is the zero-padded FFT in `sparc`, which numpy already runs in C, so it was left alone. `TRAJLENS_NO_CPP=1` forces numpy; results match to 1e-8.

## 0.1.0

- First release.
- Per-joint max velocity, acceleration and jerk, normalized jerk, SPARC, limit margins from a URDF, time ratio against the velocity-limit bound, joint-space path length.
- Optional tip-space path length and speed through MuJoCo forward kinematics (UR3).
- Tracking error and lag between a planned and an executed trajectory.
- Findings with codes and hints: LIMIT_EXCEEDED, LIMIT_NEAR, VEL_OVER, ACC_OVER, JERK_OVER, JERK_SPIKE, ROUGH, STOP_START, SAMPLING_GAP, SLOW_PROFILE, FEW_SAMPLES, TRACKING_LAG, TRACKING_ERROR, TRACKING_FINAL, EXEC_ENDS_EARLY.
- Inputs: CSV, JSON joint_trajectory, JSON lines recordings, and rosbag2 when `rosbag2_py` is importable.
- CLI: `trajlens analyze`, `trajlens compare`, `trajlens track`, with PNG and JSON output.

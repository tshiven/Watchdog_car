"""
Distance estimation module.

Responsible for estimating the real-world distance to the tracked target,
to eventually support range-aware behavior (e.g. stopping distance, IMU
fusion for stabilization).

Will eventually handle:
- Estimating distance from bounding box size and camera calibration data
- Possibly fusing with IMU or stereo/depth data in the future

Not yet implemented.
"""

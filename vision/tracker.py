"""
Target tracking module.

Responsible for tracking the locked-on target across subsequent frames and
computing its positional error relative to the center of the camera frame,
which will later drive pan/tilt servo control.

Will eventually handle:
- Frame-to-frame tracking of the selected target (e.g. via re-detection or
  a lightweight tracker)
- Computing (dx, dy) pixel error from frame center
- Reporting loss of track so the system can re-search

Not yet implemented.
"""

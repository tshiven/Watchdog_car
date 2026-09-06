"""
HSV color range tuning tool.

Responsible for providing an interactive way (e.g. OpenCV trackbars) to
find HSV lower/upper bounds for a given color, to be used by
vision/color_matcher.py.

Will eventually handle:
- Live camera preview with adjustable HSV sliders
- Saving chosen ranges to calibration_data/ for reuse

Not yet implemented.
"""

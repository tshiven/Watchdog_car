"""
Object detection module.

Responsible for running YOLO inference on a frame and returning candidate
detections (bounding boxes, class labels, confidence scores) for objects
matching the class requested by the user (e.g. "cat", "bottle", "person").

Will eventually handle:
- Loading and running a YOLO model (via ultralytics)
- Filtering detections by requested object class
- Returning a clean list of candidate detections to downstream modules

Not yet implemented.
"""

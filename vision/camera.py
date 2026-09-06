"""
Camera capture module.

Responsible for opening a video source (webcam, CSI camera, or video file)
and yielding frames for the rest of the vision pipeline to consume.

Will eventually handle:
- Camera initialization and configuration (resolution, FPS)
- Frame grabbing loop / generator interface
- Basic error handling for dropped frames or disconnects

Not yet implemented.
"""

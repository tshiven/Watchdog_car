"""
Color / visual attribute matching module.

Responsible for evaluating how well a detected object's appearance matches
a requested visual attribute, such as a color (e.g. "orange cat" vs a
non-orange cat detected in the same frame).

Will eventually handle:
- Extracting a region of interest from a bounding box
- Converting to HSV and comparing against target color ranges
- Producing a match score usable for ranking candidates

Not yet implemented.
"""

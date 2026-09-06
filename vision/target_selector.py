"""
Target selection module.

Responsible for choosing the single best-matching target when multiple
candidate detections are present, combining detection confidence with
attribute match scores (e.g. color) from color_matcher.

Will eventually handle:
- Ranking candidate detections by combined score
- Selecting and locking onto the best match
- Handling the case where no candidate meets a minimum confidence threshold

Not yet implemented.
"""

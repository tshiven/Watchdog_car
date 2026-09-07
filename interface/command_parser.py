"""
Natural-language command parsing module.

Responsible for turning user input (typed or eventually spoken) into a
structured search request, e.g. "Find the orange cat" ->
{"object": "cat", "attribute": "orange"}.

Will eventually handle:
- Parsing free-form commands like "Track the person wearing a blue shirt"
- Extracting object class and optional visual attributes
- Falling back to simple prompt-based Q&A (object, then attribute)

Not yet implemented.
"""
# Common YOLO/COCO object classes that the parser can recognize.
COCO_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}

SUPPORTED_COLORS = {
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
    "brown",
}

def parse_command(command: str) -> dict[str, str | None]:
    """Parse a user command into a target class and optional target color."""
    normalized_command = command.strip().lower()
    words = normalized_command.split()

    target_color = None
    for color in SUPPORTED_COLORS:
        if color in words:
            target_color = color
            break

    # Remove the color from consideration when looking for the object class.
    class_words = [word for word in words if word != target_color]

    target_class = None

    # Check multi-word COCO classes first.
    for object_class in sorted(
        COCO_CLASSES,
        key=lambda name: len(name.split()),
        reverse=True,
    ):
        object_class_words = object_class.split()

        for index in range(len(class_words) - len(object_class_words) + 1):
            if class_words[
                index:index + len(object_class_words)
            ] == object_class_words:
                target_class = object_class
                break

        if target_class is not None:
            break

    return {
        "target_class": target_class,
        "target_color": target_color,
    }
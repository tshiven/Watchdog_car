"""
Camera calibration tool.

Responsible for computing camera intrinsics (and later, parameters needed
for distance estimation) from calibration images, e.g. a checkerboard
pattern.

Will eventually handle:
- Capturing or loading calibration images
- Running OpenCV camera calibration routines
- Saving calibration results to calibration_data/

Not yet implemented.
"""

"""
Camera calibration tool.

Calculates focal length from a known object width, distance, and observed
pixel width, then saves the calibration information as JSON.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).parent / "calibration_data"


def calculate_focal_length(
    pixel_width: float,
    known_distance_cm: float,
    real_width_cm: float,
) -> float | None:
    """Calculate focal length in pixels using a known calibration target."""
    if pixel_width <= 0:
        return None

    if known_distance_cm <= 0:
        return None

    if real_width_cm <= 0:
        return None

    return (pixel_width * known_distance_cm) / real_width_cm


def save_calibration(
    output_path: Path,
    camera_index: int,
    camera_name: str,
    resolution_width: int,
    resolution_height: int,
    focal_length_px: float,
    calibration_distance_cm: float,
    target_real_width_cm: float,
) -> None:
    """Save camera calibration information to a JSON file."""
    calibration_data = {
        "camera_index": camera_index,
        "camera_name": camera_name,
        "resolution": {
            "width": resolution_width,
            "height": resolution_height,
        },
        "focal_length_px": focal_length_px,
        "calibration_distance_cm": calibration_distance_cm,
        "target_real_width_cm": target_real_width_cm,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(calibration_data, file, indent=4)


def main() -> None:
    """Run the camera calibration command-line tool."""
    parser = argparse.ArgumentParser(
        description="Calculate and save Watchdog camera calibration data."
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        required=True,
        help="Camera index used during calibration.",
    )
    parser.add_argument(
        "--camera-name",
        required=True,
        help="Human-readable camera name.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Camera resolution width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Camera resolution height in pixels.",
    )
    parser.add_argument(
        "--distance-cm",
        type=float,
        required=True,
        help="Known distance from camera to target in centimeters.",
    )
    parser.add_argument(
        "--real-width-cm",
        type=float,
        required=True,
        help="Known real-world width of the calibration target in centimeters.",
    )
    parser.add_argument(
        "--pixel-width",
        type=float,
        required=True,
        help="Observed target width in pixels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "calibration.json",
        help="Path where calibration JSON should be saved.",
    )

    args = parser.parse_args()

    focal_length_px = calculate_focal_length(
        pixel_width=args.pixel_width,
        known_distance_cm=args.distance_cm,
        real_width_cm=args.real_width_cm,
    )

    if focal_length_px is None:
        parser.error(
            "pixel width, distance, and real-world width must all be positive."
        )

    save_calibration(
        output_path=args.output,
        camera_index=args.camera_index,
        camera_name=args.camera_name,
        resolution_width=args.width,
        resolution_height=args.height,
        focal_length_px=focal_length_px,
        calibration_distance_cm=args.distance_cm,
        target_real_width_cm=args.real_width_cm,
    )

    print(f"Focal length: {focal_length_px:.2f} px")
    print(f"Calibration saved to: {args.output}")
    print(
        "Note: calibration is resolution-specific. "
        "Recalibrate if the camera resolution changes."
    )


if __name__ == "__main__":
    main()

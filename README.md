# Watchdog
<p align="center">
  <img src="https://github.com/user-attachments/assets/47b81b6c-373d-4211-b505-6664327e55d9" width="400">
</p>
> **Status: under active development.** The features described below are
> the project's design goals, not a description of what currently works.
> See [Current Status](#current-status) for what actually exists today.

## Overview

Watchdog is a pan-tilt camera system that finds and tracks a
user-specified object. The user tells it what to look for (and optionally
a visual attribute like color), the system detects candidates with YOLO,
picks the best match, and tracks it — computing how far the target is from
the center of the camera frame so that error can eventually drive pan/tilt
servos on an STM32.

Example interaction (target behavior):

```
What would you like to find?
> cat

What color?
> orange

Searching for: orange cat
```

Longer-term, natural-language commands such as "Find the orange cat" or
"Track the person wearing a blue shirt" are planned as well.

## Current Status

This project is in the initial scaffolding stage. The folder/module
structure exists, but detection, color matching, tracking, serial
communication, and firmware are **not yet implemented**. Nothing in this
repository should be assumed to run end-to-end yet.

## System Architecture

Planned high-level flow:

```
User input (CLI) -> command parsing -> YOLO detection -> attribute matching
   -> target selection -> tracking -> center error -> serial link -> STM32
```

### Vision Pipeline

- Capture frames from a camera (`vision/camera.py`)
- Detect candidate objects with YOLO (`vision/detector.py`)
- Score candidates against a requested attribute like color
  (`vision/color_matcher.py`)
- Select the best-matching target (`vision/target_selector.py`)
- Track the target across frames and compute center error
  (`vision/tracker.py`)
- Estimate distance to the target (`vision/distance_estimation.py`)

### Hardware

- Pan-tilt camera mount
- STM32 microcontroller for servo control (planned)
- IMU for stabilization (planned)

### Software

- Python vision pipeline (OpenCV, NumPy, Ultralytics YOLO)
- CLI for interactive object/attribute selection
- Serial protocol for host-to-STM32 communication

## Build Stages

1. Project scaffolding (this stage)
2. CLI + command parsing
3. YOLO-based object detection
4. Color/attribute matching and target selection
5. Tracking and center-error computation
6. Serial communication with STM32
7. STM32 firmware for pan/tilt servo control
8. IMU-based stabilization and distance estimation

## Validation Metrics

To be defined as each stage is implemented (e.g. detection accuracy,
tracking stability, center-error latency). Not yet available.

## Demo

No demo yet. Screenshots and video will be added to `docs/demo/` as the
project progresses.

## Repository Structure

```
Watchdog_car/
├── vision/                # Detection, color matching, target selection, tracking
├── interface/              # CLI and command parsing
├── communication/          # Host <-> STM32 serial protocol
├── firmware/               # STM32 firmware (planned)
├── calibration/            # HSV tuning and camera calibration tools
├── data/                   # Logs and measurements
├── tests/                  # Unit tests
├── docs/                   # Diagrams, screenshots, demo assets
├── scripts/                # Entry point script(s)
├── requirements.txt
├── .gitignore
└── LICENSE
```

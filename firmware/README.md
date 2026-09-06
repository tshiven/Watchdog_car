# Firmware

Placeholder for the STM32 firmware that will drive the pan/tilt servos
based on center-error data received from the vision system over serial.

Not yet implemented. Planned responsibilities:

- Receiving packets from the host (see `communication/packet_utils.py`)
- Running a control loop (e.g. PID) to convert center error into pan/tilt
  servo commands
- Later: incorporating IMU data for stabilization

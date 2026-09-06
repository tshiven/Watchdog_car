"""
Serial communication protocol module.

Responsible for sending tracking data (e.g. center error, lock status) from
the vision system to the STM32 over a serial link, and receiving any status
messages back.

Will eventually handle:
- Opening and managing a pyserial connection to the STM32
- Sending encoded packets (see packet_utils.py) at a fixed rate
- Reading and interpreting acknowledgements or status from the STM32

Not yet implemented.
"""

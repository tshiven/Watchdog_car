"""
Packet encoding/decoding utilities.

Responsible for defining the wire format used between the vision system and
the STM32, and providing helper functions to pack/unpack that format.

Will eventually handle:
- Defining a simple, fixed-size packet structure (e.g. error values, flags)
- Encoding Python values into bytes for serial_protocol.py to send
- Decoding bytes received from the STM32 back into Python values

Not yet implemented.
"""

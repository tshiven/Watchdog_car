"""
Command-line interface module.

Responsible for the interactive user-facing entry point: asking what object
to find, optionally asking for a visual attribute, and reporting the
resulting search target back to the user.

Will eventually handle:
- Prompting the user (e.g. "What would you like to find?", "What color?")
- Passing the resulting query to the command parser / vision pipeline
- Displaying status updates (searching, locked on, lost target)

Not yet implemented.
"""

"""
Command-line interface module.

Provides the interactive user-facing entry point for creating a Watchdog
target request.
"""

from interface.command_parser import parse_command


def main() -> None:
    """Run the interactive Watchdog command-line interface."""
    print("WATCHDOG")
    print()

    command = input("What would you like me to find?\n> ")

    result = parse_command(command)

    target_class = result["target_class"]
    target_color = result["target_color"]

    # If the user only entered an object, ask for an optional color.
    if target_class is not None and target_color is None:
        color_input = input("\nOptional color?\n> ").strip().lower()

        if color_input:
            color_result = parse_command(color_input)
            target_color = color_result["target_color"]

    if target_class is None:
        print("\nI couldn't identify an object to find.")
        return

    if target_color is not None:
        print(f"\nTarget: {target_color} {target_class}")
    else:
        print(f"\nTarget: {target_class}")

    print("Starting camera...")


if __name__ == "__main__":
    main()
"""Core functionality for the example package.

This module provides basic greeting functionality to demonstrate
package structure and testing patterns.
"""


def greet(name: str) -> str:
    """Generate a greeting message.

    Args:
        name: The name to greet. Must be a non-empty string.

    Returns:
        A personalized greeting message.

    Raises:
        ValueError: If name is empty or contains only whitespace.

    Examples:
        >>> greet("World")
        'Hello, World!'
        >>> greet("Alice")
        'Hello, Alice!'
    """
    if not name or not name.strip():
        raise ValueError("Name cannot be empty")

    return f"Hello, {name}!"


def greet_multiple(names: list[str]) -> list[str]:
    """Generate greeting messages for multiple names.

    Args:
        names: List of names to greet.

    Returns:
        List of greeting messages corresponding to each name.

    Examples:
        >>> greet_multiple(["Alice", "Bob"])
        ['Hello, Alice!', 'Hello, Bob!']
    """
    return [greet(name) for name in names]


class Greeter:
    """A class-based greeter with customizable prefix.

    Attributes:
        prefix: The greeting prefix (default: "Hello")

    Examples:
        >>> greeter = Greeter()
        >>> greeter.greet("World")
        'Hello, World!'
        >>> greeter = Greeter(prefix="Hi")
        >>> greeter.greet("Alice")
        'Hi, Alice!'
    """

    def __init__(self, prefix: str = "Hello") -> None:
        """Initialize the greeter.

        Args:
            prefix: The greeting prefix to use.
        """
        self.prefix = prefix

    def greet(self, name: str) -> str:
        """Generate a greeting message using the configured prefix.

        Args:
            name: The name to greet.

        Returns:
            A personalized greeting message with the custom prefix.
        """
        if not name or not name.strip():
            raise ValueError("Name cannot be empty")

        return f"{self.prefix}, {name}!"

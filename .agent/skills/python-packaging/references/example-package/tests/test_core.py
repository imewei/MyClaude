"""Tests for the core module."""

import pytest

from example_package.core import Greeter, greet, greet_multiple


class TestGreet:
    """Tests for the greet function."""

    def test_greet_basic(self):
        """Test basic greeting functionality."""
        assert greet("World") == "Hello, World!"

    def test_greet_different_names(self):
        """Test greeting with different names."""
        assert greet("Alice") == "Hello, Alice!"
        assert greet("Bob") == "Hello, Bob!"

    def test_greet_with_spaces(self):
        """Test greeting with names containing spaces."""
        assert greet("John Doe") == "Hello, John Doe!"

    def test_greet_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Name cannot be empty"):
            greet("")

    def test_greet_whitespace_only_raises(self):
        """Test that whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="Name cannot be empty"):
            greet("   ")


class TestGreetMultiple:
    """Tests for the greet_multiple function."""

    def test_greet_multiple_names(self):
        """Test greeting multiple names."""
        names = ["Alice", "Bob", "Charlie"]
        expected = ["Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"]
        assert greet_multiple(names) == expected

    def test_greet_multiple_empty_list(self):
        """Test greeting an empty list."""
        assert greet_multiple([]) == []

    def test_greet_multiple_single_name(self):
        """Test greeting a single name in a list."""
        assert greet_multiple(["Alice"]) == ["Hello, Alice!"]


class TestGreeter:
    """Tests for the Greeter class."""

    def test_greeter_default_prefix(self):
        """Test greeter with default prefix."""
        greeter = Greeter()
        assert greeter.greet("World") == "Hello, World!"

    def test_greeter_custom_prefix(self):
        """Test greeter with custom prefix."""
        greeter = Greeter(prefix="Hi")
        assert greeter.greet("Alice") == "Hi, Alice!"

    def test_greeter_different_prefixes(self):
        """Test multiple greeters with different prefixes."""
        hello_greeter = Greeter(prefix="Hello")
        hi_greeter = Greeter(prefix="Hi")
        hey_greeter = Greeter(prefix="Hey")

        assert hello_greeter.greet("World") == "Hello, World!"
        assert hi_greeter.greet("World") == "Hi, World!"
        assert hey_greeter.greet("World") == "Hey, World!"

    def test_greeter_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        greeter = Greeter()
        with pytest.raises(ValueError, match="Name cannot be empty"):
            greeter.greet("")

    def test_greeter_prefix_attribute(self):
        """Test that prefix attribute is accessible."""
        greeter = Greeter(prefix="Greetings")
        assert greeter.prefix == "Greetings"


@pytest.fixture
def sample_greeter():
    """Fixture providing a sample greeter instance."""
    return Greeter(prefix="Welcome")


class TestGreeterFixture:
    """Tests demonstrating pytest fixture usage."""

    def test_greeter_fixture(self, sample_greeter):
        """Test using the greeter fixture."""
        assert sample_greeter.greet("User") == "Welcome, User!"

    def test_greeter_fixture_reused(self, sample_greeter):
        """Test that fixture can be reused across tests."""
        result1 = sample_greeter.greet("Alice")
        result2 = sample_greeter.greet("Bob")
        assert result1 == "Welcome, Alice!"
        assert result2 == "Welcome, Bob!"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Alice", "Hello, Alice!"),
        ("Bob", "Hello, Bob!"),
        ("Charlie Brown", "Hello, Charlie Brown!"),
        ("123", "Hello, 123!"),
    ],
)
def test_greet_parametrized(name, expected):
    """Test greet function with parametrized inputs."""
    assert greet(name) == expected

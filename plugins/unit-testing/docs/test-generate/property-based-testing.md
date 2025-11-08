# Property-Based Testing Guide

Comprehensive guide for Hypothesis (Python), QuickCheck equivalents, property generation strategies, shrinking, stateful testing, and scientific computing properties.

## Hypothesis (Python) Patterns

### Basic Property Testing

```python
from hypothesis import given, strategies as st, settings, assume

@given(st.integers())
def test_addition_commutative(x):
    """Addition is commutative"""
    y = 42
    assert x + y == y + x


@given(st.integers(), st.integers())
def test_addition_associative(x, y):
    """Addition is associative"""
    z = 100
    assert (x + y) + z == x + (y + z)


@given(st.lists(st.integers()))
def test_reverse_reverse(lst):
    """Reversing twice gives original list"""
    assert list(reversed(list(reversed(lst)))) == lst
```

### Custom Strategies

```python
# Strategy for valid email addresses
@st.composite
def email_strategy(draw):
    """Generate valid email addresses"""
    local = draw(st.text(
        alphabet=st.characters(blacklist_characters='@'),
        min_size=1,
        max_size=64
    ))
    domain = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll')),
        min_size=1,
        max_size=253
    ))
    tld = draw(st.sampled_from(['com', 'org', 'net', 'edu']))

    return f"{local}@{domain}.{tld}"


@given(email_strategy())
def test_email_validation(email):
    """Test email validation with generated emails"""
    assert validate_email(email) == True


# Strategy for user objects
@st.composite
def user_strategy(draw):
    """Generate user objects"""
    return User(
        name=draw(st.text(min_size=1, max_size=100)),
        email=draw(email_strategy()),
        age=draw(st.integers(min_value=0, max_value=150)),
        active=draw(st.booleans())
    )


@given(user_strategy())
def test_user_serialization(user):
    """Test user can be serialized and deserialized"""
    json_str = user.to_json()
    restored = User.from_json(json_str)
    assert restored == user
```

### NumPy Array Strategies

```python
from hypothesis.extra.numpy import arrays

@given(
    arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=100),
            st.integers(min_value=1, max_value=100)
        ),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False
        )
    )
)
def test_matrix_properties(matrix):
    """Test matrix operations"""
    # Property: Transpose twice gives original
    assert np.allclose(matrix.T.T, matrix)

    # Property: Determinant of transpose equals determinant
    if matrix.shape[0] == matrix.shape[1]:  # Square matrix
        det_original = np.linalg.det(matrix)
        det_transpose = np.linalg.det(matrix.T)
        assert np.allclose(det_original, det_transpose)
```

### Filtering with assume()

```python
@given(st.integers(), st.integers())
def test_division(x, y):
    """Test division properties"""
    assume(y != 0)  # Filter out division by zero

    result = x / y
    assert result * y == pytest.approx(x)


@given(arrays(dtype=np.float64, shape=(10,)))
def test_normalize(arr):
    """Test vector normalization"""
    assume(np.linalg.norm(arr) > 1e-10)  # Filter out zero vectors

    normalized = normalize(arr)
    assert np.allclose(np.linalg.norm(normalized), 1.0)
```

## Property Generation Strategies

### Mathematical Properties

```python
# Idempotence: f(f(x)) = f(x)
@given(st.lists(st.integers()))
def test_idempotent(lst):
    result1 = idempotent_function(lst)
    result2 = idempotent_function(result1)
    assert result1 == result2


# Commutativity: f(x, y) = f(y, x)
@given(st.integers(), st.integers())
def test_commutative(x, y):
    assert commutative_op(x, y) == commutative_op(y, x)


# Associativity: f(f(x, y), z) = f(x, f(y, z))
@given(st.integers(), st.integers(), st.integers())
def test_associative(x, y, z):
    left = associative_op(associative_op(x, y), z)
    right = associative_op(x, associative_op(y, z))
    assert left == right


# Identity: f(x, identity) = x
@given(st.integers())
def test_identity(x):
    identity = 0  # For addition
    assert x + identity == x


# Inverse: f(x, inverse(x)) = identity
@given(st.integers())
def test_inverse(x):
    assume(x != 0)
    inverse = -x  # For addition
    assert x + inverse == 0
```

### Metamorphic Properties

```python
@given(st.lists(st.integers()), st.integers())
def test_sort_insert(lst, x):
    """Sorting after insert gives same result as insert into sorted"""
    # Metamorphic relation
    result1 = sorted(lst + [x])
    result2 = insert_sorted(sorted(lst), x)
    assert result1 == result2


@given(st.lists(st.integers()))
def test_sort_stability(lst):
    """Multiple sorts should give same result"""
    sorted_once = sorted(lst)
    sorted_twice = sorted(sorted_once)
    assert sorted_once == sorted_twice
```

## Shrinking and Minimal Failing Examples

### Understanding Shrinking

```python
@given(st.lists(st.integers()))
def test_sum_positive(lst):
    """This will shrink to minimal failing case: [-1]"""
    assume(len(lst) > 0)
    assert sum(lst) >= 0  # Fails with negative numbers


# Hypothesis will automatically find the minimal failing case:
# [-1] (smallest list with one negative element)
```

### Custom Shrinking

```python
from hypothesis import strategies as st

# Custom strategy with custom shrinking
def custom_interval_strategy():
    """Generate intervals [a, b] where a < b"""
    return st.tuples(st.integers(), st.integers()).filter(
        lambda pair: pair[0] < pair[1]
    )


@given(custom_interval_strategy())
def test_interval_width(interval):
    """Test interval properties"""
    a, b = interval
    assert b - a > 0  # Width is positive
```

## Stateful Testing

### Stateful Test Machines

```python
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition

class BankAccountMachine(RuleBasedStateMachine):
    """Stateful testing for bank account"""

    def __init__(self):
        super().__init__()
        self.account = BankAccount()
        self.expected_balance = 0

    @rule(amount=st.integers(min_value=1, max_value=1000))
    def deposit(self, amount):
        """Deposit money"""
        self.account.deposit(amount)
        self.expected_balance += amount

    @rule(amount=st.integers(min_value=1, max_value=100))
    @precondition(lambda self: self.expected_balance >= 1)
    def withdraw(self, amount):
        """Withdraw money"""
        assume(amount <= self.expected_balance)
        self.account.withdraw(amount)
        self.expected_balance -= amount

    @invariant()
    def balance_matches(self):
        """Balance should always match expected"""
        assert self.account.balance == self.expected_balance

    @invariant()
    def balance_non_negative(self):
        """Balance should never be negative"""
        assert self.account.balance >= 0


# Run stateful tests
TestBankAccount = BankAccountMachine.TestCase
```

### Stateful API Testing

```python
class APIClientMachine(RuleBasedStateMachine):
    """Test API client with stateful operations"""

    def __init__(self):
        super().__init__()
        self.client = APIClient()
        self.created_users = []

    @rule(name=st.text(min_size=1))
    def create_user(self, name):
        """Create user via API"""
        user = self.client.create_user(name)
        self.created_users.append(user)

    @rule(data=st.integers())
    @precondition(lambda self: len(self.created_users) > 0)
    def get_user(self, data):
        """Get existing user"""
        user_id = self.created_users[data % len(self.created_users)].id
        user = self.client.get_user(user_id)
        assert user is not None

    @rule()
    @precondition(lambda self: len(self.created_users) > 0)
    def delete_user(self, data):
        """Delete user"""
        if self.created_users:
            user = self.created_users.pop()
            self.client.delete_user(user.id)

    @invariant()
    def users_retrievable(self):
        """All created users should be retrievable"""
        for user in self.created_users:
            fetched = self.client.get_user(user.id)
            assert fetched is not None


TestAPI = APIClientMachine.TestCase
```

## QuickCheck Equivalents

### JavaScript (fast-check)

```javascript
const fc = require('fast-check');

// Basic property test
test('addition is commutative', () => {
  fc.assert(
    fc.property(fc.integer(), fc.integer(), (a, b) => {
      return a + b === b + a;
    })
  );
});

// Custom arbitrary
const emailArbitrary = fc.tuple(
  fc.stringOf(fc.char(), { minLength: 1, maxLength: 64 }),
  fc.stringOf(fc.char(), { minLength: 1, maxLength: 253 }),
  fc.constantFrom('com', 'org', 'net', 'edu')
).map(([local, domain, tld]) => `${local}@${domain}.${tld}`);

test('email validation', () => {
  fc.assert(
    fc.property(emailArbitrary, (email) => {
      return validateEmail(email) === true;
    })
  );
});

// Stateful testing
const commands = [
  fc.constant(new PushCommand()),
  fc.constant(new PopCommand()),
  fc.constant(new ClearCommand())
];

fc.assert(
  fc.property(fc.commands(commands, { maxCommands: 100 }), (cmds) => {
    const setup = () => ({ model: [], real: new Stack() });
    fc.modelRun(setup, cmds);
  })
);
```

### Go (gopter)

```go
import (
    "testing"
    "github.com/leanovate/gopter"
    "github.com/leanovate/gopter/prop"
)

func TestAdditionCommutative(t *testing.T) {
    properties := gopter.NewProperties(nil)

    properties.Property("addition is commutative", prop.ForAll(
        func(a, b int) bool {
            return a + b == b + a
        },
        gen.Int(),
        gen.Int(),
    ))

    properties.TestingRun(t)
}

func TestSortIdempotent(t *testing.T) {
    properties := gopter.NewProperties(nil)

    properties.Property("sorting twice gives same result", prop.ForAll(
        func(slice []int) bool {
            sorted1 := sort(slice)
            sorted2 := sort(sorted1)
            return reflect.DeepEqual(sorted1, sorted2)
        },
        gen.SliceOf(gen.Int()),
    ))

    properties.TestingRun(t)
}
```

## Scientific Computing Properties

### Linear Algebra Properties

```python
@given(
    arrays(dtype=np.float64, shape=(10, 10)),
    arrays(dtype=np.float64, shape=(10, 10))
)
def test_matrix_multiplication(A, B):
    """Test matrix multiplication properties"""

    # Property 1: (AB)^T = B^T A^T
    left = (A @ B).T
    right = B.T @ A.T
    assert_allclose(left, right, rtol=1e-10)

    # Property 2: If C = AB, then rank(C) <= min(rank(A), rank(B))
    C = A @ B
    rank_C = np.linalg.matrix_rank(C)
    rank_A = np.linalg.matrix_rank(A)
    rank_B = np.linalg.matrix_rank(B)
    assert rank_C <= min(rank_A, rank_B)


@given(arrays(dtype=np.float64, shape=(10,)))
def test_vector_norm_properties(v):
    """Test vector norm properties"""

    # Property: ||v|| >= 0
    norm = np.linalg.norm(v)
    assert norm >= 0

    # Property: ||v|| = 0 iff v = 0
    if norm < 1e-10:
        assert_allclose(v, np.zeros_like(v), atol=1e-10)

    # Property: ||av|| = |a| ||v||
    a = 2.5
    assert_allclose(np.linalg.norm(a * v), abs(a) * norm, rtol=1e-10)
```

### Optimization Properties

```python
@given(
    arrays(dtype=np.float64, shape=(10,)),
    st.floats(min_value=0.01, max_value=0.1)
)
def test_gradient_descent_decreases(x0, learning_rate):
    """Test gradient descent decreases objective"""

    def objective(x):
        return np.sum(x**2)  # Quadratic function

    x = x0.copy()

    for _ in range(10):
        grad = 2 * x  # Gradient of x^2
        x_new = x - learning_rate * grad

        # Property: Objective should decrease
        assert objective(x_new) <= objective(x)

        x = x_new
```

## Best Practices

1. **Start with simple properties** (e.g., idempotence, commutativity)
2. **Use assume() sparingly** (may slow down test generation)
3. **Let Hypothesis shrink** to minimal failing examples
4. **Test properties, not implementation**
5. **Use custom strategies** for domain-specific types
6. **Combine with example-based tests**
7. **Use stateful testing** for complex state machines
8. **Set appropriate example counts** (@settings)
9. **Use deterministic seeds** for reproducibility
10. **Document mathematical properties** being tested

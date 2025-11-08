# Refactoring Patterns

**Version**: 1.0.3
**Category**: codebase-cleanup
**Purpose**: Catalog of design patterns, code smells, and refactoring techniques for systematic code improvement

## Common Design Patterns

### Factory Pattern

**Use Case**: Object creation with flexible instantiation logic

**Before** (Direct instantiation):
```python
class DataProcessor:
    def process(self, file_path):
        if file_path.endswith('.csv'):
            processor = CSVProcessor()
        elif file_path.endswith('.json'):
            processor = JSONProcessor()
        elif file_path.endswith('.xml'):
            processor = XMLProcessor()
        else:
            raise ValueError("Unsupported format")

        return processor.process(file_path)
```

**After** (Factory Pattern):
```python
from abc import ABC, abstractmethod

class FileProcessor(ABC):
    @abstractmethod
    def process(self, file_path): pass

class CSVProcessor(FileProcessor):
    def process(self, file_path):
        # CSV processing logic
        pass

class JSONProcessor(FileProcessor):
    def process(self, file_path):
        # JSON processing logic
        pass

class FileProcessorFactory:
    _processors = {
        '.csv': CSVProcessor,
        '.json': JSONProcessor,
        '.xml': XMLProcessor
    }

    @classmethod
    def create_processor(cls, file_path: str) -> FileProcessor:
        extension = Path(file_path).suffix
        processor_class = cls._processors.get(extension)

        if not processor_class:
            raise ValueError(f"Unsupported format: {extension}")

        return processor_class()

# Usage
processor = FileProcessorFactory.create_processor("data.csv")
result = processor.process("data.csv")
```

### Strategy Pattern

**Use Case**: Encapsulate algorithms and make them interchangeable

**Before** (Conditional logic):
```typescript
class PricingEngine {
    calculatePrice(product: Product, customerType: string): number {
        if (customerType === 'regular') {
            return product.price;
        } else if (customerType === 'premium') {
            return product.price * 0.9;
        } else if (customerType === 'vip') {
            return product.price * 0.8;
        } else if (customerType === 'wholesale') {
            return product.price * 0.7;
        }
        return product.price;
    }
}
```

**After** (Strategy Pattern):
```typescript
interface PricingStrategy {
    calculatePrice(product: Product): number;
}

class RegularPricing implements PricingStrategy {
    calculatePrice(product: Product): number {
        return product.price;
    }
}

class PremiumPricing implements PricingStrategy {
    calculatePrice(product: Product): number {
        return product.price * 0.9;
    }
}

class VIPPricing implements PricingStrategy {
    calculatePrice(product: Product): number {
        return product.price * 0.8;
    }
}

class PricingEngine {
    constructor(private strategy: PricingStrategy) {}

    calculatePrice(product: Product): number {
        return this.strategy.calculatePrice(product);
    }

    setStrategy(strategy: PricingStrategy): void {
        this.strategy = strategy;
    }
}

// Usage
const engine = new PricingEngine(new PremiumPricing());
const price = engine.calculatePrice(product);
```

### Repository Pattern

**Use Case**: Abstract data access layer for testability and flexibility

**Before** (Direct database access):
```python
class UserService:
    def get_user(self, user_id):
        conn = psycopg2.connect(database="mydb")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return User.from_row(row)

    def save_user(self, user):
        conn = psycopg2.connect(database="mydb")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)",
            (user.name, user.email)
        )
        conn.commit()
        conn.close()
```

**After** (Repository Pattern):
```python
from abc import ABC, abstractmethod

class UserRepository(ABC):
    @abstractmethod
    def get_by_id(self, user_id: int) -> User:
        pass

    @abstractmethod
    def save(self, user: User) -> None:
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> Optional[User]:
        pass

class PostgresUserRepository(UserRepository):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def get_by_id(self, user_id: int) -> User:
        with psycopg2.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cursor.fetchone()
            return User.from_row(row)

    def save(self, user: User) -> None:
        with psycopg2.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email) VALUES (%s, %s)",
                (user.name, user.email)
            )
            conn.commit()

    def find_by_email(self, email: str) -> Optional[User]:
        with psycopg2.connect(self.connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            row = cursor.fetchone()
            return User.from_row(row) if row else None

class UserService:
    def __init__(self, user_repository: UserRepository):
        self.repository = user_repository

    def get_user(self, user_id: int) -> User:
        return self.repository.get_by_id(user_id)

    def save_user(self, user: User) -> None:
        self.repository.save(user)

# Usage - Easy to test with mock repository
service = UserService(PostgresUserRepository("postgresql://localhost/mydb"))
```

### Observer Pattern

**Use Case**: Event-driven architecture with loose coupling

```python
from typing import List, Callable

class Event:
    def __init__(self):
        self._subscribers: List[Callable] = []

    def subscribe(self, callback: Callable) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        self._subscribers.remove(callback)

    def emit(self, *args, **kwargs) -> None:
        for callback in self._subscribers:
            callback(*args, **kwargs)

class OrderProcessor:
    def __init__(self):
        self.on_order_created = Event()
        self.on_order_completed = Event()
        self.on_order_failed = Event()

    def create_order(self, order_data):
        order = Order(**order_data)
        # Process order...

        self.on_order_created.emit(order)
        return order

    def complete_order(self, order_id):
        order = self.get_order(order_id)
        order.status = 'completed'

        self.on_order_completed.emit(order)

# Subscribers
def send_confirmation_email(order):
    EmailService.send(order.customer.email, "Order confirmed")

def update_inventory(order):
    InventoryService.decrement_stock(order.items)

def log_order(order):
    Logger.info(f"Order created: {order.id}")

# Setup
processor = OrderProcessor()
processor.on_order_created.subscribe(send_confirmation_email)
processor.on_order_created.subscribe(update_inventory)
processor.on_order_created.subscribe(log_order)
```

### Decorator Pattern

**Use Case**: Add responsibilities to objects dynamically

```typescript
interface Coffee {
    cost(): number;
    description(): string;
}

class SimpleCoffee implements Coffee {
    cost(): number {
        return 2.0;
    }

    description(): string {
        return "Simple coffee";
    }
}

abstract class CoffeeDecorator implements Coffee {
    constructor(protected coffee: Coffee) {}

    abstract cost(): number;
    abstract description(): string;
}

class MilkDecorator extends CoffeeDecorator {
    cost(): number {
        return this.coffee.cost() + 0.5;
    }

    description(): string {
        return this.coffee.description() + ", milk";
    }
}

class SugarDecorator extends CoffeeDecorator {
    cost(): number {
        return this.coffee.cost() + 0.2;
    }

    description(): string {
        return this.coffee.description() + ", sugar";
    }
}

class WhippedCreamDecorator extends CoffeeDecorator {
    cost(): number {
        return this.coffee.cost() + 0.7;
    }

    description(): string {
        return this.coffee.description() + ", whipped cream";
    }
}

// Usage
let coffee: Coffee = new SimpleCoffee();
coffee = new MilkDecorator(coffee);
coffee = new SugarDecorator(coffee);
coffee = new WhippedCreamDecorator(coffee);

console.log(coffee.description()); // "Simple coffee, milk, sugar, whipped cream"
console.log(coffee.cost()); // 3.4
```

## Code Smell Catalog

### Long Method

**Smell**: Method with 30+ lines or multiple responsibility levels

**Detection**:
```python
def process_order(order_data):
    # Validation (10 lines)
    if not order_data.get('customer_id'):
        raise ValueError("Customer ID required")
    # ... more validation

    # Database operations (15 lines)
    conn = get_connection()
    cursor = conn.cursor()
    # ... database code

    # Email notification (10 lines)
    smtp = get_smtp_connection()
    # ... email code

    # Logging (5 lines)
    # ... logging code

    # Analytics (8 lines)
    # ... analytics code
```

**Refactoring**: Extract Method

```python
def process_order(order_data):
    validate_order(order_data)
    order = save_order_to_database(order_data)
    send_confirmation_email(order)
    log_order_creation(order)
    track_order_analytics(order)
    return order

def validate_order(order_data):
    if not order_data.get('customer_id'):
        raise ValueError("Customer ID required")
    # ... focused validation logic

def save_order_to_database(order_data):
    # ... focused database logic
    return order

def send_confirmation_email(order):
    # ... focused email logic
    pass

def log_order_creation(order):
    # ... focused logging logic
    pass

def track_order_analytics(order):
    # ... focused analytics logic
    pass
```

### Large Class

**Smell**: Class with 300+ lines or 10+ methods

**Detection**:
```python
class UserManager:
    # User CRUD operations (50 lines)
    def create_user(self): pass
    def update_user(self): pass
    def delete_user(self): pass

    # Authentication (60 lines)
    def login(self): pass
    def logout(self): pass
    def reset_password(self): pass

    # Authorization (40 lines)
    def check_permission(self): pass
    def assign_role(self): pass

    # Email operations (50 lines)
    def send_welcome_email(self): pass
    def send_password_reset(self): pass

    # Profile management (40 lines)
    def update_profile(self): pass
    def upload_avatar(self): pass

    # Reporting (30 lines)
    def generate_user_report(self): pass
```

**Refactoring**: Extract Class

```python
class UserRepository:
    def create(self, user): pass
    def update(self, user): pass
    def delete(self, user_id): pass
    def find_by_id(self, user_id): pass

class AuthenticationService:
    def login(self, credentials): pass
    def logout(self, user_id): pass
    def reset_password(self, email): pass

class AuthorizationService:
    def check_permission(self, user, resource): pass
    def assign_role(self, user, role): pass

class UserEmailService:
    def send_welcome_email(self, user): pass
    def send_password_reset(self, user): pass

class UserProfileService:
    def update_profile(self, user_id, profile_data): pass
    def upload_avatar(self, user_id, file): pass

class UserReportingService:
    def generate_user_report(self, filters): pass
```

### Duplicate Code

**Smell**: Identical or very similar code blocks in multiple places

**Detection**:
```python
# In OrderController
def create_order(self, request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    if not request.user.has_permission('create_order'):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    # ... order creation logic

# In ProductController
def create_product(self, request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    if not request.user.has_permission('create_product'):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    # ... product creation logic
```

**Refactoring**: Extract Function/Decorator

```python
def require_authentication(func):
    def wrapper(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'Unauthorized'}, status=401)
        return func(self, request, *args, **kwargs)
    return wrapper

def require_permission(permission):
    def decorator(func):
        def wrapper(self, request, *args, **kwargs):
            if not request.user.has_permission(permission):
                return JsonResponse({'error': 'Forbidden'}, status=403)
            return func(self, request, *args, **kwargs)
        return wrapper
    return decorator

# Usage
@require_authentication
@require_permission('create_order')
def create_order(self, request):
    # ... order creation logic

@require_authentication
@require_permission('create_product')
def create_product(self, request):
    # ... product creation logic
```

### Feature Envy

**Smell**: Method uses more features of another class than its own

**Detection**:
```python
class Order:
    def __init__(self, customer, items):
        self.customer = customer
        self.items = items

    def calculate_discount(self):
        # Uses customer features extensively
        if self.customer.is_premium():
            base_discount = 0.1
        else:
            base_discount = 0.05

        if self.customer.loyalty_points > 1000:
            base_discount += 0.05

        if self.customer.orders_count > 10:
            base_discount += 0.03

        return base_discount
```

**Refactoring**: Move Method

```python
class Customer:
    def __init__(self, is_premium, loyalty_points, orders_count):
        self.is_premium = is_premium
        self.loyalty_points = loyalty_points
        self.orders_count = orders_count

    def calculate_discount(self):
        base_discount = 0.1 if self.is_premium else 0.05

        if self.loyalty_points > 1000:
            base_discount += 0.05

        if self.orders_count > 10:
            base_discount += 0.03

        return base_discount

class Order:
    def __init__(self, customer, items):
        self.customer = customer
        self.items = items

    def calculate_discount(self):
        return self.customer.calculate_discount()
```

### Primitive Obsession

**Smell**: Using primitives instead of small objects for simple tasks

**Detection**:
```python
def send_email(to_address: str, subject: str, body: str):
    # Email validation in multiple places
    if '@' not in to_address or '.' not in to_address.split('@')[1]:
        raise ValueError("Invalid email")
    # ... send email

def validate_user_email(email: str):
    if '@' not in email or '.' not in email.split('@')[1]:
        raise ValueError("Invalid email")
```

**Refactoring**: Introduce Value Object

```python
class EmailAddress:
    def __init__(self, address: str):
        if not self._is_valid(address):
            raise ValueError(f"Invalid email address: {address}")
        self.address = address

    @staticmethod
    def _is_valid(address: str) -> bool:
        if '@' not in address:
            return False
        local, domain = address.split('@', 1)
        return '.' in domain and len(local) > 0 and len(domain) > 0

    def __str__(self):
        return self.address

    def __eq__(self, other):
        return isinstance(other, EmailAddress) and self.address == other.address

def send_email(to: EmailAddress, subject: str, body: str):
    # No validation needed - EmailAddress guarantees validity
    # ... send email

def validate_user_email(email: EmailAddress):
    # Email is already validated by construction
    pass
```

## Refactoring Techniques

### Extract Method

**When**: Method is too long or does multiple things

**Steps**:
1. Identify cohesive code block
2. Create new method with descriptive name
3. Move code to new method
4. Replace original code with method call

**Example**:
```python
# Before
def generate_report(data):
    # Calculate totals
    total = 0
    for item in data:
        total += item.amount

    # Format output
    lines = []
    lines.append(f"Total: ${total}")
    lines.append("-" * 40)
    for item in data:
        lines.append(f"{item.name}: ${item.amount}")

    return "\n".join(lines)

# After
def generate_report(data):
    total = calculate_total(data)
    return format_report(data, total)

def calculate_total(data):
    return sum(item.amount for item in data)

def format_report(data, total):
    lines = [
        f"Total: ${total}",
        "-" * 40
    ]
    lines.extend(f"{item.name}: ${item.amount}" for item in data)
    return "\n".join(lines)
```

### Extract Class

**When**: Class has too many responsibilities

**Steps**:
1. Identify cohesive group of fields/methods
2. Create new class
3. Move fields and methods to new class
4. Establish relationship between classes

**Example**: See "Large Class" smell example above

### Introduce Parameter Object

**When**: Multiple parameters are always passed together

```python
# Before
def create_user(first_name, last_name, email, phone, street, city, state, zip_code):
    pass

# After
@dataclass
class Address:
    street: str
    city: str
    state: str
    zip_code: str

@dataclass
class ContactInfo:
    email: str
    phone: str
    address: Address

@dataclass
class UserProfile:
    first_name: str
    last_name: str
    contact: ContactInfo

def create_user(profile: UserProfile):
    pass
```

### Replace Conditional with Polymorphism

**When**: Conditional logic based on object type

```python
# Before
def calculate_area(shape):
    if shape.type == 'circle':
        return 3.14 * shape.radius ** 2
    elif shape.type == 'rectangle':
        return shape.width * shape.height
    elif shape.type == 'triangle':
        return 0.5 * shape.base * shape.height

# After
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def calculate_area(self): pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def calculate_area(self):
        return 3.14 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def calculate_area(self):
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def calculate_area(self):
        return 0.5 * self.base * self.height
```

## Refactoring Safety Checklist

- [ ] All tests pass before refactoring
- [ ] One refactoring at a time (no mixing with feature changes)
- [ ] Commit after each successful refactoring
- [ ] Run full test suite after each refactoring
- [ ] Code review for architectural changes
- [ ] Performance testing for hot paths
- [ ] Database migration plan for data layer changes
- [ ] Rollback plan documented

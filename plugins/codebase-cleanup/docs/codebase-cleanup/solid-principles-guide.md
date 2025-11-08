# SOLID Principles Guide

**Version**: 1.0.3
**Category**: codebase-cleanup
**Purpose**: Comprehensive guide to SOLID principles with practical examples and refactoring patterns

## Single Responsibility Principle (SRP)

### Definition
A class should have one, and only one, reason to change.

### Before (Violates SRP)
```python
class UserManager:
    def create_user(self, data):
        # Validation
        if not data.get('email'):
            raise ValueError("Email required")

        # Database
        cursor.execute("INSERT INTO users...")

        # Email
        smtp.send_mail(data['email'], "Welcome!")

        # Logging
        log.write(f"User created: {data['email']}")

        # Cache
        cache.set(f"user:{data['email']}", user_data)
```

### After (Follows SRP)
```python
class UserValidator:
    def validate(self, data):
        if not data.get('email'):
            raise ValueError("Email required")

class UserRepository:
    def save(self, user):
        cursor.execute("INSERT INTO users...")

class EmailService:
    def send_welcome(self, email):
        smtp.send_mail(email, "Welcome!")

class UserActivityLogger:
    def log_creation(self, email):
        log.write(f"User created: {email}")

class UserService:
    def __init__(self, validator, repository, email_service, logger):
        self.validator = validator
        self.repository = repository
        self.email_service = email_service
        self.logger = logger

    def create_user(self, data):
        self.validator.validate(data)
        user = self.repository.save(data)
        self.email_service.send_welcome(user.email)
        self.logger.log_creation(user.email)
        return user
```

## Open/Closed Principle (OCP)

### Definition
Software entities should be open for extension but closed for modification.

### Before (Violates OCP)
```python
class DiscountCalculator:
    def calculate(self, order, discount_type):
        if discount_type == "percentage":
            return order.total * 0.1
        elif discount_type == "fixed":
            return 10
        elif discount_type == "tiered":
            if order.total > 1000: return order.total * 0.15
            if order.total > 500: return order.total * 0.10
            return order.total * 0.05
        # Adding new type requires modifying this class!
```

### After (Follows OCP)
```python
from abc import ABC, abstractmethod

class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, order): pass

class PercentageDiscount(DiscountStrategy):
    def __init__(self, percentage):
        self.percentage = percentage

    def calculate(self, order):
        return order.total * self.percentage

class FixedDiscount(DiscountStrategy):
    def __init__(self, amount):
        self.amount = amount

    def calculate(self, order):
        return self.amount

class TieredDiscount(DiscountStrategy):
    def calculate(self, order):
        if order.total > 1000: return order.total * 0.15
        if order.total > 500: return order.total * 0.10
        return order.total * 0.05

# Adding new discount: just create new class, no modification needed
class BuyOneGetOneDiscount(DiscountStrategy):
    def calculate(self, order):
        return min(item.price for item in order.items)
```

## Liskov Substitution Principle (LSP)

### Definition
Derived classes must be substitutable for their base classes.

### Before (Violates LSP)
```typescript
class Rectangle {
    protected width: number;
    protected height: number;

    setWidth(width: number) { this.width = width; }
    setHeight(height: number) { this.height = height; }
    area(): number { return this.width * this.height; }
}

class Square extends Rectangle {
    setWidth(width: number) {
        this.width = width;
        this.height = width; // Breaks LSP!
    }

    setHeight(height: number) {
        this.width = height;
        this.height = height; // Breaks LSP!
    }
}

// This breaks:
function processRectangle(rect: Rectangle) {
    rect.setWidth(5);
    rect.setHeight(4);
    console.assert(rect.area() === 20); // Fails for Square!
}
```

### After (Follows LSP)
```typescript
interface Shape {
    area(): number;
}

class Rectangle implements Shape {
    constructor(private width: number, private height: number) {}
    area(): number { return this.width * this.height; }
}

class Square implements Shape {
    constructor(private side: number) {}
    area(): number { return this.side * this.side; }
}

// Both implement Shape correctly, no LSP violation
function calculateArea(shape: Shape): number {
    return shape.area(); // Works for any Shape
}
```

## Interface Segregation Principle (ISP)

### Definition
Clients should not be forced to depend on interfaces they don't use.

### Before (Violates ISP)
```java
interface Worker {
    void work();
    void eat();
    void sleep();
}

class Robot implements Worker {
    public void work() { /* work */ }
    public void eat() { throw new UnsupportedOperationException(); } // Wrong!
    public void sleep() { throw new UnsupportedOperationException(); } // Wrong!
}
```

### After (Follows ISP)
```java
interface Workable {
    void work();
}

interface Eatable {
    void eat();
}

interface Sleepable {
    void sleep();
}

class Human implements Workable, Eatable, Sleepable {
    public void work() { /* work */ }
    public void eat() { /* eat */ }
    public void sleep() { /* sleep */ }
}

class Robot implements Workable {
    public void work() { /* work */ }
    // No need to implement eat() or sleep()
}
```

## Dependency Inversion Principle (DIP)

### Definition
High-level modules should not depend on low-level modules. Both should depend on abstractions.

### Before (Violates DIP)
```go
type MySQLDatabase struct{}

func (db *MySQLDatabase) Save(data string) {
    // MySQL-specific code
}

type UserService struct {
    db *MySQLDatabase // Tight coupling to MySQL!
}

func (s *UserService) CreateUser(name string) {
    s.db.Save(name)
}
```

### After (Follows DIP)
```go
// Abstraction
type Database interface {
    Save(data string)
}

// Low-level module
type MySQLDatabase struct{}
func (db *MySQLDatabase) Save(data string) { /* MySQL */ }

type PostgresDatabase struct{}
func (db *PostgresDatabase) Save(data string) { /* Postgres */ }

// High-level module depends on abstraction
type UserService struct {
    db Database // Depends on interface
}

func NewUserService(db Database) *UserService {
    return &UserService{db: db}
}

func (s *UserService) CreateUser(name string) {
    s.db.Save(name) // Works with any Database implementation
}

// Usage
service := NewUserService(&MySQLDatabase{})
// or
service := NewUserService(&PostgresDatabase{})
```

## Practical Application Checklist

- [ ] Each class has a single, well-defined responsibility
- [ ] New features can be added without modifying existing code
- [ ] Subclasses can be used wherever base class is expected
- [ ] Interfaces are client-specific, not fat interfaces
- [ ] Dependencies point towards abstractions, not concretions

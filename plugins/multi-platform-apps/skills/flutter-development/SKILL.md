# Flutter Development Skill

> **Comprehensive Flutter development patterns, best practices, and common workflows for building production-ready applications.**

---

## Skill Overview

This skill provides systematic knowledge transfer for Flutter development, covering essential patterns, state management strategies, performance optimization, and production best practices. Use this skill to accelerate Flutter learning and establish consistent development patterns.

**Target Audience**: Developers learning Flutter or teams establishing Flutter development standards

**Estimated Learning Time**: 4-6 hours to master core concepts

---

## Core Concepts

### 1. Widget Composition Fundamentals

**Key Principle**: Everything in Flutter is a widget. Master composition over inheritance.

#### Basic Widget Structure

```dart
// ✅ Good: Composition-based widget
class UserCard extends StatelessWidget {
  const UserCard({
    super.key,
    required this.name,
    required this.email,
  });

  final String name;
  final String email;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: const CircleAvatar(
          child: Icon(Icons.person),
        ),
        title: Text(name),
        subtitle: Text(email),
      ),
    );
  }
}
```

#### Widget Best Practices

1. **Use `const` constructors wherever possible**
   - Reduces rebuilds and improves performance
   - Enables compile-time widget caching

```dart
// ✅ Good: Const constructor
const Text('Hello', style: TextStyle(fontSize: 16));

// ❌ Bad: Non-const when possible
Text('Hello', style: TextStyle(fontSize: 16));
```

2. **Extract widgets for reusability**
   - Keep build methods under 20 lines
   - Create custom widgets for repeated UI patterns

3. **Use keys strategically**
   - For list items that can be reordered
   - To preserve state across rebuilds

```dart
// ✅ Good: Using keys in a list
ListView.builder(
  itemCount: items.length,
  itemBuilder: (context, index) {
    return UserCard(
      key: ValueKey(items[index].id),
      name: items[index].name,
      email: items[index].email,
    );
  },
);
```

---

### 2. State Management Strategies

**Choose based on app complexity and team preferences**

#### StatefulWidget (Simple Local State)

```dart
class Counter extends StatefulWidget {
  const Counter({super.key});

  @override
  State<Counter> createState() => _CounterState();
}

class _CounterState extends State<Counter> {
  int _count = 0;

  void _increment() {
    setState(() {
      _count++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text('Count: $_count'),
        ElevatedButton(
          onPressed: _increment,
          child: const Text('Increment'),
        ),
      ],
    );
  }
}
```

#### Riverpod 2.x (Recommended for Complex Apps)

```dart
// 1. Define providers
final counterProvider = StateProvider<int>((ref) => 0);

// 2. Use in widgets
class Counter extends ConsumerWidget {
  const Counter({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final count = ref.watch(counterProvider);

    return Column(
      children: [
        Text('Count: $count'),
        ElevatedButton(
          onPressed: () => ref.read(counterProvider.notifier).state++,
          child: const Text('Increment'),
        ),
      ],
    );
  }
}

// 3. Wrap app with ProviderScope
void main() {
  runApp(
    const ProviderScope(
      child: MyApp(),
    ),
  );
}
```

#### State Management Decision Matrix

| Complexity | Recommended Solution | Use Case |
|------------|---------------------|----------|
| Simple (1-2 screens) | StatefulWidget + InheritedWidget | Form state, toggles |
| Medium (3-10 screens) | Provider or Riverpod | Todo apps, small business apps |
| Complex (10+ screens) | Riverpod + Repository pattern | E-commerce, social media |
| Enterprise | Bloc/Cubit + Clean Architecture | Banking, healthcare apps |

---

### 3. Navigation Patterns

#### Basic Navigation (Navigator 1.0)

```dart
// Push to new screen
Navigator.push(
  context,
  MaterialPageRoute(
    builder: (context) => DetailScreen(id: item.id),
  ),
);

// Pop back
Navigator.pop(context);

// Named routes
Navigator.pushNamed(context, '/details', arguments: item.id);
```

#### Declarative Navigation (Navigator 2.0 / Go Router)

```dart
// Using go_router package
final router = GoRouter(
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => const HomeScreen(),
      routes: [
        GoRoute(
          path: 'details/:id',
          builder: (context, state) {
            final id = state.pathParameters['id']!;
            return DetailScreen(id: id);
          },
        ),
      ],
    ),
  ],
);

// Usage
context.go('/details/123');
context.push('/settings');
context.pop();
```

---

### 4. Async Operations & Data Fetching

#### Future-based Data Loading

```dart
class UserProfile extends StatefulWidget {
  const UserProfile({super.key, required this.userId});

  final String userId;

  @override
  State<UserProfile> createState() => _UserProfileState();
}

class _UserProfileState extends State<UserProfile> {
  late Future<User> _userFuture;

  @override
  void initState() {
    super.initState();
    _userFuture = _fetchUser();
  }

  Future<User> _fetchUser() async {
    final response = await http.get(
      Uri.parse('https://api.example.com/users/${widget.userId}'),
    );

    if (response.statusCode == 200) {
      return User.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to load user');
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<User>(
      future: _userFuture,
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          return UserCard(user: snapshot.data!);
        } else if (snapshot.hasError) {
          return ErrorWidget(error: snapshot.error.toString());
        }
        return const CircularProgressIndicator();
      },
    );
  }
}
```

#### Stream-based Real-time Updates

```dart
class MessageList extends StatelessWidget {
  const MessageList({super.key});

  Stream<List<Message>> _messageStream() {
    return FirebaseFirestore.instance
        .collection('messages')
        .orderBy('timestamp', descending: true)
        .limit(50)
        .snapshots()
        .map((snapshot) => snapshot.docs
            .map((doc) => Message.fromFirestore(doc))
            .toList());
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<List<Message>>(
      stream: _messageStream(),
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          final messages = snapshot.data!;
          return ListView.builder(
            itemCount: messages.length,
            itemBuilder: (context, index) {
              return MessageCard(message: messages[index]);
            },
          );
        } else if (snapshot.hasError) {
          return ErrorWidget(error: snapshot.error.toString());
        }
        return const CircularProgressIndicator();
      },
    );
  }
}
```

---

### 5. Performance Optimization

#### Minimize Widget Rebuilds

```dart
// ✅ Good: Using const widgets
class MyWidget extends StatelessWidget {
  const MyWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Text('Static text'), // Won't rebuild
        const Icon(Icons.star),     // Won't rebuild
        DynamicContent(),            // Only this rebuilds
      ],
    );
  }
}

// ✅ Good: Using memo for expensive computations
class ExpensiveWidget extends StatelessWidget {
  const ExpensiveWidget({super.key, required this.items});

  final List<Item> items;

  @override
  Widget build(BuildContext context) {
    final sortedItems = useMemo(
      () => items.toList()..sort(),
      [items],
    );

    return ListView.builder(
      itemCount: sortedItems.length,
      itemBuilder: (context, index) => ItemCard(item: sortedItems[index]),
    );
  }
}
```

#### ListView Optimization

```dart
// ✅ Good: Efficient list rendering
ListView.builder(
  itemCount: items.length,
  itemBuilder: (context, index) {
    return ItemCard(
      key: ValueKey(items[index].id),
      item: items[index],
    );
  },
  // Add caching for better performance
  cacheExtent: 100,
);

// ✅ Better: Separated list items
class ItemCard extends StatelessWidget {
  const ItemCard({super.key, required this.item});

  final Item item;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        title: Text(item.name),
        subtitle: Text(item.description),
      ),
    );
  }
}
```

#### Image Optimization

```dart
// ✅ Good: Cached network images
CachedNetworkImage(
  imageUrl: imageUrl,
  placeholder: (context, url) => const CircularProgressIndicator(),
  errorWidget: (context, url, error) => const Icon(Icons.error),
  memCacheWidth: 200, // Resize for memory efficiency
  memCacheHeight: 200,
);

// ✅ Good: Lazy loading with precaching
Future<void> precacheImages(BuildContext context) async {
  for (final imageUrl in imageUrls) {
    await precacheImage(
      CachedNetworkImageProvider(imageUrl),
      context,
    );
  }
}
```

---

### 6. Error Handling & Loading States

#### Comprehensive State Management

```dart
// Sealed class for state management
sealed class DataState<T> {
  const DataState();
}

class Loading<T> extends DataState<T> {
  const Loading();
}

class Success<T> extends DataState<T> {
  const Success(this.data);
  final T data;
}

class Error<T> extends DataState<T> {
  const Error(this.message);
  final String message;
}

// Usage in widget
class DataWidget extends ConsumerWidget {
  const DataWidget({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(dataProvider);

    return switch (state) {
      Loading() => const CircularProgressIndicator(),
      Success(:final data) => ContentWidget(data: data),
      Error(:final message) => ErrorWidget(message: message),
    };
  }
}
```

---

### 7. Testing Strategies

#### Widget Testing

```dart
void main() {
  group('Counter Widget Tests', () {
    testWidgets('Initial count is zero', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(home: Counter()),
      );

      expect(find.text('Count: 0'), findsOneWidget);
    });

    testWidgets('Increment increases count', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(home: Counter()),
      );

      await tester.tap(find.text('Increment'));
      await tester.pump();

      expect(find.text('Count: 1'), findsOneWidget);
    });
  });
}
```

#### Integration Testing

```dart
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Complete user flow test', (tester) async {
    await tester.pumpWidget(const MyApp());

    // Navigate to login
    await tester.tap(find.text('Login'));
    await tester.pumpAndSettle();

    // Enter credentials
    await tester.enterText(find.byKey(const Key('email')), 'test@example.com');
    await tester.enterText(find.byKey(const Key('password')), 'password123');

    // Submit login
    await tester.tap(find.text('Submit'));
    await tester.pumpAndSettle();

    // Verify navigation to home
    expect(find.text('Welcome'), findsOneWidget);
  });
}
```

---

## Common Patterns & Recipes

### 1. Repository Pattern for Data Layer

```dart
// Abstract repository interface
abstract class UserRepository {
  Future<User> getUser(String id);
  Future<void> updateUser(User user);
}

// Implementation with API
class ApiUserRepository implements UserRepository {
  ApiUserRepository(this.apiClient);

  final ApiClient apiClient;

  @override
  Future<User> getUser(String id) async {
    final response = await apiClient.get('/users/$id');
    return User.fromJson(response.data);
  }

  @override
  Future<void> updateUser(User user) async {
    await apiClient.put('/users/${user.id}', data: user.toJson());
  }
}

// Provider for dependency injection
final userRepositoryProvider = Provider<UserRepository>((ref) {
  return ApiUserRepository(ref.watch(apiClientProvider));
});
```

### 2. Form Validation Pattern

```dart
class LoginForm extends StatefulWidget {
  const LoginForm({super.key});

  @override
  State<LoginForm> createState() => _LoginFormState();
}

class _LoginFormState extends State<LoginForm> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  String? _validateEmail(String? value) {
    if (value == null || value.isEmpty) {
      return 'Email is required';
    }
    if (!value.contains('@')) {
      return 'Invalid email format';
    }
    return null;
  }

  String? _validatePassword(String? value) {
    if (value == null || value.isEmpty) {
      return 'Password is required';
    }
    if (value.length < 8) {
      return 'Password must be at least 8 characters';
    }
    return null;
  }

  Future<void> _submit() async {
    if (_formKey.currentState!.validate()) {
      // Process login
      await login(_emailController.text, _passwordController.text);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          TextFormField(
            controller: _emailController,
            decoration: const InputDecoration(labelText: 'Email'),
            validator: _validateEmail,
            keyboardType: TextInputType.emailAddress,
          ),
          TextFormField(
            controller: _passwordController,
            decoration: const InputDecoration(labelText: 'Password'),
            validator: _validatePassword,
            obscureText: true,
          ),
          ElevatedButton(
            onPressed: _submit,
            child: const Text('Login'),
          ),
        ],
      ),
    );
  }
}
```

### 3. Responsive Design Pattern

```dart
class ResponsiveLayout extends StatelessWidget {
  const ResponsiveLayout({
    super.key,
    required this.mobile,
    this.tablet,
    this.desktop,
  });

  final Widget mobile;
  final Widget? tablet;
  final Widget? desktop;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth >= 1200) {
          return desktop ?? tablet ?? mobile;
        } else if (constraints.maxWidth >= 600) {
          return tablet ?? mobile;
        } else {
          return mobile;
        }
      },
    );
  }
}

// Usage
ResponsiveLayout(
  mobile: MobileLayout(),
  tablet: TabletLayout(),
  desktop: DesktopLayout(),
);
```

---

## Architecture Best Practices

### Clean Architecture Structure

```
lib/
├── core/
│   ├── constants/
│   ├── theme/
│   └── utils/
├── features/
│   └── user/
│       ├── data/
│       │   ├── models/
│       │   ├── repositories/
│       │   └── data_sources/
│       ├── domain/
│       │   ├── entities/
│       │   ├── repositories/
│       │   └── use_cases/
│       └── presentation/
│           ├── pages/
│           ├── widgets/
│           └── providers/
└── main.dart
```

---

## Quick Reference

### Essential Packages

| Package | Purpose | Use Case |
|---------|---------|----------|
| `riverpod` | State management | App-wide state |
| `go_router` | Navigation | Declarative routing |
| `freezed` | Code generation | Immutable models |
| `dio` | HTTP client | API requests |
| `hive` | Local database | Offline storage |
| `cached_network_image` | Image caching | Network images |

### Performance Checklist

- [ ] Use `const` constructors everywhere possible
- [ ] Extract widgets for reusability (build methods < 20 lines)
- [ ] Use `ListView.builder` for long lists
- [ ] Implement proper image caching
- [ ] Use keys for list items that can be reordered
- [ ] Profile with Flutter DevTools before optimizing
- [ ] Lazy load data and images
- [ ] Use `RepaintBoundary` for expensive widgets

### Testing Checklist

- [ ] Unit tests for business logic (70% coverage)
- [ ] Widget tests for UI components (50% coverage)
- [ ] Integration tests for critical flows (3-5 key journeys)
- [ ] Golden file tests for visual regression
- [ ] Mock external dependencies

---

## Anti-Patterns to Avoid

### ❌ Don't: Build everything in one widget

```dart
// Bad: 200-line build method
class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // 200 lines of nested widgets...
    );
  }
}
```

### ✅ Do: Extract into composable widgets

```dart
// Good: Extracted widgets
class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const HomeAppBar(),
      body: HomeBody(),
      bottomNavigationBar: const HomeBottomNav(),
    );
  }
}
```

### ❌ Don't: Use `setState` for complex state

```dart
// Bad: Complex state in StatefulWidget
class ComplexWidget extends StatefulWidget {
  @override
  State<ComplexWidget> createState() => _ComplexWidgetState();
}

class _ComplexWidgetState extends State<ComplexWidget> {
  User? user;
  List<Post> posts = [];
  bool isLoading = false;
  String? error;

  // Many setState calls, hard to maintain
}
```

### ✅ Do: Use proper state management

```dart
// Good: State management with Riverpod
final userProvider = FutureProvider<User>((ref) async {
  return await ref.watch(userRepositoryProvider).getUser();
});

class ComplexWidget extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userAsync = ref.watch(userProvider);

    return userAsync.when(
      data: (user) => UserView(user: user),
      loading: () => const LoadingIndicator(),
      error: (error, stack) => ErrorView(error: error),
    );
  }
}
```

---

## Learning Resources

- **Official Docs**: https://docs.flutter.dev
- **Riverpod Docs**: https://riverpod.dev
- **Flutter Cookbook**: https://docs.flutter.dev/cookbook
- **Widget Catalog**: https://docs.flutter.dev/ui/widgets

---

**Skill Version**: 1.0.0
**Last Updated**: October 27, 2024
**Difficulty**: Intermediate
**Estimated Time**: 4-6 hours

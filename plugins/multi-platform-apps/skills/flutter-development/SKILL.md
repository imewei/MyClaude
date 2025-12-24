---
name: flutter-development
version: "1.0.6"
maturity: "5-Expert"
specialization: Flutter/Dart Development
description: Comprehensive Flutter development patterns for mobile, web, and desktop apps. Use when writing Flutter/Dart code, implementing state management (Riverpod, Bloc), creating widgets, optimizing performance, handling navigation, integrating APIs, implementing offline persistence, or writing Flutter tests.
---

# Flutter Development

Production-ready Flutter patterns and best practices.

---

## State Management Selection

| Complexity | Solution | Use Case |
|------------|----------|----------|
| Simple | StatefulWidget | Local form state |
| Medium | Provider/Riverpod | Small apps |
| Complex | Riverpod + Repository | E-commerce, social |
| Enterprise | Bloc + Clean Architecture | Banking, healthcare |

---

## Widget Patterns

```dart
// ✅ Good: Use const constructors
class UserCard extends StatelessWidget {
  const UserCard({super.key, required this.name});
  final String name;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: const Icon(Icons.person),  // const!
        title: Text(name),
      ),
    );
  }
}

// ✅ Good: Use keys for lists
ListView.builder(
  itemCount: items.length,
  itemBuilder: (context, index) => ItemCard(
    key: ValueKey(items[index].id),
    item: items[index],
  ),
);
```

---

## Riverpod State Management

```dart
// 1. Define providers
final counterProvider = StateProvider<int>((ref) => 0);

final userProvider = FutureProvider<User>((ref) async {
  return await ref.watch(userRepositoryProvider).getUser();
});

// 2. Use in widgets
class Counter extends ConsumerWidget {
  const Counter({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final count = ref.watch(counterProvider);

    return ElevatedButton(
      onPressed: () => ref.read(counterProvider.notifier).state++,
      child: Text('Count: $count'),
    );
  }
}

// 3. Wrap app
void main() => runApp(const ProviderScope(child: MyApp()));
```

---

## Navigation (go_router)

```dart
final router = GoRouter(
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => const HomeScreen(),
      routes: [
        GoRoute(
          path: 'details/:id',
          builder: (context, state) => DetailScreen(
            id: state.pathParameters['id']!,
          ),
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

## Async Data Loading

```dart
class UserProfile extends ConsumerWidget {
  const UserProfile({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userAsync = ref.watch(userProvider);

    return userAsync.when(
      data: (user) => UserCard(user: user),
      loading: () => const CircularProgressIndicator(),
      error: (e, _) => Text('Error: $e'),
    );
  }
}
```

---

## Async State Pattern

```dart
// Sealed class for state
sealed class DataState<T> {}
class Loading<T> extends DataState<T> {}
class Success<T> extends DataState<T> { final T data; Success(this.data); }
class Error<T> extends DataState<T> { final String msg; Error(this.msg); }

// Pattern matching (Dart 3)
Widget build(context, ref) {
  final state = ref.watch(dataProvider);
  return switch (state) {
    Loading() => const CircularProgressIndicator(),
    Success(:final data) => ContentWidget(data: data),
    Error(:final msg) => Text('Error: $msg'),
  };
}
```

---

## Performance Optimization

```dart
// ✅ Use const widgets
const Text('Static');
const Icon(Icons.star);

// ✅ Extract widgets (keep build < 20 lines)
class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: const HomeAppBar(),
    body: const HomeBody(),
  );
}

// ✅ Use ListView.builder for long lists
ListView.builder(
  itemCount: items.length,
  cacheExtent: 100,
  itemBuilder: (_, i) => ItemCard(item: items[i]),
);

// ✅ Cache network images
CachedNetworkImage(
  imageUrl: url,
  memCacheWidth: 200,
  placeholder: (_, __) => const CircularProgressIndicator(),
);
```

---

## Repository Pattern

```dart
abstract class UserRepository {
  Future<User> getUser(String id);
  Future<void> updateUser(User user);
}

class ApiUserRepository implements UserRepository {
  ApiUserRepository(this.client);
  final ApiClient client;

  @override
  Future<User> getUser(String id) async {
    final response = await client.get('/users/$id');
    return User.fromJson(response.data);
  }

  @override
  Future<void> updateUser(User user) async {
    await client.put('/users/${user.id}', data: user.toJson());
  }
}

// Provider
final userRepositoryProvider = Provider<UserRepository>((ref) {
  return ApiUserRepository(ref.watch(apiClientProvider));
});
```

---

## Form Validation

```dart
class LoginForm extends StatefulWidget {
  @override
  State<LoginForm> createState() => _LoginFormState();
}

class _LoginFormState extends State<LoginForm> {
  final _formKey = GlobalKey<FormState>();
  final _email = TextEditingController();
  final _password = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          TextFormField(
            controller: _email,
            validator: (v) => v?.contains('@') != true ? 'Invalid email' : null,
          ),
          TextFormField(
            controller: _password,
            obscureText: true,
            validator: (v) => (v?.length ?? 0) < 8 ? 'Min 8 chars' : null,
          ),
          ElevatedButton(
            onPressed: () {
              if (_formKey.currentState!.validate()) { /* submit */ }
            },
            child: const Text('Login'),
          ),
        ],
      ),
    );
  }
}
```

---

## Widget Testing

```dart
void main() {
  testWidgets('Counter increments', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: Counter()));

    expect(find.text('Count: 0'), findsOneWidget);

    await tester.tap(find.text('Increment'));
    await tester.pump();

    expect(find.text('Count: 1'), findsOneWidget);
  });
}
```

---

## Essential Packages

| Package | Purpose |
|---------|---------|
| riverpod | State management |
| go_router | Navigation |
| freezed | Immutable models |
| dio | HTTP client |
| hive | Local database |
| cached_network_image | Image caching |

---

## Project Structure

```
lib/
├── core/           # Constants, theme, utils
├── features/
│   └── user/
│       ├── data/        # Models, repositories
│       ├── domain/      # Entities, use cases
│       └── presentation/  # Pages, widgets, providers
└── main.dart
```

---

## Checklist

- [ ] Use `const` constructors everywhere possible
- [ ] Extract widgets (build < 20 lines)
- [ ] Use `ListView.builder` for long lists
- [ ] Implement proper state management
- [ ] Cache network images
- [ ] Use keys for reorderable lists
- [ ] Write widget tests (50% coverage)
- [ ] Profile with Flutter DevTools

---

**Version**: 1.0.5

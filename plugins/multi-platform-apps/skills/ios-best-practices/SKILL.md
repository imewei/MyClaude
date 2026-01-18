---
name: ios-best-practices
version: "2.1.0"
maturity: "5-Expert"
specialization: Native iOS Development
description: Native iOS development with Swift 6, SwiftUI, MVVM, Swift concurrency (async/await, actors), Core Data/SwiftData, NavigationStack, and XCTest. Use when building production iOS apps, implementing Swift concurrency, or preparing for App Store.
---

# iOS Development Best Practices

Modern iOS with SwiftUI, Swift 6, and production patterns.

---

## Architecture Overview

| Layer | Responsibility | Components |
|-------|---------------|------------|
| Presentation | UI, user interaction | SwiftUI Views, ViewModels |
| Domain | Business logic | Entities, Use Cases |
| Data | Persistence, networking | Repositories, Data Sources |

---

## SwiftUI View Pattern

```swift
struct UserCard: View {
    let name: String
    let email: String
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                Circle()
                    .fill(Color.blue)
                    .frame(width: 48, height: 48)
                    .overlay(Text(name.prefix(1)).foregroundColor(.white))
                VStack(alignment: .leading) {
                    Text(name).font(.headline)
                    Text(email).font(.subheadline).foregroundColor(.secondary)
                }
                Spacer()
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(12)
        }
        .buttonStyle(.plain)
    }
}
```

---

## MVVM with ObservableObject

```swift
@MainActor
class UserViewModel: ObservableObject {
    @Published var user: User?
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let repository: UserRepository

    init(repository: UserRepository = UserRepositoryImpl()) {
        self.repository = repository
    }

    func loadUser(id: String) async {
        isLoading = true
        errorMessage = nil
        do {
            user = try await repository.fetchUser(id: id)
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}

struct UserProfileView: View {
    @StateObject private var viewModel = UserViewModel()
    let userId: String

    var body: some View {
        Group {
            if viewModel.isLoading { ProgressView() }
            else if let error = viewModel.errorMessage { Text(error) }
            else if let user = viewModel.user { UserDetailView(user: user) }
        }
        .task { await viewModel.loadUser(id: userId) }
    }
}
```

---

## Swift Concurrency

```swift
// Actor for thread-safe state
actor UserRepository {
    private var cache: [String: User] = [:]

    func fetchUser(id: String) async throws -> User {
        if let cached = cache[id] { return cached }

        let url = URL(string: "https://api.example.com/users/\(id)")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let user = try JSONDecoder().decode(User.self, from: data)
        cache[id] = user
        return user
    }
}

// Parallel fetch with TaskGroup
func fetchMultipleUsers(ids: [String]) async throws -> [User] {
    try await withThrowingTaskGroup(of: User.self) { group in
        for id in ids { group.addTask { try await fetchUser(id: id) } }
        return try await group.reduce(into: []) { $0.append($1) }
    }
}
```

---

## SwiftData Persistence

```swift
@Model
class Task {
    var title: String
    var isCompleted: Bool
    var createdAt: Date

    init(title: String) {
        self.title = title
        self.isCompleted = false
        self.createdAt = Date()
    }
}

struct TaskListView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Task.createdAt, order: .reverse) private var tasks: [Task]

    var body: some View {
        List {
            ForEach(tasks) { task in TaskRow(task: task) }
                .onDelete { for i in $0 { modelContext.delete(tasks[i]) } }
        }
        .toolbar {
            Button("Add") { modelContext.insert(Task(title: "New Task")) }
        }
    }
}
```

---

## Type-Safe Navigation

```swift
enum Route: Hashable {
    case profile(userId: String)
    case settings
    case detail(itemId: Int)
}

struct ContentView: View {
    @State private var path: [Route] = []

    var body: some View {
        NavigationStack(path: $path) {
            HomeView()
                .navigationDestination(for: Route.self) { route in
                    switch route {
                    case .profile(let id): ProfileView(userId: id)
                    case .settings: SettingsView()
                    case .detail(let id): DetailView(itemId: id)
                    }
                }
        }
    }
}
```

---

## Testing

```swift
final class UserViewModelTests: XCTestCase {
    var sut: UserViewModel!
    var mockRepository: MockUserRepository!

    override func setUp() {
        mockRepository = MockUserRepository()
        sut = UserViewModel(repository: mockRepository)
    }

    func testLoadUserSuccess() async {
        mockRepository.mockUser = User(id: "1", name: "Test")
        await sut.loadUser(id: "1")
        XCTAssertEqual(sut.user?.id, "1")
        XCTAssertFalse(sut.isLoading)
        XCTAssertNil(sut.errorMessage)
    }
}
```

---

## State Management

| Property Wrapper | Use Case |
|-----------------|----------|
| `@State` | Simple local state |
| `@StateObject` | ViewModel ownership |
| `@ObservedObject` | ViewModel from parent |
| `@EnvironmentObject` | Shared app-wide state |
| `@Binding` | Two-way child binding |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| @MainActor | All ViewModels |
| Async/await | All network calls |
| Actors | Shared mutable state |
| Dependency injection | Repository pattern |
| SwiftData | Persistence over Core Data |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| @ObservedObject for owned | Use @StateObject |
| Blocking main thread | Use async/await |
| Force unwrapping optionals | Use guard/if-let |
| Missing error handling | Always handle async errors |

---

## Parallelization Strategies

| Feature | Implementation | Use Case |
|---------|----------------|----------|
| **TaskGroup** | `withTaskGroup` | Parallel fetching of independent resources |
| **Async Let** | `async let x = ...` | Concurrent execution of fixed tasks |
| **Actors** | `actor DataStore` | Thread-safe mutable state without locks |
| **Detached Tasks** | `Task.detached` | Background work independent of view |

---

## Checklist

- [ ] MVVM architecture with @StateObject
- [ ] Swift concurrency (async/await, actors)
- [ ] Type-safe navigation with NavigationStack
- [ ] SwiftData for persistence
- [ ] Unit tests for ViewModels
- [ ] Accessibility labels added

---

**Version**: 1.0.5

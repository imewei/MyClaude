# iOS Development Best Practices

> **Modern iOS development with SwiftUI, Swift 6, and production-ready patterns for App Store applications.**

---

## Skill Overview

Comprehensive guide for native iOS development covering SwiftUI patterns, Swift concurrency, data persistence, and App Store optimization.

**Target Audience**: iOS developers or teams building native Apple platform apps

**Estimated Learning Time**: 5-6 hours to master core concepts

---

## Core Concepts

### 1. SwiftUI View Patterns

```swift
import SwiftUI

// ✅ Good: Extracted, reusable view
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
                    .overlay(
                        Text(name.prefix(1))
                            .foregroundColor(.white)
                            .font(.title2)
                    )

                VStack(alignment: .leading, spacing: 4) {
                    Text(name)
                        .font(.headline)
                    Text(email)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
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

### 2. MVVM with ObservableObject

```swift
// ViewModel
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

// View
struct UserProfileView: View {
    @StateObject private var viewModel = UserViewModel()
    let userId: String

    var body: some View {
        Group {
            if viewModel.isLoading {
                ProgressView()
            } else if let error = viewModel.errorMessage {
                ErrorView(message: error)
            } else if let user = viewModel.user {
                UserDetailView(user: user)
            }
        }
        .task {
            await viewModel.loadUser(id: userId)
        }
    }
}
```

### 3. Swift Concurrency

```swift
// ✅ Good: Async/await with actors
actor UserRepository {
    private var cache: [String: User] = [:]

    func fetchUser(id: String) async throws -> User {
        // Check cache
        if let cached = cache[id] {
            return cached
        }

        // Fetch from API
        let url = URL(string: "https://api.example.com/users/\(id)")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let user = try JSONDecoder().decode(User.self, from: data)

        // Cache result
        cache[id] = user

        return user
    }
}

// Task groups for parallel operations
func fetchMultipleUsers(ids: [String]) async throws -> [User] {
    try await withThrowingTaskGroup(of: User.self) { group in
        for id in ids {
            group.addTask {
                try await fetchUser(id: id)
            }
        }

        var users: [User] = []
        for try await user in group {
            users.append(user)
        }
        return users
    }
}
```

### 4. Core Data with SwiftUI

```swift
// Core Data model
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

// SwiftUI view with Core Data
struct TaskListView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Task.createdAt, order: .reverse) private var tasks: [Task]

    var body: some View {
        List {
            ForEach(tasks) { task in
                TaskRow(task: task)
            }
            .onDelete(perform: deleteTasks)
        }
        .toolbar {
            Button("Add") {
                addTask()
            }
        }
    }

    private func addTask() {
        let task = Task(title: "New Task")
        modelContext.insert(task)
    }

    private func deleteTasks(at offsets: IndexSet) {
        for index in offsets {
            modelContext.delete(tasks[index])
        }
    }
}
```

### 5. Navigation with NavigationStack

```swift
// ✅ Good: Type-safe navigation
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
                    case .profile(let userId):
                        ProfileView(userId: userId)
                    case .settings:
                        SettingsView()
                    case .detail(let itemId):
                        DetailView(itemId: itemId)
                    }
                }
        }
    }
}
```

### 6. Testing

```swift
import XCTest
@testable import MyApp

final class UserViewModelTests: XCTestCase {
    var sut: UserViewModel!
    var mockRepository: MockUserRepository!

    override func setUp() {
        super.setUp()
        mockRepository = MockUserRepository()
        sut = UserViewModel(repository: mockRepository)
    }

    func testLoadUserSuccess() async throws {
        // Given
        let expectedUser = User(id: "1", name: "Test User")
        mockRepository.mockUser = expectedUser

        // When
        await sut.loadUser(id: "1")

        // Then
        XCTAssertEqual(sut.user?.id, expectedUser.id)
        XCTAssertFalse(sut.isLoading)
        XCTAssertNil(sut.errorMessage)
    }

    func testLoadUserFailure() async throws {
        // Given
        mockRepository.shouldFail = true

        // When
        await sut.loadUser(id: "1")

        // Then
        XCTAssertNil(sut.user)
        XCTAssertFalse(sut.isLoading)
        XCTAssertNotNil(sut.errorMessage)
    }
}
```

---

## Architecture Best Practices

### Clean Architecture Structure

```
MyApp/
├── Features/
│   ├── User/
│   │   ├── Domain/
│   │   │   ├── Entities/
│   │   │   │   └── User.swift
│   │   │   └── Repositories/
│   │   │       └── UserRepository.swift
│   │   ├── Data/
│   │   │   ├── Repositories/
│   │   │   │   └── UserRepositoryImpl.swift
│   │   │   └── DataSources/
│   │   │       └── UserAPIDataSource.swift
│   │   └── Presentation/
│   │       ├── ViewModels/
│   │       │   └── UserViewModel.swift
│   │       └── Views/
│   │           └── UserProfileView.swift
│   └── Posts/
├── Core/
│   ├── Networking/
│   ├── Storage/
│   └── Extensions/
└── App/
    └── MyApp.swift
```

---

## Performance Optimization

- Use `@State` for simple local state
- Use `@StateObject` for view models
- Use `@ObservedObject` when passed from parent
- Minimize view updates with `equatable`
- Use `@ViewBuilder` for conditional views
- Profile with Instruments regularly

---

## Quick Reference

### Essential Packages

- **SwiftUI**: Native UI framework
- **Combine**: Reactive programming
- **Core Data / SwiftData**: Persistence
- **URLSession**: Networking
- **XCTest**: Testing

---

**Skill Version**: 1.0.0
**Last Updated**: October 27, 2024
**Difficulty**: Intermediate
**Estimated Time**: 5-6 hours

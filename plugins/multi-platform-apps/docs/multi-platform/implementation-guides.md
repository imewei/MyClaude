# Implementation Guides

Platform-specific implementation patterns for React/Next.js, SwiftUI, Jetpack Compose, Flutter, and Electron/Tauri with production-ready code examples and best practices.

## Table of Contents

1. [React/Next.js Web Implementation](#reactnextjs-web-implementation)
2. [SwiftUI iOS Implementation](#swiftui-ios-implementation)
3. [Jetpack Compose Android Implementation](#jetpack-compose-android-implementation)
4. [Flutter Cross-Platform Implementation](#flutter-cross-platform-implementation)
5. [Electron/Tauri Desktop Implementation](#electrontauri-desktop-implementation)

---

## React/Next.js Web Implementation

### Project Setup

**Next.js 15 App Router with TypeScript:**

```bash
npx create-next-app@latest my-app \
  --typescript \
  --tailwind \
  --app \
  --src-dir \
  --import-alias "@/*"

cd my-app

# Install essential dependencies
npm install \
  @tanstack/react-query \
  zustand \
  zod \
  react-hook-form \
  @hookform/resolvers \
  next-auth \
  @next/bundle-analyzer
```

**Project Structure:**

```
src/
├── app/                    # Next.js App Router
│   ├── (auth)/            # Route groups
│   │   ├── login/
│   │   └── register/
│   ├── (dashboard)/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── profile/
│   ├── api/               # API routes
│   │   ├── auth/
│   │   └── trpc/
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── ui/                # Shadcn components
│   ├── features/          # Feature-specific components
│   └── layouts/
├── lib/
│   ├── api/               # API clients
│   ├── hooks/             # Custom hooks
│   ├── utils/             # Utility functions
│   └── validations/       # Zod schemas
├── stores/                # Zustand stores
├── types/                 # TypeScript types
└── styles/
```

### Server Components and Data Fetching

**Server Component with Streaming:**

```typescript
// app/(dashboard)/profile/page.tsx
import { Suspense } from 'react';
import { ProfileHeader } from '@/components/features/profile/ProfileHeader';
import { ProfileFeed } from '@/components/features/profile/ProfileFeed';
import { ProfileSkeleton } from '@/components/features/profile/ProfileSkeleton';

interface ProfilePageProps {
  params: {
    userId: string;
  };
  searchParams: {
    tab?: string;
  };
}

export default async function ProfilePage({ params, searchParams }: ProfilePageProps) {
  // Fetch critical data in parallel
  const [user, stats] = await Promise.all([
    fetchUser(params.userId),
    fetchUserStats(params.userId)
  ]);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Critical content rendered immediately */}
      <ProfileHeader user={user} stats={stats} />

      {/* Non-critical content with Suspense boundary */}
      <Suspense fallback={<ProfileSkeleton />}>
        <ProfileFeed userId={params.userId} tab={searchParams.tab} />
      </Suspense>
    </div>
  );
}

// Streaming component
async function ProfileFeed({ userId, tab }: { userId: string; tab?: string }) {
  // This data fetch is streamed
  const feed = await fetchUserFeed(userId, { tab });

  return (
    <div className="mt-8">
      {feed.items.map(item => (
        <FeedItem key={item.id} item={item} />
      ))}
    </div>
  );
}
```

**Route Handlers (API Routes):**

```typescript
// app/api/users/[userId]/profile/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { auth } from '@/lib/auth';

const updateProfileSchema = z.object({
  displayName: z.string().max(100).optional(),
  bio: z.string().max(500).optional()
});

export async function GET(
  request: NextRequest,
  { params }: { params: { userId: string } }
) {
  try {
    const user = await db.user.findUnique({
      where: { id: params.userId },
      include: {
        preferences: true,
        stats: true
      }
    });

    if (!user) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      );
    }

    return NextResponse.json(user);
  } catch (error) {
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { userId: string } }
) {
  try {
    // Authenticate
    const session = await auth();
    if (!session || session.user.id !== params.userId) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    // Parse and validate body
    const body = await request.json();
    const validation = updateProfileSchema.safeParse(body);

    if (!validation.success) {
      return NextResponse.json(
        { error: 'Validation failed', details: validation.error.errors },
        { status: 400 }
      );
    }

    // Update user
    const updatedUser = await db.user.update({
      where: { id: params.userId },
      data: validation.data
    });

    return NextResponse.json(updatedUser);
  } catch (error) {
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

### State Management with Zustand

**Global Store:**

```typescript
// stores/useUserStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface UserState {
  user: UserProfile | null;
  isAuthenticated: boolean;
  setUser: (user: UserProfile | null) => void;
  logout: () => void;
  updateProfile: (updates: Partial<UserProfile>) => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,

      setUser: (user) => set({
        user,
        isAuthenticated: !!user
      }),

      logout: () => set({
        user: null,
        isAuthenticated: false
      }),

      updateProfile: (updates) => set((state) => ({
        user: state.user ? { ...state.user, ...updates } : null
      }))
    }),
    {
      name: 'user-storage',
      storage: createJSONStorage(() => localStorage)
    }
  )
);
```

**Feature-Specific Store:**

```typescript
// stores/useFeedStore.ts
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

interface FeedState {
  items: ContentItem[];
  filter: 'all' | 'following' | 'trending';
  isLoading: boolean;
  hasMore: boolean;
  cursor: string | null;

  setItems: (items: ContentItem[]) => void;
  addItems: (items: ContentItem[]) => void;
  updateItem: (id: string, updates: Partial<ContentItem>) => void;
  removeItem: (id: string) => void;
  setFilter: (filter: FeedState['filter']) => void;
  reset: () => void;
}

export const useFeedStore = create<FeedState>()(
  immer((set) => ({
    items: [],
    filter: 'all',
    isLoading: false,
    hasMore: true,
    cursor: null,

    setItems: (items) => set((state) => {
      state.items = items;
    }),

    addItems: (items) => set((state) => {
      state.items.push(...items);
    }),

    updateItem: (id, updates) => set((state) => {
      const index = state.items.findIndex(item => item.id === id);
      if (index !== -1) {
        state.items[index] = { ...state.items[index], ...updates };
      }
    }),

    removeItem: (id) => set((state) => {
      state.items = state.items.filter(item => item.id !== id);
    }),

    setFilter: (filter) => set((state) => {
      state.filter = filter;
      state.items = [];
      state.cursor = null;
    }),

    reset: () => set({
      items: [],
      filter: 'all',
      isLoading: false,
      hasMore: true,
      cursor: null
    })
  }))
);
```

### API Integration with TanStack Query

**Query Client Setup:**

```typescript
// lib/queryClient.ts
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      gcTime: 1000 * 60 * 30, // 30 minutes (formerly cacheTime)
      refetchOnWindowFocus: false,
      retry: 1
    }
  }
});
```

**Custom Hooks with React Query:**

```typescript
// lib/hooks/useProfile.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

export function useProfile(userId: string) {
  return useQuery({
    queryKey: ['user', userId],
    queryFn: () => apiClient.getUser(userId),
    staleTime: 1000 * 60 * 5 // 5 minutes
  });
}

export function useUpdateProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { userId: string; displayName?: string; bio?: string }) =>
      apiClient.updateProfile(data.userId, data),

    onMutate: async (newData) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: ['user', newData.userId] });

      // Snapshot previous value
      const previousUser = queryClient.getQueryData(['user', newData.userId]);

      // Optimistically update
      queryClient.setQueryData(['user', newData.userId], (old: any) => ({
        ...old,
        ...newData
      }));

      return { previousUser };
    },

    onError: (err, newData, context) => {
      // Rollback on error
      queryClient.setQueryData(
        ['user', newData.userId],
        context?.previousUser
      );
    },

    onSuccess: (data, variables) => {
      // Invalidate and refetch
      queryClient.invalidateQueries({ queryKey: ['user', variables.userId] });
    }
  });
}
```

**Infinite Query for Pagination:**

```typescript
// lib/hooks/useFeed.ts
import { useInfiniteQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

export function useFeed(filter: 'all' | 'following' | 'trending' = 'all') {
  return useInfiniteQuery({
    queryKey: ['feed', filter],
    queryFn: ({ pageParam }) =>
      apiClient.getFeed({ filter, cursor: pageParam, limit: 20 }),
    initialPageParam: null,
    getNextPageParam: (lastPage) =>
      lastPage.pagination.hasMore ? lastPage.pagination.nextCursor : undefined
  });
}

// Usage in component
function FeedComponent() {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage
  } = useFeed('all');

  const items = data?.pages.flatMap(page => page.data) ?? [];

  return (
    <div>
      {items.map(item => (
        <FeedItem key={item.id} item={item} />
      ))}

      {hasNextPage && (
        <button
          onClick={() => fetchNextPage()}
          disabled={isFetchingNextPage}
        >
          {isFetchingNextPage ? 'Loading...' : 'Load More'}
        </button>
      )}
    </div>
  );
}
```

### Form Handling with React Hook Form

**Complex Form with Validation:**

```typescript
// components/features/profile/ProfileForm.tsx
'use client';

import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useUpdateProfile } from '@/lib/hooks/useProfile';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';

const profileSchema = z.object({
  displayName: z.string()
    .max(100, 'Display name must be 100 characters or less')
    .optional(),
  bio: z.string()
    .max(500, 'Bio must be 500 characters or less')
    .optional(),
  preferences: z.object({
    theme: z.enum(['light', 'dark', 'auto']),
    notifications: z.object({
      email: z.boolean(),
      push: z.boolean(),
      inApp: z.boolean()
    })
  })
});

type ProfileFormData = z.infer<typeof profileSchema>;

export function ProfileForm({ userId, initialData }: {
  userId: string;
  initialData: ProfileFormData;
}) {
  const { mutate: updateProfile, isPending } = useUpdateProfile();

  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors, isDirty }
  } = useForm<ProfileFormData>({
    resolver: zodResolver(profileSchema),
    defaultValues: initialData
  });

  const bio = watch('bio');

  const onSubmit = async (data: ProfileFormData) => {
    updateProfile({
      userId,
      ...data
    });
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      <div>
        <label htmlFor="displayName" className="block text-sm font-medium text-gray-700">
          Display Name
        </label>
        <Input
          {...register('displayName')}
          id="displayName"
          className="mt-1"
          aria-invalid={!!errors.displayName}
          aria-describedby={errors.displayName ? 'displayName-error' : undefined}
        />
        {errors.displayName && (
          <p id="displayName-error" className="mt-1 text-sm text-red-600">
            {errors.displayName.message}
          </p>
        )}
      </div>

      <div>
        <label htmlFor="bio" className="block text-sm font-medium text-gray-700">
          Bio
        </label>
        <Textarea
          {...register('bio')}
          id="bio"
          rows={4}
          className="mt-1"
          aria-invalid={!!errors.bio}
          aria-describedby="bio-info bio-error"
        />
        <div className="mt-1 flex justify-between text-sm">
          {errors.bio ? (
            <p id="bio-error" className="text-red-600">{errors.bio.message}</p>
          ) : (
            <span />
          )}
          <span id="bio-info" className="text-gray-500">
            {bio?.length || 0}/500
          </span>
        </div>
      </div>

      <div className="space-y-4">
        <h3 className="text-lg font-medium">Preferences</h3>

        <div>
          <label className="block text-sm font-medium text-gray-700">Theme</label>
          <Controller
            name="preferences.theme"
            control={control}
            render={({ field }) => (
              <select {...field} className="mt-1 block w-full rounded-md border-gray-300">
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="auto">Auto</option>
              </select>
            )}
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">Notifications</label>

          <div className="flex items-center">
            <input
              {...register('preferences.notifications.email')}
              type="checkbox"
              id="email-notifications"
              className="h-4 w-4 rounded border-gray-300"
            />
            <label htmlFor="email-notifications" className="ml-2 text-sm text-gray-700">
              Email notifications
            </label>
          </div>

          <div className="flex items-center">
            <input
              {...register('preferences.notifications.push')}
              type="checkbox"
              id="push-notifications"
              className="h-4 w-4 rounded border-gray-300"
            />
            <label htmlFor="push-notifications" className="ml-2 text-sm text-gray-700">
              Push notifications
            </label>
          </div>

          <div className="flex items-center">
            <input
              {...register('preferences.notifications.inApp')}
              type="checkbox"
              id="inApp-notifications"
              className="h-4 w-4 rounded border-gray-300"
            />
            <label htmlFor="inApp-notifications" className="ml-2 text-sm text-gray-700">
              In-app notifications
            </label>
          </div>
        </div>
      </div>

      <div className="flex justify-end space-x-3">
        <Button type="button" variant="outline" disabled={!isDirty || isPending}>
          Cancel
        </Button>
        <Button type="submit" disabled={!isDirty || isPending}>
          {isPending ? 'Saving...' : 'Save Changes'}
        </Button>
      </div>
    </form>
  );
}
```

### Progressive Web App (PWA) Setup

**next.config.js with PWA:**

```javascript
// next.config.js
const withPWA = require('@ducanh2912/next-pwa').default({
  dest: 'public',
  register: true,
  skipWaiting: true,
  disable: process.env.NODE_ENV === 'development',
  runtimeCaching: [
    {
      urlPattern: /^https:\/\/api\.example\.com\/.*$/,
      handler: 'NetworkFirst',
      options: {
        cacheName: 'api-cache',
        expiration: {
          maxEntries: 50,
          maxAgeSeconds: 5 * 60 // 5 minutes
        },
        networkTimeoutSeconds: 10
      }
    },
    {
      urlPattern: /\.(?:jpg|jpeg|png|gif|webp|svg)$/,
      handler: 'CacheFirst',
      options: {
        cacheName: 'image-cache',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 30 * 24 * 60 * 60 // 30 days
        }
      }
    }
  ]
});

module.exports = withPWA({
  // Next.js config
  reactStrictMode: true,
  swcMinify: true
});
```

**Web Manifest:**

```json
// public/manifest.json
{
  "name": "My App",
  "short_name": "App",
  "description": "My multi-platform application",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#000000",
  "orientation": "portrait-primary",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/icon-512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any maskable"
    }
  ]
}
```

---

## SwiftUI iOS Implementation

### Project Setup

**Xcode Project Structure:**

```
MyApp/
├── MyApp/
│   ├── MyAppApp.swift        # App entry point
│   ├── ContentView.swift
│   ├── Models/
│   │   ├── UserProfile.swift
│   │   └── ContentItem.swift
│   ├── ViewModels/
│   │   ├── ProfileViewModel.swift
│   │   └── FeedViewModel.swift
│   ├── Views/
│   │   ├── Profile/
│   │   ├── Feed/
│   │   └── Components/
│   ├── Services/
│   │   ├── APIService.swift
│   │   ├── CacheService.swift
│   │   └── AuthService.swift
│   ├── Utilities/
│   │   ├── Extensions/
│   │   └── Helpers/
│   └── Resources/
│       ├── Assets.xcassets
│       └── Localizable.strings
├── MyAppTests/
└── MyAppUITests/
```

### Models and Codable

**Domain Models:**

```swift
// Models/UserProfile.swift
import Foundation

struct UserProfile: Codable, Identifiable, Equatable {
    let id: String
    let username: String
    let email: String
    var displayName: String?
    var bio: String?
    var avatarUrl: String?
    var preferences: UserPreferences
    var stats: UserStats
    let createdAt: Date
    var updatedAt: Date

    enum CodingKeys: String, CodingKey {
        case id, username, email, displayName, bio, avatarUrl, preferences, stats
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

struct UserPreferences: Codable, Equatable {
    var theme: Theme
    var notifications: NotificationSettings
    var privacy: PrivacySettings

    enum Theme: String, Codable {
        case light, dark, auto
    }
}

struct NotificationSettings: Codable, Equatable {
    var email: Bool
    var push: Bool
    var inApp: Bool
}

struct PrivacySettings: Codable, Equatable {
    var profileVisibility: Visibility
    var showEmail: Bool

    enum Visibility: String, Codable {
        case public = "PUBLIC"
        case friends = "FRIENDS"
        case private = "PRIVATE"
    }
}

struct UserStats: Codable, Equatable {
    let followersCount: Int
    let followingCount: Int
    let postsCount: Int
    let engagementRate: Double
}
```

### Networking with URLSession and Async/Await

**API Service:**

```swift
// Services/APIService.swift
import Foundation

enum APIError: Error {
    case invalidURL
    case requestFailed(Error)
    case invalidResponse
    case decodingFailed(Error)
    case serverError(statusCode: Int, message: String)
    case notFound
}

actor APIService {
    private let baseURL = "https://api.example.com/v1"
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.waitsForConnectivity = true
        self.session = URLSession(configuration: config)

        self.decoder = JSONDecoder()
        self.decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.decoder.dateDecodingStrategy = .iso8601

        self.encoder = JSONEncoder()
        self.encoder.keyEncodingStrategy = .convertToSnakeCase
        self.encoder.dateEncodingStrategy = .iso8601
    }

    func getProfile(userId: String) async throws -> UserProfile {
        let endpoint = "\(baseURL)/users/\(userId)"
        guard let url = URL(string: endpoint) else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Add auth token if available
        if let token = await getAuthToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        do {
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }

            switch httpResponse.statusCode {
            case 200:
                return try decoder.decode(UserProfile.self, from: data)
            case 404:
                throw APIError.notFound
            case 400...499:
                let errorMessage = String(data: data, encoding: .utf8) ?? "Client error"
                throw APIError.serverError(statusCode: httpResponse.statusCode, message: errorMessage)
            case 500...599:
                let errorMessage = String(data: data, encoding: .utf8) ?? "Server error"
                throw APIError.serverError(statusCode: httpResponse.statusCode, message: errorMessage)
            default:
                throw APIError.invalidResponse
            }
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.requestFailed(error)
        }
    }

    func updateProfile(
        userId: String,
        displayName: String?,
        bio: String?
    ) async throws -> UserProfile {
        let endpoint = "\(baseURL)/users/\(userId)"
        guard let url = URL(string: endpoint) else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "PATCH"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let token = await getAuthToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let body: [String: Any?] = [
            "displayName": displayName,
            "bio": bio
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body.compactMapValues { $0 })

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }

        guard httpResponse.statusCode == 200 else {
            let errorMessage = String(data: data, encoding: .utf8) ?? "Update failed"
            throw APIError.serverError(statusCode: httpResponse.statusCode, message: errorMessage)
        }

        return try decoder.decode(UserProfile.self, from: data)
    }

    private func getAuthToken() async -> String? {
        // Retrieve from Keychain or AuthService
        return await AuthService.shared.getAccessToken()
    }
}
```

### MVVM Architecture with @Observable

**ViewModel:**

```swift
// ViewModels/ProfileViewModel.swift
import SwiftUI
import Observation

@Observable
class ProfileViewModel {
    var profile: UserProfile?
    var isLoading = false
    var error: String?

    // Form fields
    var displayName = ""
    var bio = ""

    // Validation errors
    var displayNameError: String?
    var bioError: String?

    private let apiService = APIService()
    private let cacheService = CacheService.shared
    private let userId: String

    init(userId: String) {
        self.userId = userId
    }

    @MainActor
    func loadProfile() async {
        isLoading = true
        error = nil

        // Check cache first
        if let cached = await cacheService.getProfile(userId: userId) {
            self.profile = cached
            self.displayName = cached.displayName ?? ""
            self.bio = cached.bio ?? ""
        }

        do {
            let profile = try await apiService.getProfile(userId: userId)
            self.profile = profile
            self.displayName = profile.displayName ?? ""
            self.bio = profile.bio ?? ""

            await cacheService.cacheProfile(profile)
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    @MainActor
    func updateProfile() async {
        guard validate() else { return }

        isLoading = true
        error = nil

        do {
            let updated = try await apiService.updateProfile(
                userId: userId,
                displayName: displayName.isEmpty ? nil : displayName,
                bio: bio.isEmpty ? nil : bio
            )

            self.profile = updated
            await cacheService.cacheProfile(updated)
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    func validate() -> Bool {
        displayNameError = nil
        bioError = nil

        if displayName.count > 100 {
            displayNameError = "Display name must be 100 characters or less"
            return false
        }

        if bio.count > 500 {
            bioError = "Bio must be 500 characters or less"
            return false
        }

        return true
    }
}
```

### SwiftUI Views

**Profile Form View:**

```swift
// Views/Profile/ProfileFormView.swift
import SwiftUI

struct ProfileFormView: View {
    @State private var viewModel: ProfileViewModel
    @Environment(\.dismiss) private var dismiss
    @FocusState private var focusedField: Field?

    enum Field: Hashable {
        case displayName
        case bio
    }

    init(userId: String) {
        _viewModel = State(wrappedValue: ProfileViewModel(userId: userId))
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Display Name", text: $viewModel.displayName)
                        .focused($focusedField, equals: .displayName)
                        .textContentType(.name)
                        .autocapitalization(.words)
                        .disabled(viewModel.isLoading)

                    if let error = viewModel.displayNameError {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                } header: {
                    Text("Display Name")
                }

                Section {
                    TextEditor(text: $viewModel.bio)
                        .focused($focusedField, equals: .bio)
                        .frame(minHeight: 100)
                        .disabled(viewModel.isLoading)

                    if let error = viewModel.bioError {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                } header: {
                    Text("Bio")
                } footer: {
                    HStack {
                        Spacer()
                        Text("\(viewModel.bio.count)/500")
                            .font(.caption)
                            .foregroundStyle(viewModel.bio.count > 500 ? .red : .secondary)
                    }
                }

                if let error = viewModel.error {
                    Section {
                        Text(error)
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("Edit Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .disabled(viewModel.isLoading)
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        Task {
                            await viewModel.updateProfile()
                            dismiss()
                        }
                    }
                    .disabled(viewModel.isLoading || viewModel.bio.count > 500)
                }

                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        focusedField = nil
                    }
                }
            }
            .overlay {
                if viewModel.isLoading {
                    ProgressView()
                        .scaleEffect(1.5)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color.black.opacity(0.2))
                }
            }
            .task {
                await viewModel.loadProfile()
            }
        }
    }
}
```

### Core Data Integration

**Core Data Stack:**

```swift
// Services/CoreDataStack.swift
import CoreData

class CoreDataStack {
    static let shared = CoreDataStack()

    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "MyApp")
        container.loadPersistentStores { description, error in
            if let error = error {
                fatalError("Unable to load persistent stores: \(error)")
            }
        }
        container.viewContext.automaticallyMergesChangesFromParent = true
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
        return container
    }()

    var viewContext: NSManagedObjectContext {
        persistentContainer.viewContext
    }

    func save() {
        let context = viewContext
        if context.hasChanges {
            do {
                try context.save()
            } catch {
                print("Error saving context: \(error)")
            }
        }
    }
}
```

**Cache Service with Core Data:**

```swift
// Services/CacheService.swift
import CoreData

actor CacheService {
    static let shared = CacheService()
    private let context = CoreDataStack.shared.viewContext

    func cacheProfile(_ profile: UserProfile) async {
        let fetchRequest: NSFetchRequest<UserProfileEntity> = UserProfileEntity.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "id == %@", profile.id)

        do {
            let results = try context.fetch(fetchRequest)
            let entity = results.first ?? UserProfileEntity(context: context)

            entity.id = profile.id
            entity.username = profile.username
            entity.email = profile.email
            entity.displayName = profile.displayName
            entity.bio = profile.bio
            entity.avatarUrl = profile.avatarUrl
            entity.cachedAt = Date()

            // Encode preferences and stats as JSON
            let encoder = JSONEncoder()
            entity.preferencesData = try? encoder.encode(profile.preferences)
            entity.statsData = try? encoder.encode(profile.stats)

            try context.save()
        } catch {
            print("Failed to cache profile: \(error)")
        }
    }

    func getProfile(userId: String) async -> UserProfile? {
        let fetchRequest: NSFetchRequest<UserProfileEntity> = UserProfileEntity.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "id == %@", userId)
        fetchRequest.fetchLimit = 1

        do {
            let results = try context.fetch(fetchRequest)
            guard let entity = results.first else { return nil }

            // Check if cache is stale (older than 5 minutes)
            if let cachedAt = entity.cachedAt,
               Date().timeIntervalSince(cachedAt) > 5 * 60 {
                return nil
            }

            // Decode preferences and stats
            let decoder = JSONDecoder()
            let preferences = try? entity.preferencesData.flatMap {
                try decoder.decode(UserPreferences.self, from: $0)
            }
            let stats = try? entity.statsData.flatMap {
                try decoder.decode(UserStats.self, from: $0)
            }

            return UserProfile(
                id: entity.id ?? "",
                username: entity.username ?? "",
                email: entity.email ?? "",
                displayName: entity.displayName,
                bio: entity.bio,
                avatarUrl: entity.avatarUrl,
                preferences: preferences ?? UserPreferences(
                    theme: .auto,
                    notifications: NotificationSettings(email: true, push: true, inApp: true),
                    privacy: PrivacySettings(profileVisibility: .public, showEmail: false)
                ),
                stats: stats ?? UserStats(
                    followersCount: 0,
                    followingCount: 0,
                    postsCount: 0,
                    engagementRate: 0
                ),
                createdAt: Date(),
                updatedAt: Date()
            )
        } catch {
            print("Failed to fetch cached profile: \(error)")
            return nil
        }
    }
}
```

---

## Jetpack Compose Android Implementation

### Project Setup

**Gradle Configuration:**

```kotlin
// build.gradle.kts (app module)
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.dagger.hilt.android")
    id("kotlin-kapt")
    id("kotlinx-serialization")
}

android {
    namespace = "com.example.myapp"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.myapp"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.3"
    }
}

dependencies {
    // Compose BOM
    val composeBom = platform("androidx.compose:compose-bom:2024.01.00")
    implementation(composeBom)

    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")
    implementation("androidx.navigation:navigation-compose:2.7.6")

    // Hilt
    implementation("com.google.dagger:hilt-android:2.50")
    kapt("com.google.dagger:hilt-compiler:2.50")
    implementation("androidx.hilt:hilt-navigation-compose:1.1.0")

    // Networking
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.jakewharton.retrofit:retrofit2-kotlinx-serialization-converter:1.0.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    // Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.2")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Room
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    kapt("androidx.room:room-compiler:2.6.1")

    // DataStore
    implementation("androidx.datastore:datastore-preferences:1.0.0")

    // Coil for image loading
    implementation("io.coil-kt:coil-compose:2.5.0")
}
```

### Models and Serialization

**Domain Models:**

```kotlin
// data/models/UserProfile.kt
package com.example.myapp.data.models

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class UserProfile(
    val id: String,
    val username: String,
    val email: String,
    val displayName: String? = null,
    val bio: String? = null,
    val avatarUrl: String? = null,
    val preferences: UserPreferences,
    val stats: UserStats,
    @SerialName("created_at") val createdAt: String,
    @SerialName("updated_at") val updatedAt: String
)

@Serializable
data class UserPreferences(
    val theme: Theme = Theme.AUTO,
    val notifications: NotificationSettings = NotificationSettings(),
    val privacy: PrivacySettings = PrivacySettings()
)

@Serializable
enum class Theme {
    @SerialName("light") LIGHT,
    @SerialName("dark") DARK,
    @SerialName("auto") AUTO
}

@Serializable
data class NotificationSettings(
    val email: Boolean = true,
    val push: Boolean = true,
    val inApp: Boolean = true
)

@Serializable
data class PrivacySettings(
    val profileVisibility: Visibility = Visibility.PUBLIC,
    val showEmail: Boolean = false
)

@Serializable
enum class Visibility {
    @SerialName("PUBLIC") PUBLIC,
    @SerialName("FRIENDS") FRIENDS,
    @SerialName("PRIVATE") PRIVATE
}

@Serializable
data class UserStats(
    val followersCount: Int,
    val followingCount: Int,
    val postsCount: Int,
    val engagementRate: Double
)
```

### Networking with Retrofit and Kotlin Serialization

**API Service:**

```kotlin
// data/remote/ApiService.kt
package com.example.myapp.data.remote

import com.example.myapp.data.models.UserProfile
import kotlinx.serialization.Serializable
import retrofit2.Response
import retrofit2.http.*

interface ApiService {
    @GET("users/{userId}")
    suspend fun getProfile(@Path("userId") userId: String): Response<UserProfile>

    @PATCH("users/{userId}")
    suspend fun updateProfile(
        @Path("userId") userId: String,
        @Body request: UpdateProfileRequest
    ): Response<UserProfile>

    @GET("feed")
    suspend fun getFeed(
        @Query("cursor") cursor: String? = null,
        @Query("limit") limit: Int = 20,
        @Query("filter") filter: String = "all"
    ): Response<FeedResponse>
}

@Serializable
data class UpdateProfileRequest(
    val displayName: String? = null,
    val bio: String? = null
)

@Serializable
data class FeedResponse(
    val data: List<ContentItem>,
    val pagination: PaginationMetadata
)

@Serializable
data class ContentItem(
    val id: String,
    val type: String,
    val author: UserProfile,
    val content: String,
    val createdAt: String
)

@Serializable
data class PaginationMetadata(
    val nextCursor: String?,
    val hasMore: Boolean,
    val totalCount: Int
)
```

**Retrofit Setup with Hilt:**

```kotlin
// di/NetworkModule.kt
package com.example.myapp.di

import com.example.myapp.data.remote.ApiService
import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideJson(): Json = Json {
        ignoreUnknownKeys = true
        coerceInputValues = true
        isLenient = true
    }

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        return OkHttpClient.Builder()
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            })
            .addInterceptor { chain ->
                val request = chain.request().newBuilder()
                    .addHeader("Content-Type", "application/json")
                    // Add auth token if available
                    // .addHeader("Authorization", "Bearer $token")
                    .build()
                chain.proceed(request)
            }
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }

    @Provides
    @Singleton
    fun provideRetrofit(
        okHttpClient: OkHttpClient,
        json: Json
    ): Retrofit {
        val contentType = "application/json".toMediaType()
        return Retrofit.Builder()
            .baseUrl("https://api.example.com/v1/")
            .client(okHttpClient)
            .addConverterFactory(json.asConverterFactory(contentType))
            .build()
    }

    @Provides
    @Singleton
    fun provideApiService(retrofit: Retrofit): ApiService {
        return retrofit.create(ApiService::class.java)
    }
}
```

### Repository Pattern

**Repository:**

```kotlin
// data/repository/UserRepository.kt
package com.example.myapp.data.repository

import com.example.myapp.data.local.UserDao
import com.example.myapp.data.local.UserEntity
import com.example.myapp.data.models.UserProfile
import com.example.myapp.data.remote.ApiService
import com.example.myapp.data.remote.UpdateProfileRequest
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import javax.inject.Singleton

sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error(val message: String) : Result<Nothing>()
    object Loading : Result<Nothing>()
}

@Singleton
class UserRepository @Inject constructor(
    private val apiService: ApiService,
    private val userDao: UserDao
) {
    fun getProfile(userId: String): Flow<Result<UserProfile>> = flow {
        emit(Result.Loading)

        // Emit cached data first
        userDao.getUser(userId)?.let { cached ->
            if (System.currentTimeMillis() - cached.cachedAt < 5 * 60 * 1000) {
                emit(Result.Success(cached.toUserProfile()))
            }
        }

        // Fetch from API
        try {
            val response = apiService.getProfile(userId)
            if (response.isSuccessful) {
                val profile = response.body()!!
                // Cache the result
                userDao.insertUser(UserEntity.fromUserProfile(profile))
                emit(Result.Success(profile))
            } else {
                emit(Result.Error("Error: ${response.code()}"))
            }
        } catch (e: Exception) {
            emit(Result.Error(e.message ?: "Unknown error"))
        }
    }

    suspend fun updateProfile(
        userId: String,
        displayName: String?,
        bio: String?
    ): Result<UserProfile> {
        return try {
            val request = UpdateProfileRequest(displayName, bio)
            val response = apiService.updateProfile(userId, request)

            if (response.isSuccessful) {
                val profile = response.body()!!
                // Update cache
                userDao.insertUser(UserEntity.fromUserProfile(profile))
                Result.Success(profile)
            } else {
                Result.Error("Error: ${response.code()}")
            }
        } catch (e: Exception) {
            Result.Error(e.message ?: "Unknown error")
        }
    }
}
```

### ViewModel with StateFlow

**ViewModel:**

```kotlin
// ui/profile/ProfileViewModel.kt
package com.example.myapp.ui.profile

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.myapp.data.models.UserProfile
import com.example.myapp.data.repository.Result
import com.example.myapp.data.repository.UserRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ProfileUiState(
    val profile: UserProfile? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    val displayName: String = "",
    val bio: String = "",
    val displayNameError: String? = null,
    val bioError: String? = null
)

@HiltViewModel
class ProfileViewModel @Inject constructor(
    private val repository: UserRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(ProfileUiState())
    val uiState: StateFlow<ProfileUiState> = _uiState.asStateFlow()

    fun loadProfile(userId: String) {
        viewModelScope.launch {
            repository.getProfile(userId).collect { result ->
                when (result) {
                    is Result.Loading -> {
                        _uiState.update { it.copy(isLoading = true, error = null) }
                    }
                    is Result.Success -> {
                        _uiState.update {
                            it.copy(
                                profile = result.data,
                                displayName = result.data.displayName ?: "",
                                bio = result.data.bio ?: "",
                                isLoading = false,
                                error = null
                            )
                        }
                    }
                    is Result.Error -> {
                        _uiState.update {
                            it.copy(isLoading = false, error = result.message)
                        }
                    }
                }
            }
        }
    }

    fun updateDisplayName(value: String) {
        _uiState.update { it.copy(displayName = value, displayNameError = null) }
    }

    fun updateBio(value: String) {
        if (value.length <= 500) {
            _uiState.update { it.copy(bio = value, bioError = null) }
        }
    }

    fun updateProfile(userId: String) {
        if (!validate()) return

        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, error = null) }

            when (val result = repository.updateProfile(
                userId,
                _uiState.value.displayName.ifEmpty { null },
                _uiState.value.bio.ifEmpty { null }
            )) {
                is Result.Success -> {
                    _uiState.update {
                        it.copy(profile = result.data, isLoading = false)
                    }
                }
                is Result.Error -> {
                    _uiState.update {
                        it.copy(isLoading = false, error = result.message)
                    }
                }
                else -> {}
            }
        }
    }

    private fun validate(): Boolean {
        val state = _uiState.value
        var isValid = true

        if (state.displayName.length > 100) {
            _uiState.update {
                it.copy(displayNameError = "Display name must be 100 characters or less")
            }
            isValid = false
        }

        if (state.bio.length > 500) {
            _uiState.update {
                it.copy(bioError = "Bio must be 500 characters or less")
            }
            isValid = false
        }

        return isValid
    }
}
```

### Jetpack Compose UI

**Profile Screen:**

```kotlin
// ui/profile/ProfileScreen.kt
package com.example.myapp.ui.profile

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProfileScreen(
    userId: String,
    viewModel: ProfileViewModel = hiltViewModel(),
    onNavigateBack: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(userId) {
        viewModel.loadProfile(userId)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Edit Profile") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    TextButton(
                        onClick = { viewModel.updateProfile(userId) },
                        enabled = !uiState.isLoading
                    ) {
                        Text("Save")
                    }
                }
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                OutlinedTextField(
                    value = uiState.displayName,
                    onValueChange = viewModel::updateDisplayName,
                    label = { Text("Display Name") },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !uiState.isLoading,
                    isError = uiState.displayNameError != null,
                    supportingText = {
                        uiState.displayNameError?.let { error ->
                            Text(
                                text = error,
                                color = MaterialTheme.colorScheme.error
                            )
                        }
                    }
                )

                OutlinedTextField(
                    value = uiState.bio,
                    onValueChange = viewModel::updateBio,
                    label = { Text("Bio") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(150.dp),
                    enabled = !uiState.isLoading,
                    maxLines = 5,
                    isError = uiState.bioError != null,
                    supportingText = {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            uiState.bioError?.let { error ->
                                Text(
                                    text = error,
                                    color = MaterialTheme.colorScheme.error
                                )
                            } ?: Spacer(modifier = Modifier.weight(1f))

                            Text(
                                text = "${uiState.bio.length}/500",
                                color = if (uiState.bio.length > 500) {
                                    MaterialTheme.colorScheme.error
                                } else {
                                    MaterialTheme.colorScheme.onSurfaceVariant
                                }
                            )
                        }
                    }
                )

                uiState.error?.let { error ->
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.errorContainer
                        )
                    ) {
                        Text(
                            text = error,
                            modifier = Modifier.padding(16.dp),
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                    }
                }
            }

            if (uiState.isLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = androidx.compose.ui.Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            }
        }
    }
}
```

---

## Flutter Cross-Platform Implementation

### Project Setup

```bash
flutter create my_app
cd my_app

flutter pub add \
  riverpod \
  flutter_riverpod \
  riverpod_annotation \
  dio \
  freezed_annotation \
  json_annotation \
  go_router

flutter pub add --dev \
  build_runner \
  riverpod_generator \
  freezed \
  json_serializable
```

**Project Structure:**

```
lib/
├── main.dart
├── app.dart
├── core/
│   ├── constants/
│   ├── theme/
│   └── utils/
├── data/
│   ├── models/
│   ├── repositories/
│   └── services/
├── presentation/
│   ├── screens/
│   ├── widgets/
│   └── providers/
└── router/
```

### Models with Freezed

```dart
// lib/data/models/user_profile.dart
import 'package:freezed_annotation/freezed_annotation.dart';

part 'user_profile.freezed.dart';
part 'user_profile.g.dart';

@freezed
class UserProfile with _$UserProfile {
  const factory UserProfile({
    required String id,
    required String username,
    required String email,
    String? displayName,
    String? bio,
    String? avatarUrl,
    required UserPreferences preferences,
    required UserStats stats,
    @JsonKey(name: 'created_at') required DateTime createdAt,
    @JsonKey(name: 'updated_at') required DateTime updatedAt,
  }) = _UserProfile;

  factory UserProfile.fromJson(Map<String, dynamic> json) =>
      _$UserProfileFromJson(json);
}

@freezed
class UserPreferences with _$UserPreferences {
  const factory UserPreferences({
    @Default(Theme.auto) Theme theme,
    @Default(NotificationSettings()) NotificationSettings notifications,
    @Default(PrivacySettings()) PrivacySettings privacy,
  }) = _UserPreferences;

  factory UserPreferences.fromJson(Map<String, dynamic> json) =>
      _$UserPreferencesFromJson(json);
}

enum Theme {
  @JsonValue('light') light,
  @JsonValue('dark') dark,
  @JsonValue('auto') auto,
}
```

### Riverpod State Management

```dart
// lib/presentation/providers/profile_provider.dart
import 'package:riverpod_annotation/riverpod_annotation.dart';
import '../../data/models/user_profile.dart';
import '../../data/repositories/user_repository.dart';

part 'profile_provider.g.dart';

@riverpod
class Profile extends _$Profile {
  @override
  Future<UserProfile> build(String userId) async {
    final repository = ref.watch(userRepositoryProvider);
    return repository.getProfile(userId);
  }

  Future<void> updateProfile({
    String? displayName,
    String? bio,
  }) async {
    state = const AsyncValue.loading();

    state = await AsyncValue.guard(() async {
      final repository = ref.read(userRepositoryProvider);
      return repository.updateProfile(
        userId: state.value!.id,
        displayName: displayName,
        bio: bio,
      );
    });
  }
}
```

### Flutter UI

```dart
// lib/presentation/screens/profile_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/profile_provider.dart';

class ProfileScreen extends ConsumerStatefulWidget {
  final String userId;

  const ProfileScreen({required this.userId, super.key});

  @override
  ConsumerState<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends ConsumerState<ProfileScreen> {
  late TextEditingController _displayNameController;
  late TextEditingController _bioController;

  @override
  void initState() {
    super.initState();
    _displayNameController = TextEditingController();
    _bioController = TextEditingController();
  }

  @override
  void dispose() {
    _displayNameController.dispose();
    _bioController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final profileAsync = ref.watch(profileProvider(widget.userId));

    return Scaffold(
      appBar: AppBar(
        title: const Text('Edit Profile'),
        actions: [
          TextButton(
            onPressed: profileAsync.isLoading
                ? null
                : () => _saveProfile(),
            child: const Text('Save'),
          ),
        ],
      ),
      body: profileAsync.when(
        data: (profile) {
          if (_displayNameController.text.isEmpty) {
            _displayNameController.text = profile.displayName ?? '';
            _bioController.text = profile.bio ?? '';
          }

          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                TextField(
                  controller: _displayNameController,
                  decoration: const InputDecoration(
                    labelText: 'Display Name',
                    border: OutlineInputBorder(),
                  ),
                  maxLength: 100,
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _bioController,
                  decoration: InputDecoration(
                    labelText: 'Bio',
                    border: const OutlineInputBorder(),
                    helperText: '${_bioController.text.length}/500',
                  ),
                  maxLength: 500,
                  maxLines: 5,
                ),
              ],
            ),
          );
        },
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, stack) => Center(child: Text('Error: $error')),
      ),
    );
  }

  Future<void> _saveProfile() async {
    await ref.read(profileProvider(widget.userId).notifier).updateProfile(
          displayName: _displayNameController.text,
          bio: _bioController.text,
        );
  }
}
```

---

## Electron/Tauri Desktop Implementation

### Tauri Setup

```bash
npm create tauri-app@latest

# Choose template: React with TypeScript
# Then install additional dependencies
npm install \
  @tanstack/react-query \
  zustand \
  @tauri-apps/api
```

**Tauri Configuration:**

```json
// src-tauri/tauri.conf.json
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:1420",
    "distDir": "../dist"
  },
  "package": {
    "productName": "MyApp",
    "version": "1.0.0"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "open": true
      },
      "fs": {
        "all": false,
        "readFile": true,
        "writeFile": true,
        "scope": ["$APPDATA/*", "$APPDATA/**"]
      },
      "notification": {
        "all": true
      }
    },
    "windows": [
      {
        "title": "MyApp",
        "width": 1200,
        "height": 800,
        "minWidth": 800,
        "minHeight": 600,
        "resizable": true,
        "fullscreen": false
      }
    ]
  }
}
```

### Native Integration

```typescript
// src/lib/tauri/fs.ts
import { invoke } from '@tauri-apps/api/tauri';
import { appDataDir } from '@tauri-apps/api/path';
import { writeTextFile, readTextFile } from '@tauri-apps/api/fs';

export async function saveUserData(userId: string, data: any): Promise<void> {
  const appData = await appDataDir();
  const filePath = `${appData}/users/${userId}.json`;
  await writeTextFile(filePath, JSON.stringify(data));
}

export async function loadUserData(userId: string): Promise<any | null> {
  try {
    const appData = await appDataDir();
    const filePath = `${appData}/users/${userId}.json`;
    const contents = await readTextFile(filePath);
    return JSON.parse(contents);
  } catch {
    return null;
  }
}

// System notifications
import { isPermissionGranted, requestPermission, sendNotification } from '@tauri-apps/api/notification';

export async function showNotification(title: string, body: string): Promise<void> {
  let permissionGranted = await isPermissionGranted();
  if (!permissionGranted) {
    const permission = await requestPermission();
    permissionGranted = permission === 'granted';
  }
  if (permissionGranted) {
    sendNotification({ title, body });
  }
}
```

---

This implementation guide provides comprehensive, production-ready patterns for building cross-platform applications with modern frameworks and best practices.

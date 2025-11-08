# Platform Architecture Guide

Comprehensive architectural patterns for building scalable multi-platform applications with API-first design, shared business logic, and platform abstraction layers.

## Table of Contents

1. [API-First Architecture](#api-first-architecture)
2. [Shared Business Logic Strategies](#shared-business-logic-strategies)
3. [Platform Abstraction Layers](#platform-abstraction-layers)
4. [Offline-First Architecture](#offline-first-architecture)
5. [Code Examples](#code-examples)

---

## API-First Architecture

### Overview

API-First architecture ensures all platforms consume the same contracts, enabling parallel development, consistent behavior, and easier testing. By defining APIs before implementation, teams can work independently while maintaining integration confidence.

### OpenAPI 3.1 Specification

**Complete API Contract Example:**

```yaml
openapi: 3.1.0
info:
  title: Multi-Platform Feature API
  version: 1.0.0
  description: Backend API for cross-platform feature implementation

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://staging-api.example.com/v1
    description: Staging

paths:
  /users/{userId}/profile:
    get:
      summary: Get user profile
      operationId: getUserProfile
      tags:
        - users
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
        - name: include
          in: query
          description: Related resources to include
          schema:
            type: array
            items:
              type: string
              enum: [preferences, stats, achievements]
      responses:
        '200':
          description: User profile retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserProfile'
        '404':
          $ref: '#/components/responses/NotFound'
        '401':
          $ref: '#/components/responses/Unauthorized'

    patch:
      summary: Update user profile
      operationId: updateUserProfile
      tags:
        - users
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserProfileUpdate'
      responses:
        '200':
          description: Profile updated successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserProfile'
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          $ref: '#/components/responses/Conflict'

  /feed:
    get:
      summary: Get personalized content feed
      operationId: getContentFeed
      tags:
        - content
      parameters:
        - name: cursor
          in: query
          description: Pagination cursor
          schema:
            type: string
        - name: limit
          in: query
          description: Number of items to return
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
        - name: filter
          in: query
          description: Content filter
          schema:
            type: string
            enum: [all, following, trending]
            default: all
      responses:
        '200':
          description: Feed retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/ContentItem'
                  pagination:
                    $ref: '#/components/schemas/PaginationMetadata'

components:
  schemas:
    UserProfile:
      type: object
      required:
        - id
        - username
        - email
        - createdAt
      properties:
        id:
          type: string
          format: uuid
        username:
          type: string
          minLength: 3
          maxLength: 30
          pattern: '^[a-zA-Z0-9_]+$'
        email:
          type: string
          format: email
        displayName:
          type: string
          maxLength: 100
        avatarUrl:
          type: string
          format: uri
          nullable: true
        bio:
          type: string
          maxLength: 500
          nullable: true
        preferences:
          $ref: '#/components/schemas/UserPreferences'
        stats:
          $ref: '#/components/schemas/UserStats'
        createdAt:
          type: string
          format: date-time
        updatedAt:
          type: string
          format: date-time

    UserProfileUpdate:
      type: object
      properties:
        displayName:
          type: string
          maxLength: 100
        bio:
          type: string
          maxLength: 500
        preferences:
          $ref: '#/components/schemas/UserPreferences'

    UserPreferences:
      type: object
      properties:
        theme:
          type: string
          enum: [light, dark, auto]
          default: auto
        notifications:
          type: object
          properties:
            email:
              type: boolean
              default: true
            push:
              type: boolean
              default: true
            inApp:
              type: boolean
              default: true
        privacy:
          type: object
          properties:
            profileVisibility:
              type: string
              enum: [public, friends, private]
              default: public
            showEmail:
              type: boolean
              default: false

    UserStats:
      type: object
      properties:
        followersCount:
          type: integer
          minimum: 0
        followingCount:
          type: integer
          minimum: 0
        postsCount:
          type: integer
          minimum: 0
        engagementRate:
          type: number
          format: float
          minimum: 0
          maximum: 100

    ContentItem:
      type: object
      required:
        - id
        - type
        - author
        - createdAt
      properties:
        id:
          type: string
          format: uuid
        type:
          type: string
          enum: [post, article, video, poll]
        author:
          $ref: '#/components/schemas/UserProfile'
        title:
          type: string
          maxLength: 200
        content:
          type: string
        mediaUrls:
          type: array
          items:
            type: string
            format: uri
        tags:
          type: array
          items:
            type: string
        engagementMetrics:
          type: object
          properties:
            likes:
              type: integer
              minimum: 0
            comments:
              type: integer
              minimum: 0
            shares:
              type: integer
              minimum: 0
        createdAt:
          type: string
          format: date-time

    PaginationMetadata:
      type: object
      properties:
        nextCursor:
          type: string
          nullable: true
        hasMore:
          type: boolean
        totalCount:
          type: integer
          minimum: 0

    Error:
      type: object
      required:
        - code
        - message
      properties:
        code:
          type: string
        message:
          type: string
        details:
          type: object
          additionalProperties: true
        traceId:
          type: string
          format: uuid

  responses:
    BadRequest:
      description: Invalid request parameters
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: INVALID_INPUT
            message: Invalid email format
            details:
              field: email
              value: invalid-email

    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: UNAUTHORIZED
            message: Valid authentication token required

    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: NOT_FOUND
            message: User profile not found

    Conflict:
      description: Resource conflict
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: CONFLICT
            message: Username already taken

  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    OAuth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: https://api.example.com/oauth/authorize
          tokenUrl: https://api.example.com/oauth/token
          scopes:
            read:profile: Read user profile
            write:profile: Update user profile
            read:feed: Access content feed

security:
  - BearerAuth: []
  - OAuth2:
      - read:profile
```

### GraphQL Schema Design

**For Complex Data Queries:**

```graphql
type Query {
  """
  Get user profile by ID or username
  """
  user(id: ID, username: String): User

  """
  Get personalized content feed with filtering and pagination
  """
  feed(
    first: Int = 20
    after: String
    filter: FeedFilter = ALL
  ): FeedConnection!

  """
  Search content across multiple types
  """
  search(
    query: String!
    type: [ContentType!]
    limit: Int = 20
  ): SearchResults!
}

type Mutation {
  """
  Update user profile information
  """
  updateProfile(input: UpdateProfileInput!): UserProfilePayload!

  """
  Create new content item
  """
  createContent(input: CreateContentInput!): ContentPayload!

  """
  Like or unlike content
  """
  toggleLike(contentId: ID!): LikePayload!
}

type Subscription {
  """
  Subscribe to real-time feed updates
  """
  feedUpdated(userId: ID!): ContentItem!

  """
  Subscribe to notification events
  """
  notificationReceived(userId: ID!): Notification!
}

type User {
  id: ID!
  username: String!
  email: String!
  displayName: String
  avatarUrl: String
  bio: String
  preferences: UserPreferences!
  stats: UserStats!
  posts(first: Int, after: String): PostConnection!
  followers(first: Int, after: String): UserConnection!
  following(first: Int, after: String): UserConnection!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type UserPreferences {
  theme: Theme!
  notifications: NotificationSettings!
  privacy: PrivacySettings!
}

enum Theme {
  LIGHT
  DARK
  AUTO
}

type NotificationSettings {
  email: Boolean!
  push: Boolean!
  inApp: Boolean!
}

type PrivacySettings {
  profileVisibility: Visibility!
  showEmail: Boolean!
}

enum Visibility {
  PUBLIC
  FRIENDS
  PRIVATE
}

type UserStats {
  followersCount: Int!
  followingCount: Int!
  postsCount: Int!
  engagementRate: Float!
}

interface ContentItem {
  id: ID!
  type: ContentType!
  author: User!
  createdAt: DateTime!
  engagementMetrics: EngagementMetrics!
}

enum ContentType {
  POST
  ARTICLE
  VIDEO
  POLL
}

type Post implements ContentItem {
  id: ID!
  type: ContentType!
  author: User!
  content: String!
  mediaUrls: [String!]!
  tags: [String!]!
  comments(first: Int, after: String): CommentConnection!
  createdAt: DateTime!
  engagementMetrics: EngagementMetrics!
}

type EngagementMetrics {
  likes: Int!
  comments: Int!
  shares: Int!
  isLikedByViewer: Boolean!
}

type FeedConnection {
  edges: [FeedEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type FeedEdge {
  node: ContentItem!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  endCursor: String
}

enum FeedFilter {
  ALL
  FOLLOWING
  TRENDING
}

input UpdateProfileInput {
  displayName: String
  bio: String
  preferences: UserPreferencesInput
}

input UserPreferencesInput {
  theme: Theme
  notifications: NotificationSettingsInput
  privacy: PrivacySettingsInput
}

input NotificationSettingsInput {
  email: Boolean
  push: Boolean
  inApp: Boolean
}

input PrivacySettingsInput {
  profileVisibility: Visibility
  showEmail: Boolean
}

type UserProfilePayload {
  user: User!
  errors: [Error!]
}

type Error {
  field: String
  message: String!
  code: String!
}

scalar DateTime
```

### WebSocket Real-Time Events

**For Live Updates:**

```typescript
// WebSocket Event Contract
interface WebSocketMessage {
  type: 'feed.updated' | 'notification.received' | 'presence.changed' | 'ping' | 'pong';
  payload: unknown;
  timestamp: string;
  eventId: string;
}

// Feed Update Event
interface FeedUpdatedEvent extends WebSocketMessage {
  type: 'feed.updated';
  payload: {
    contentItem: ContentItem;
    action: 'created' | 'updated' | 'deleted';
  };
}

// Notification Event
interface NotificationReceivedEvent extends WebSocketMessage {
  type: 'notification.received';
  payload: {
    id: string;
    type: 'like' | 'comment' | 'follow' | 'mention';
    actor: UserProfile;
    contentId?: string;
    message: string;
    createdAt: string;
  };
}

// Presence Event
interface PresenceChangedEvent extends WebSocketMessage {
  type: 'presence.changed';
  payload: {
    userId: string;
    status: 'online' | 'away' | 'offline';
    lastSeen: string;
  };
}

// Connection Protocol
const wsProtocol = {
  // Client -> Server: Authentication
  authenticate: {
    type: 'auth',
    token: 'jwt-token'
  },

  // Server -> Client: Authentication Success
  authSuccess: {
    type: 'auth.success',
    userId: 'user-id',
    sessionId: 'session-id'
  },

  // Client -> Server: Subscribe to channels
  subscribe: {
    type: 'subscribe',
    channels: ['feed', 'notifications', 'presence']
  },

  // Server -> Client: Subscription Confirmed
  subscribed: {
    type: 'subscribed',
    channels: ['feed', 'notifications', 'presence']
  },

  // Keep-alive
  ping: { type: 'ping' },
  pong: { type: 'pong' }
};
```

### Backend-for-Frontend (BFF) Pattern

**Platform-Specific API Gateways:**

```typescript
// Web BFF - Optimized for browser clients
interface WebBFFEndpoints {
  // Server-side rendering data
  '/api/bff/web/ssr/profile/:userId': {
    response: {
      profile: UserProfile;
      initialFeed: ContentItem[];
      metadata: {
        seoTitle: string;
        seoDescription: string;
        ogImage: string;
      };
    };
  };

  // Aggregated dashboard data
  '/api/bff/web/dashboard': {
    response: {
      user: UserProfile;
      feedItems: ContentItem[];
      notifications: Notification[];
      trendingTopics: Topic[];
      recommendations: ContentItem[];
    };
  };
}

// Mobile BFF - Optimized for bandwidth and battery
interface MobileBFFEndpoints {
  // Minimal data for list views
  '/api/bff/mobile/feed/compact': {
    response: {
      items: Array<{
        id: string;
        type: string;
        authorId: string;
        authorName: string;
        authorAvatar: string; // Thumbnail URL
        preview: string; // First 200 chars
        mediaCount: number;
        engagement: {
          likes: number;
          comments: number;
        };
        timestamp: string;
      }>;
      nextCursor: string;
    };
  };

  // Batch requests to reduce round trips
  '/api/bff/mobile/batch': {
    request: {
      requests: Array<{
        id: string;
        endpoint: string;
        params: Record<string, unknown>;
      }>;
    };
    response: {
      responses: Array<{
        id: string;
        status: number;
        data: unknown;
      }>;
    };
  };
}

// Desktop BFF - Rich data for larger screens
interface DesktopBFFEndpoints {
  // Multi-panel layout data
  '/api/bff/desktop/workspace': {
    response: {
      leftPanel: {
        navigation: NavigationItem[];
        quickActions: Action[];
      };
      centerPanel: {
        feed: ContentItem[];
        filters: FilterOptions[];
      };
      rightPanel: {
        trending: TrendingItem[];
        suggestions: Suggestion[];
        analytics: AnalyticsWidget[];
      };
    };
  };
}
```

### API Versioning Strategy

**Multiple Versioning Approaches:**

1. **URL Versioning** (Recommended for major versions):
```
https://api.example.com/v1/users
https://api.example.com/v2/users
```

2. **Header Versioning** (For minor versions):
```http
GET /users HTTP/1.1
Accept: application/vnd.example.v2+json
API-Version: 2.1
```

3. **Deprecation Headers**:
```http
HTTP/1.1 200 OK
Deprecation: true
Sunset: Sat, 31 Dec 2025 23:59:59 GMT
Link: </v2/users>; rel="successor-version"
```

4. **Version Compatibility Matrix**:
```yaml
api_versions:
  v1:
    status: deprecated
    sunset_date: 2025-12-31
    supported_clients:
      - ios: "<=1.0.2"
      - android: "<=1.0.2"
      - web: "<=1.5.0"

  v2:
    status: stable
    release_date: 2025-01-15
    supported_clients:
      - ios: ">=2.1.0"
      - android: ">=2.1.0"
      - web: ">=1.6.0"

  v3:
    status: beta
    release_date: 2025-11-01
    breaking_changes:
      - Removed deprecated /users/search endpoint
      - Changed authentication to OAuth 2.1
      - New pagination using cursor-based approach
```

### Rate Limiting and Caching

**API Rate Limit Headers:**

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1699564800
X-RateLimit-Resource: api
X-RateLimit-Used: 13
Retry-After: 3600
```

**Caching Strategy:**

```http
# Immutable Content
HTTP/1.1 200 OK
Cache-Control: public, max-age=31536000, immutable
ETag: "a7b3c9d8e5f2"

# User-Specific Data
HTTP/1.1 200 OK
Cache-Control: private, max-age=300
Vary: Authorization

# Real-Time Data
HTTP/1.1 200 OK
Cache-Control: no-cache, no-store, must-revalidate
Pragma: no-cache
Expires: 0
```

**CDN Cache Tags for Invalidation:**

```http
Cache-Tag: user:123, feed:homepage, content:456
```

```typescript
// Platform-specific cache implementation
interface CacheStrategy {
  web: {
    // Browser cache + service worker
    staticAssets: 'max-age=31536000, immutable';
    apiResponses: 'max-age=300, stale-while-revalidate=60';
    userContent: 'no-cache, must-revalidate';
  };

  ios: {
    // URLCache + custom disk cache
    images: '7 days';
    feed: '5 minutes with background refresh';
    profile: '30 minutes';
  };

  android: {
    // OkHttp cache + Room database
    images: '7 days, max 100MB';
    feed: '5 minutes with background refresh';
    profile: '30 minutes';
  };
}
```

---

## Shared Business Logic Strategies

### Kotlin Multiplatform Mobile (KMM)

**Shared Domain Models:**

```kotlin
// commonMain/kotlin/domain/models/User.kt
package com.example.domain.models

import kotlinx.serialization.Serializable
import kotlinx.datetime.Instant

@Serializable
data class UserProfile(
    val id: String,
    val username: String,
    val email: String,
    val displayName: String? = null,
    val avatarUrl: String? = null,
    val bio: String? = null,
    val preferences: UserPreferences,
    val stats: UserStats,
    val createdAt: Instant,
    val updatedAt: Instant
)

@Serializable
data class UserPreferences(
    val theme: Theme = Theme.AUTO,
    val notifications: NotificationSettings = NotificationSettings(),
    val privacy: PrivacySettings = PrivacySettings()
)

enum class Theme {
    LIGHT, DARK, AUTO
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

enum class Visibility {
    PUBLIC, FRIENDS, PRIVATE
}

@Serializable
data class UserStats(
    val followersCount: Int,
    val followingCount: Int,
    val postsCount: Int,
    val engagementRate: Float
)
```

**Shared Business Logic:**

```kotlin
// commonMain/kotlin/domain/usecases/UpdateProfileUseCase.kt
package com.example.domain.usecases

import com.example.domain.models.UserProfile
import com.example.domain.repositories.UserRepository
import com.example.domain.validators.ProfileValidator

class UpdateProfileUseCase(
    private val repository: UserRepository,
    private val validator: ProfileValidator
) {
    suspend operator fun invoke(
        userId: String,
        displayName: String?,
        bio: String?
    ): Result<UserProfile> {
        // Validation logic shared across platforms
        displayName?.let {
            validator.validateDisplayName(it).onFailure { error ->
                return Result.failure(error)
            }
        }

        bio?.let {
            validator.validateBio(it).onFailure { error ->
                return Result.failure(error)
            }
        }

        // Business rule: Check profanity
        val cleanDisplayName = displayName?.let {
            validator.checkProfanity(it).getOrNull()
        }

        val cleanBio = bio?.let {
            validator.checkProfanity(it).getOrNull()
        }

        // Update via repository
        return repository.updateProfile(
            userId = userId,
            displayName = cleanDisplayName,
            bio = cleanBio
        )
    }
}
```

**Platform-Specific Repository Interfaces:**

```kotlin
// commonMain/kotlin/domain/repositories/UserRepository.kt
package com.example.domain.repositories

import com.example.domain.models.UserProfile

interface UserRepository {
    suspend fun getProfile(userId: String): Result<UserProfile>
    suspend fun updateProfile(
        userId: String,
        displayName: String?,
        bio: String?
    ): Result<UserProfile>
    suspend fun cacheProfile(profile: UserProfile)
    suspend fun getCachedProfile(userId: String): UserProfile?
}
```

**iOS Implementation:**

```kotlin
// iosMain/kotlin/data/repositories/UserRepositoryImpl.kt
package com.example.data.repositories

import com.example.domain.models.UserProfile
import com.example.domain.repositories.UserRepository
import platform.Foundation.NSUserDefaults

actual class UserRepositoryImpl(
    private val api: ApiClient,
    private val userDefaults: NSUserDefaults
) : UserRepository {
    override suspend fun getProfile(userId: String): Result<UserProfile> {
        // Try cache first
        getCachedProfile(userId)?.let {
            return Result.success(it)
        }

        // Fetch from API
        return try {
            val profile = api.fetchProfile(userId)
            cacheProfile(profile)
            Result.success(profile)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun cacheProfile(profile: UserProfile) {
        // iOS-specific: Use NSUserDefaults for caching
        val jsonString = Json.encodeToString(UserProfile.serializer(), profile)
        userDefaults.setObject(jsonString, forKey = "profile_${profile.id}")
    }

    override suspend fun getCachedProfile(userId: String): UserProfile? {
        val jsonString = userDefaults.stringForKey("profile_$userId") ?: return null
        return try {
            Json.decodeFromString(UserProfile.serializer(), jsonString)
        } catch (e: Exception) {
            null
        }
    }
}
```

**Android Implementation:**

```kotlin
// androidMain/kotlin/data/repositories/UserRepositoryImpl.kt
package com.example.data.repositories

import android.content.Context
import android.content.SharedPreferences
import com.example.domain.models.UserProfile
import com.example.domain.repositories.UserRepository

actual class UserRepositoryImpl(
    private val api: ApiClient,
    private val context: Context
) : UserRepository {
    private val prefs: SharedPreferences =
        context.getSharedPreferences("user_cache", Context.MODE_PRIVATE)

    override suspend fun getProfile(userId: String): Result<UserProfile> {
        // Try cache first
        getCachedProfile(userId)?.let {
            return Result.success(it)
        }

        // Fetch from API
        return try {
            val profile = api.fetchProfile(userId)
            cacheProfile(profile)
            Result.success(profile)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun cacheProfile(profile: UserProfile) {
        // Android-specific: Use SharedPreferences
        val jsonString = Json.encodeToString(UserProfile.serializer(), profile)
        prefs.edit()
            .putString("profile_${profile.id}", jsonString)
            .apply()
    }

    override suspend fun getCachedProfile(userId: String): UserProfile? {
        val jsonString = prefs.getString("profile_$userId", null) ?: return null
        return try {
            Json.decodeFromString(UserProfile.serializer(), jsonString)
        } catch (e: Exception) {
            null
        }
    }
}
```

### TypeScript/JavaScript Sharing

**Shared Validation Logic (Node.js + Browser + React Native):**

```typescript
// shared/validators/profileValidator.ts
export interface ValidationResult {
  isValid: boolean;
  errors: Array<{
    field: string;
    message: string;
    code: string;
  }>;
}

export class ProfileValidator {
  private static readonly USERNAME_PATTERN = /^[a-zA-Z0-9_]{3,30}$/;
  private static readonly EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  static validateDisplayName(displayName: string): ValidationResult {
    const errors: ValidationResult['errors'] = [];

    if (displayName.length === 0) {
      errors.push({
        field: 'displayName',
        message: 'Display name cannot be empty',
        code: 'DISPLAY_NAME_EMPTY'
      });
    }

    if (displayName.length > 100) {
      errors.push({
        field: 'displayName',
        message: 'Display name must be 100 characters or less',
        code: 'DISPLAY_NAME_TOO_LONG'
      });
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  static validateBio(bio: string): ValidationResult {
    const errors: ValidationResult['errors'] = [];

    if (bio.length > 500) {
      errors.push({
        field: 'bio',
        message: 'Bio must be 500 characters or less',
        code: 'BIO_TOO_LONG'
      });
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  static validateEmail(email: string): ValidationResult {
    const errors: ValidationResult['errors'] = [];

    if (!this.EMAIL_PATTERN.test(email)) {
      errors.push({
        field: 'email',
        message: 'Invalid email format',
        code: 'EMAIL_INVALID'
      });
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  static validateUsername(username: string): ValidationResult {
    const errors: ValidationResult['errors'] = [];

    if (!this.USERNAME_PATTERN.test(username)) {
      errors.push({
        field: 'username',
        message: 'Username must be 3-30 characters and contain only letters, numbers, and underscores',
        code: 'USERNAME_INVALID'
      });
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }
}
```

**Shared State Machines (XState):**

```typescript
// shared/machines/authMachine.ts
import { createMachine, assign } from 'xstate';

export interface AuthContext {
  user: UserProfile | null;
  accessToken: string | null;
  refreshToken: string | null;
  error: string | null;
}

export type AuthEvent =
  | { type: 'LOGIN'; email: string; password: string }
  | { type: 'LOGIN_SUCCESS'; user: UserProfile; tokens: { access: string; refresh: string } }
  | { type: 'LOGIN_FAILURE'; error: string }
  | { type: 'LOGOUT' }
  | { type: 'REFRESH_TOKEN' }
  | { type: 'REFRESH_SUCCESS'; accessToken: string }
  | { type: 'REFRESH_FAILURE'; error: string };

export const authMachine = createMachine<AuthContext, AuthEvent>({
  id: 'auth',
  initial: 'unauthenticated',
  context: {
    user: null,
    accessToken: null,
    refreshToken: null,
    error: null
  },
  states: {
    unauthenticated: {
      on: {
        LOGIN: {
          target: 'loggingIn'
        }
      }
    },
    loggingIn: {
      on: {
        LOGIN_SUCCESS: {
          target: 'authenticated',
          actions: assign({
            user: (_, event) => event.user,
            accessToken: (_, event) => event.tokens.access,
            refreshToken: (_, event) => event.tokens.refresh,
            error: null
          })
        },
        LOGIN_FAILURE: {
          target: 'unauthenticated',
          actions: assign({
            error: (_, event) => event.error
          })
        }
      }
    },
    authenticated: {
      on: {
        LOGOUT: {
          target: 'unauthenticated',
          actions: assign({
            user: null,
            accessToken: null,
            refreshToken: null,
            error: null
          })
        },
        REFRESH_TOKEN: {
          target: 'refreshing'
        }
      }
    },
    refreshing: {
      on: {
        REFRESH_SUCCESS: {
          target: 'authenticated',
          actions: assign({
            accessToken: (_, event) => event.accessToken,
            error: null
          })
        },
        REFRESH_FAILURE: {
          target: 'unauthenticated',
          actions: assign({
            user: null,
            accessToken: null,
            refreshToken: null,
            error: (_, event) => event.error
          })
        }
      }
    }
  }
});
```

**Usage in React (Web):**

```typescript
// web/hooks/useAuth.ts
import { useMachine } from '@xstate/react';
import { authMachine } from '@shared/machines/authMachine';
import { useCallback } from 'react';

export function useAuth() {
  const [state, send] = useMachine(authMachine);

  const login = useCallback(async (email: string, password: string) => {
    send({ type: 'LOGIN', email, password });

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();
      send({
        type: 'LOGIN_SUCCESS',
        user: data.user,
        tokens: data.tokens
      });
    } catch (error) {
      send({
        type: 'LOGIN_FAILURE',
        error: error.message
      });
    }
  }, [send]);

  return {
    user: state.context.user,
    isAuthenticated: state.matches('authenticated'),
    isLoading: state.matches('loggingIn') || state.matches('refreshing'),
    error: state.context.error,
    login,
    logout: () => send({ type: 'LOGOUT' })
  };
}
```

**Usage in React Native (Mobile):**

```typescript
// mobile/hooks/useAuth.ts
import { useMachine } from '@xstate/react';
import { authMachine } from '@shared/machines/authMachine';
import { useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export function useAuth() {
  const [state, send] = useMachine(authMachine);

  const login = useCallback(async (email: string, password: string) => {
    send({ type: 'LOGIN', email, password });

    try {
      const response = await fetch('https://api.example.com/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();

      // Persist tokens in AsyncStorage
      await AsyncStorage.multiSet([
        ['accessToken', data.tokens.access],
        ['refreshToken', data.tokens.refresh]
      ]);

      send({
        type: 'LOGIN_SUCCESS',
        user: data.user,
        tokens: data.tokens
      });
    } catch (error) {
      send({
        type: 'LOGIN_FAILURE',
        error: error.message
      });
    }
  }, [send]);

  return {
    user: state.context.user,
    isAuthenticated: state.matches('authenticated'),
    isLoading: state.matches('loggingIn') || state.matches('refreshing'),
    error: state.context.error,
    login,
    logout: () => send({ type: 'LOGOUT' })
  };
}
```

### Domain-Driven Design (DDD) for Shared Logic

**Value Objects (Platform-Agnostic):**

```typescript
// shared/domain/valueObjects/Email.ts
export class Email {
  private readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }

  static create(email: string): Result<Email, string> {
    if (!email || email.trim().length === 0) {
      return Result.error('Email cannot be empty');
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return Result.error('Invalid email format');
    }

    return Result.ok(new Email(email.toLowerCase()));
  }

  getValue(): string {
    return this.value;
  }

  equals(other: Email): boolean {
    return this.value === other.value;
  }
}

// shared/domain/valueObjects/Username.ts
export class Username {
  private readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }

  static create(username: string): Result<Username, string> {
    if (!username || username.trim().length === 0) {
      return Result.error('Username cannot be empty');
    }

    if (username.length < 3 || username.length > 30) {
      return Result.error('Username must be between 3 and 30 characters');
    }

    const usernameRegex = /^[a-zA-Z0-9_]+$/;
    if (!usernameRegex.test(username)) {
      return Result.error('Username can only contain letters, numbers, and underscores');
    }

    return Result.ok(new Username(username));
  }

  getValue(): string {
    return this.value;
  }

  equals(other: Username): boolean {
    return this.value === other.value;
  }
}
```

**Entities:**

```typescript
// shared/domain/entities/User.ts
import { Email } from '../valueObjects/Email';
import { Username } from '../valueObjects/Username';

export class User {
  private constructor(
    public readonly id: string,
    public readonly username: Username,
    public readonly email: Email,
    private displayName: string | null,
    private bio: string | null,
    private avatarUrl: string | null
  ) {}

  static create(props: {
    id: string;
    username: string;
    email: string;
    displayName?: string;
    bio?: string;
    avatarUrl?: string;
  }): Result<User, string> {
    const usernameOrError = Username.create(props.username);
    if (usernameOrError.isError()) {
      return Result.error(usernameOrError.error);
    }

    const emailOrError = Email.create(props.email);
    if (emailOrError.isError()) {
      return Result.error(emailOrError.error);
    }

    return Result.ok(new User(
      props.id,
      usernameOrError.value,
      emailOrError.value,
      props.displayName || null,
      props.bio || null,
      props.avatarUrl || null
    ));
  }

  updateDisplayName(displayName: string): Result<void, string> {
    if (displayName.length > 100) {
      return Result.error('Display name must be 100 characters or less');
    }

    this.displayName = displayName;
    return Result.ok(undefined);
  }

  updateBio(bio: string): Result<void, string> {
    if (bio.length > 500) {
      return Result.error('Bio must be 500 characters or less');
    }

    this.bio = bio;
    return Result.ok(undefined);
  }

  getDisplayName(): string {
    return this.displayName || this.username.getValue();
  }

  getBio(): string | null {
    return this.bio;
  }

  getAvatarUrl(): string | null {
    return this.avatarUrl;
  }
}
```

**Result Type (Railway-Oriented Programming):**

```typescript
// shared/utils/Result.ts
export class Result<T, E> {
  private constructor(
    private readonly success: boolean,
    private readonly value?: T,
    private readonly errorValue?: E
  ) {}

  static ok<T, E>(value: T): Result<T, E> {
    return new Result(true, value);
  }

  static error<T, E>(error: E): Result<T, E> {
    return new Result(false, undefined, error);
  }

  isOk(): this is { value: T } {
    return this.success;
  }

  isError(): this is { error: E } {
    return !this.success;
  }

  unwrap(): T {
    if (!this.success) {
      throw new Error('Cannot unwrap error result');
    }
    return this.value!;
  }

  unwrapOr(defaultValue: T): T {
    return this.success ? this.value! : defaultValue;
  }

  map<U>(fn: (value: T) => U): Result<U, E> {
    if (this.success) {
      return Result.ok(fn(this.value!));
    }
    return Result.error(this.errorValue!);
  }

  mapError<F>(fn: (error: E) => F): Result<T, F> {
    if (!this.success) {
      return Result.error(fn(this.errorValue!));
    }
    return Result.ok(this.value!);
  }

  andThen<U>(fn: (value: T) => Result<U, E>): Result<U, E> {
    if (this.success) {
      return fn(this.value!);
    }
    return Result.error(this.errorValue!);
  }
}
```

---

## Platform Abstraction Layers

### Repository Pattern

**Abstract Repository Interface:**

```typescript
// shared/repositories/IUserRepository.ts
export interface IUserRepository {
  getById(id: string): Promise<Result<User, RepositoryError>>;
  getByUsername(username: string): Promise<Result<User, RepositoryError>>;
  update(user: User): Promise<Result<User, RepositoryError>>;
  delete(id: string): Promise<Result<void, RepositoryError>>;

  // Caching methods
  cacheUser(user: User): Promise<void>;
  getCachedUser(id: string): Promise<User | null>;
  invalidateCache(id: string): Promise<void>;
}

export type RepositoryError =
  | { type: 'NetworkError'; message: string }
  | { type: 'NotFound'; id: string }
  | { type: 'ValidationError'; errors: ValidationError[] }
  | { type: 'ServerError'; statusCode: number; message: string };
```

**Web Implementation (IndexedDB + Fetch):**

```typescript
// web/repositories/UserRepository.ts
import { IUserRepository, RepositoryError } from '@shared/repositories/IUserRepository';
import { User } from '@shared/domain/entities/User';
import { Result } from '@shared/utils/Result';

export class WebUserRepository implements IUserRepository {
  private dbName = 'app-cache';
  private storeName = 'users';

  async getById(id: string): Promise<Result<User, RepositoryError>> {
    // Try cache first
    const cached = await this.getCachedUser(id);
    if (cached) {
      return Result.ok(cached);
    }

    // Fetch from API
    try {
      const response = await fetch(`/api/users/${id}`);

      if (!response.ok) {
        if (response.status === 404) {
          return Result.error({ type: 'NotFound', id });
        }
        return Result.error({
          type: 'ServerError',
          statusCode: response.status,
          message: await response.text()
        });
      }

      const data = await response.json();
      const user = this.mapToEntity(data);

      await this.cacheUser(user);
      return Result.ok(user);
    } catch (error) {
      return Result.error({
        type: 'NetworkError',
        message: error.message
      });
    }
  }

  async cacheUser(user: User): Promise<void> {
    const db = await this.openDatabase();
    const transaction = db.transaction(this.storeName, 'readwrite');
    const store = transaction.objectStore(this.storeName);

    store.put({
      id: user.id,
      username: user.username.getValue(),
      email: user.email.getValue(),
      displayName: user.getDisplayName(),
      bio: user.getBio(),
      avatarUrl: user.getAvatarUrl(),
      cachedAt: Date.now()
    });

    return new Promise((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  }

  async getCachedUser(id: string): Promise<User | null> {
    const db = await this.openDatabase();
    const transaction = db.transaction(this.storeName, 'readonly');
    const store = transaction.objectStore(this.storeName);

    return new Promise((resolve, reject) => {
      const request = store.get(id);

      request.onsuccess = () => {
        const data = request.result;
        if (!data) {
          resolve(null);
          return;
        }

        // Check if cache is stale (older than 5 minutes)
        if (Date.now() - data.cachedAt > 5 * 60 * 1000) {
          resolve(null);
          return;
        }

        const user = User.create({
          id: data.id,
          username: data.username,
          email: data.email,
          displayName: data.displayName,
          bio: data.bio,
          avatarUrl: data.avatarUrl
        });

        resolve(user.isOk() ? user.unwrap() : null);
      };

      request.onerror = () => reject(request.error);
    });
  }

  private async openDatabase(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: 'id' });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  private mapToEntity(data: any): User {
    return User.create({
      id: data.id,
      username: data.username,
      email: data.email,
      displayName: data.displayName,
      bio: data.bio,
      avatarUrl: data.avatarUrl
    }).unwrap();
  }
}
```

**iOS Implementation (Core Data + URLSession):**

```swift
// iOS/Repositories/UserRepository.swift
import Foundation
import CoreData

class IOSUserRepository: IUserRepository {
    private let apiClient: APIClient
    private let context: NSManagedObjectContext

    init(apiClient: APIClient, context: NSManagedObjectContext) {
        self.apiClient = apiClient
        self.context = context
    }

    func getById(id: String) async -> Result<User, RepositoryError> {
        // Try cache first
        if let cached = await getCachedUser(id: id) {
            return .success(cached)
        }

        // Fetch from API
        do {
            let endpoint = "/users/\(id)"
            let data = try await apiClient.get(endpoint)
            let dto = try JSONDecoder().decode(UserDTO.self, from: data)
            let user = try mapToEntity(dto)

            await cacheUser(user)
            return .success(user)
        } catch {
            if let apiError = error as? APIError {
                switch apiError {
                case .notFound:
                    return .failure(.notFound(id: id))
                case .serverError(let code, let message):
                    return .failure(.serverError(statusCode: code, message: message))
                default:
                    return .failure(.networkError(message: error.localizedDescription))
                }
            }
            return .failure(.networkError(message: error.localizedDescription))
        }
    }

    func cacheUser(_ user: User) async {
        let fetchRequest: NSFetchRequest<UserEntity> = UserEntity.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "id == %@", user.id)

        do {
            let results = try context.fetch(fetchRequest)
            let entity = results.first ?? UserEntity(context: context)

            entity.id = user.id
            entity.username = user.username.value
            entity.email = user.email.value
            entity.displayName = user.displayName
            entity.bio = user.bio
            entity.avatarUrl = user.avatarUrl
            entity.cachedAt = Date()

            try context.save()
        } catch {
            print("Failed to cache user: \(error)")
        }
    }

    func getCachedUser(id: String) async -> User? {
        let fetchRequest: NSFetchRequest<UserEntity> = UserEntity.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "id == %@", id)
        fetchRequest.fetchLimit = 1

        do {
            let results = try context.fetch(fetchRequest)
            guard let entity = results.first else { return nil }

            // Check if cache is stale (older than 5 minutes)
            if let cachedAt = entity.cachedAt,
               Date().timeIntervalSince(cachedAt) > 5 * 60 {
                return nil
            }

            return try? User.create(
                id: entity.id ?? "",
                username: entity.username ?? "",
                email: entity.email ?? "",
                displayName: entity.displayName,
                bio: entity.bio,
                avatarUrl: entity.avatarUrl
            ).get()
        } catch {
            return nil
        }
    }
}
```

**Android Implementation (Room + Retrofit):**

```kotlin
// android/repositories/UserRepository.kt
package com.example.repositories

import com.example.data.local.UserDao
import com.example.data.local.UserEntity
import com.example.data.remote.ApiService
import com.example.domain.User
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Date

class AndroidUserRepository(
    private val apiService: ApiService,
    private val userDao: UserDao
) : IUserRepository {

    override suspend fun getById(id: String): Result<User> = withContext(Dispatchers.IO) {
        // Try cache first
        getCachedUser(id)?.let {
            return@withContext Result.success(it)
        }

        // Fetch from API
        try {
            val response = apiService.getUser(id)
            if (!response.isSuccessful) {
                return@withContext when (response.code()) {
                    404 -> Result.failure(RepositoryError.NotFound(id))
                    else -> Result.failure(
                        RepositoryError.ServerError(
                            response.code(),
                            response.message()
                        )
                    )
                }
            }

            val dto = response.body() ?: return@withContext Result.failure(
                RepositoryError.ServerError(500, "Empty response body")
            )

            val user = mapToEntity(dto)
            cacheUser(user)
            Result.success(user)
        } catch (e: Exception) {
            Result.failure(RepositoryError.NetworkError(e.message ?: "Unknown error"))
        }
    }

    override suspend fun cacheUser(user: User) = withContext(Dispatchers.IO) {
        val entity = UserEntity(
            id = user.id,
            username = user.username.value,
            email = user.email.value,
            displayName = user.displayName,
            bio = user.bio,
            avatarUrl = user.avatarUrl,
            cachedAt = Date()
        )
        userDao.insert(entity)
    }

    override suspend fun getCachedUser(id: String): User? = withContext(Dispatchers.IO) {
        val entity = userDao.getById(id) ?: return@withContext null

        // Check if cache is stale (older than 5 minutes)
        val now = Date()
        if (now.time - entity.cachedAt.time > 5 * 60 * 1000) {
            return@withContext null
        }

        User.create(
            id = entity.id,
            username = entity.username,
            email = entity.email,
            displayName = entity.displayName,
            bio = entity.bio,
            avatarUrl = entity.avatarUrl
        ).getOrNull()
    }
}
```

### Dependency Injection

**Platform-Specific DI Configuration:**

**Web (TypeScript with InversifyJS):**

```typescript
// web/di/container.ts
import { Container } from 'inversify';
import { IUserRepository } from '@shared/repositories/IUserRepository';
import { WebUserRepository } from '@web/repositories/UserRepository';
import { UpdateProfileUseCase } from '@shared/usecases/UpdateProfileUseCase';

export const container = new Container();

// Register repositories
container.bind<IUserRepository>('IUserRepository').to(WebUserRepository).inSingletonScope();

// Register use cases
container.bind<UpdateProfileUseCase>('UpdateProfileUseCase').to(UpdateProfileUseCase);

// Usage in React
import { useInjection } from '@web/hooks/useInjection';

function ProfilePage() {
  const updateProfile = useInjection<UpdateProfileUseCase>('UpdateProfileUseCase');

  const handleSubmit = async (data: ProfileData) => {
    const result = await updateProfile(userId, data.displayName, data.bio);
    // Handle result
  };
}
```

**iOS (Swift with Swinject):**

```swift
// iOS/DI/Container.swift
import Swinject

let container = Container()

// Register repositories
container.register(IUserRepository.self) { resolver in
    IOSUserRepository(
        apiClient: resolver.resolve(APIClient.self)!,
        context: resolver.resolve(NSManagedObjectContext.self)!
    )
}.inObjectScope(.container)

// Register use cases
container.register(UpdateProfileUseCase.self) { resolver in
    UpdateProfileUseCase(
        repository: resolver.resolve(IUserRepository.self)!,
        validator: resolver.resolve(ProfileValidator.self)!
    )
}

// Usage in SwiftUI
struct ProfileView: View {
    @Injected private var updateProfile: UpdateProfileUseCase

    func handleSubmit(data: ProfileData) async {
        let result = await updateProfile.invoke(
            userId: userId,
            displayName: data.displayName,
            bio: data.bio
        )
        // Handle result
    }
}
```

**Android (Kotlin with Hilt):**

```kotlin
// android/di/AppModule.kt
package com.example.di

import com.example.data.local.AppDatabase
import com.example.data.remote.ApiService
import com.example.domain.repositories.IUserRepository
import com.example.repositories.AndroidUserRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideUserRepository(
        apiService: ApiService,
        database: AppDatabase
    ): IUserRepository {
        return AndroidUserRepository(apiService, database.userDao())
    }

    @Provides
    fun provideUpdateProfileUseCase(
        repository: IUserRepository,
        validator: ProfileValidator
    ): UpdateProfileUseCase {
        return UpdateProfileUseCase(repository, validator)
    }
}

// Usage in Composable
@Composable
fun ProfileScreen(
    viewModel: ProfileViewModel = hiltViewModel()
) {
    val updateProfile by viewModel.updateProfile.collectAsState()

    Button(onClick = {
        viewModel.updateProfile(displayName, bio)
    }) {
        Text("Save Profile")
    }
}
```

---

## Offline-First Architecture

### Local-First Synchronization

**Conflict Resolution Strategy:**

```typescript
// shared/sync/ConflictResolver.ts
export enum ConflictResolutionStrategy {
  CLIENT_WINS,
  SERVER_WINS,
  LAST_WRITE_WINS,
  MERGE,
  MANUAL
}

export interface SyncConflict<T> {
  localVersion: T;
  serverVersion: T;
  localTimestamp: Date;
  serverTimestamp: Date;
  field: keyof T;
}

export class ConflictResolver<T> {
  resolve(
    conflict: SyncConflict<T>,
    strategy: ConflictResolutionStrategy
  ): T {
    switch (strategy) {
      case ConflictResolutionStrategy.CLIENT_WINS:
        return conflict.localVersion;

      case ConflictResolutionStrategy.SERVER_WINS:
        return conflict.serverVersion;

      case ConflictResolutionStrategy.LAST_WRITE_WINS:
        return conflict.localTimestamp > conflict.serverTimestamp
          ? conflict.localVersion
          : conflict.serverVersion;

      case ConflictResolutionStrategy.MERGE:
        return this.mergeVersions(conflict.localVersion, conflict.serverVersion);

      case ConflictResolutionStrategy.MANUAL:
        throw new Error('Manual conflict resolution required');
    }
  }

  private mergeVersions(local: T, server: T): T {
    // Field-by-field merge logic
    const merged: any = { ...server };

    for (const key in local) {
      if (local[key] !== server[key]) {
        // Custom merge logic per field type
        if (typeof local[key] === 'string' && local[key]) {
          merged[key] = local[key]; // Prefer non-empty strings
        } else if (typeof local[key] === 'number' && local[key] > 0) {
          merged[key] = Math.max(local[key] as number, server[key] as number);
        }
      }
    }

    return merged as T;
  }
}
```

**Delta Synchronization:**

```typescript
// shared/sync/DeltaSync.ts
export interface DeltaOperation {
  type: 'create' | 'update' | 'delete';
  entity: string;
  id: string;
  changes: Record<string, any>;
  timestamp: Date;
  version: number;
}

export class DeltaSyncEngine {
  private pendingOperations: DeltaOperation[] = [];
  private lastSyncTimestamp: Date | null = null;

  async trackChange(
    type: 'create' | 'update' | 'delete',
    entity: string,
    id: string,
    changes: Record<string, any>
  ): Promise<void> {
    const operation: DeltaOperation = {
      type,
      entity,
      id,
      changes,
      timestamp: new Date(),
      version: await this.getNextVersion(entity, id)
    };

    this.pendingOperations.push(operation);
    await this.persistPendingOperations();
  }

  async sync(): Promise<SyncResult> {
    try {
      // 1. Fetch server changes since last sync
      const serverChanges = await this.fetchServerChanges(this.lastSyncTimestamp);

      // 2. Apply server changes locally
      const conflicts: SyncConflict<any>[] = [];
      for (const change of serverChanges) {
        const conflict = await this.applyServerChange(change);
        if (conflict) {
          conflicts.push(conflict);
        }
      }

      // 3. Push local changes to server
      const pushResult = await this.pushLocalChanges();

      // 4. Update last sync timestamp
      this.lastSyncTimestamp = new Date();
      await this.persistSyncState();

      return {
        success: true,
        conflictsResolved: conflicts.length,
        changesPushed: pushResult.count,
        changesPulled: serverChanges.length
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  private async applyServerChange(change: DeltaOperation): Promise<SyncConflict<any> | null> {
    // Check if we have a local pending change for the same entity
    const localChange = this.pendingOperations.find(
      op => op.entity === change.entity && op.id === change.id
    );

    if (localChange && localChange.version !== change.version) {
      // Conflict detected
      return {
        localVersion: localChange.changes,
        serverVersion: change.changes,
        localTimestamp: localChange.timestamp,
        serverTimestamp: change.timestamp,
        field: 'all'
      };
    }

    // No conflict, apply server change
    await this.applyChange(change);
    return null;
  }

  private async pushLocalChanges(): Promise<{ count: number }> {
    if (this.pendingOperations.length === 0) {
      return { count: 0 };
    }

    // Batch push to reduce network requests
    await fetch('/api/sync/push', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        operations: this.pendingOperations,
        lastSyncTimestamp: this.lastSyncTimestamp
      })
    });

    const count = this.pendingOperations.length;
    this.pendingOperations = [];
    await this.persistPendingOperations();

    return { count };
  }
}
```

### Background Sync (Web Service Worker):**

```typescript
// web/service-worker.ts
self.addEventListener('sync', (event: any) => {
  if (event.tag === 'sync-data') {
    event.waitUntil(syncData());
  }
});

async function syncData() {
  const cache = await caches.open('pending-requests');
  const requests = await cache.keys();

  for (const request of requests) {
    try {
      const response = await fetch(request);
      if (response.ok) {
        await cache.delete(request);
      }
    } catch (error) {
      console.error('Sync failed:', error);
    }
  }
}

// Register background sync
if ('serviceWorker' in navigator && 'SyncManager' in window) {
  navigator.serviceWorker.ready.then(registration => {
    return registration.sync.register('sync-data');
  });
}
```

**iOS Background Sync (Background Tasks):**

```swift
// iOS/BackgroundSync/BackgroundSyncManager.swift
import BackgroundTasks

class BackgroundSyncManager {
    static let shared = BackgroundSyncManager()
    private let taskIdentifier = "com.example.app.sync"

    func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: taskIdentifier,
            using: nil
        ) { task in
            self.handleBackgroundSync(task: task as! BGAppRefreshTask)
        }
    }

    func scheduleBackgroundSync() {
        let request = BGAppRefreshTaskRequest(identifier: taskIdentifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60) // 15 minutes

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("Failed to schedule background sync: \(error)")
        }
    }

    private func handleBackgroundSync(task: BGAppRefreshTask) {
        task.expirationHandler = {
            task.setTaskCompleted(success: false)
        }

        Task {
            let syncEngine = DeltaSyncEngine()
            let result = await syncEngine.sync()

            task.setTaskCompleted(success: result.success)
            scheduleBackgroundSync() // Schedule next sync
        }
    }
}
```

**Android Background Sync (WorkManager):**

```kotlin
// android/sync/SyncWorker.kt
package com.example.sync

import android.content.Context
import androidx.work.*
import java.util.concurrent.TimeUnit

class SyncWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val syncEngine = DeltaSyncEngine()
        val result = syncEngine.sync()

        return if (result.success) {
            Result.success()
        } else {
            Result.retry()
        }
    }

    companion object {
        fun schedulePeriodicSync(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .setRequiresBatteryNotLow(true)
                .build()

            val syncRequest = PeriodicWorkRequestBuilder<SyncWorker>(
                15, TimeUnit.MINUTES
            )
                .setConstraints(constraints)
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    WorkRequest.MIN_BACKOFF_MILLIS,
                    TimeUnit.MILLISECONDS
                )
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                "periodic-sync",
                ExistingPeriodicWorkPolicy.KEEP,
                syncRequest
            )
        }
    }
}
```

---

## Code Examples

### Complete Feature: User Profile Update

**1. Shared Domain Model (Kotlin Multiplatform):**

```kotlin
// shared/domain/models/UpdateProfileRequest.kt
@Serializable
data class UpdateProfileRequest(
    val displayName: String?,
    val bio: String?
)

@Serializable
data class UpdateProfileResponse(
    val user: UserProfile,
    val success: Boolean
)
```

**2. Shared Use Case:**

```kotlin
// shared/domain/usecases/UpdateProfileUseCase.kt
class UpdateProfileUseCase(
    private val repository: IUserRepository,
    private val validator: ProfileValidator,
    private val syncEngine: DeltaSyncEngine
) {
    suspend operator fun invoke(
        userId: String,
        displayName: String?,
        bio: String?
    ): Result<UserProfile, UpdateProfileError> {
        // Validation
        displayName?.let {
            val validation = validator.validateDisplayName(it)
            if (!validation.isValid) {
                return Result.failure(
                    UpdateProfileError.ValidationError(validation.errors)
                )
            }
        }

        bio?.let {
            val validation = validator.validateBio(it)
            if (!validation.isValid) {
                return Result.failure(
                    UpdateProfileError.ValidationError(validation.errors)
                )
            }
        }

        // Track change for offline support
        syncEngine.trackChange(
            type = "update",
            entity = "user",
            id = userId,
            changes = mapOf(
                "displayName" to displayName,
                "bio" to bio
            )
        )

        // Update
        return repository.updateProfile(userId, displayName, bio)
            .mapError { error ->
                when (error.type) {
                    "NotFound" -> UpdateProfileError.UserNotFound
                    "NetworkError" -> UpdateProfileError.NetworkError(error.message)
                    else -> UpdateProfileError.Unknown(error.message)
                }
            }
    }
}

sealed class UpdateProfileError {
    data class ValidationError(val errors: List<String>) : UpdateProfileError()
    object UserNotFound : UpdateProfileError()
    data class NetworkError(val message: String) : UpdateProfileError()
    data class Unknown(val message: String) : UpdateProfileError()
}
```

**3. Web UI (React + TypeScript):**

```typescript
// web/components/ProfileForm.tsx
import React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useUpdateProfile } from '@web/hooks/useUpdateProfile';

const profileSchema = z.object({
  displayName: z.string().max(100).optional(),
  bio: z.string().max(500).optional()
});

type ProfileFormData = z.infer<typeof profileSchema>;

export function ProfileForm({ userId }: { userId: string }) {
  const { mutate: updateProfile, isLoading } = useUpdateProfile();

  const {
    register,
    handleSubmit,
    formState: { errors }
  } = useForm<ProfileFormData>({
    resolver: zodResolver(profileSchema)
  });

  const onSubmit = async (data: ProfileFormData) => {
    await updateProfile({
      userId,
      ...data
    });
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
      <div>
        <label htmlFor="displayName" className="block text-sm font-medium">
          Display Name
        </label>
        <input
          {...register('displayName')}
          id="displayName"
          type="text"
          className="mt-1 block w-full rounded-md border-gray-300"
        />
        {errors.displayName && (
          <p className="mt-1 text-sm text-red-600">{errors.displayName.message}</p>
        )}
      </div>

      <div>
        <label htmlFor="bio" className="block text-sm font-medium">
          Bio
        </label>
        <textarea
          {...register('bio')}
          id="bio"
          rows={4}
          className="mt-1 block w-full rounded-md border-gray-300"
        />
        {errors.bio && (
          <p className="mt-1 text-sm text-red-600">{errors.bio.message}</p>
        )}
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
      >
        {isLoading ? 'Saving...' : 'Save Profile'}
      </button>
    </form>
  );
}
```

**4. iOS UI (SwiftUI):**

```swift
// iOS/Views/ProfileFormView.swift
import SwiftUI

struct ProfileFormView: View {
    @StateObject private var viewModel: ProfileViewModel
    @State private var displayName: String = ""
    @State private var bio: String = ""
    @FocusState private var focusedField: Field?

    enum Field {
        case displayName, bio
    }

    init(userId: String) {
        _viewModel = StateObject(wrappedValue: ProfileViewModel(userId: userId))
    }

    var body: some View {
        Form {
            Section {
                TextField("Display Name", text: $displayName)
                    .focused($focusedField, equals: .displayName)
                    .textContentType(.name)
                    .autocapitalization(.words)

                if let error = viewModel.displayNameError {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.red)
                }
            } header: {
                Text("Display Name")
            }

            Section {
                TextEditor(text: $bio)
                    .focused($focusedField, equals: .bio)
                    .frame(minHeight: 100)

                if let error = viewModel.bioError {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.red)
                }
            } header: {
                Text("Bio")
            } footer: {
                Text("\(bio.count)/500")
                    .font(.caption)
                    .foregroundColor(bio.count > 500 ? .red : .secondary)
            }

            Section {
                Button(action: saveProfile) {
                    if viewModel.isLoading {
                        ProgressView()
                    } else {
                        Text("Save Profile")
                    }
                }
                .disabled(viewModel.isLoading || bio.count > 500)
            }
        }
        .navigationTitle("Edit Profile")
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    focusedField = nil
                }
            }
        }
    }

    private func saveProfile() {
        Task {
            await viewModel.updateProfile(
                displayName: displayName.isEmpty ? nil : displayName,
                bio: bio.isEmpty ? nil : bio
            )
        }
    }
}
```

**5. Android UI (Jetpack Compose):**

```kotlin
// android/ui/ProfileScreen.kt
package com.example.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun ProfileScreen(
    userId: String,
    viewModel: ProfileViewModel = hiltViewModel()
) {
    var displayName by remember { mutableStateOf("") }
    var bio by remember { mutableStateOf("") }

    val uiState by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        OutlinedTextField(
            value = displayName,
            onValueChange = { displayName = it },
            label = { Text("Display Name") },
            modifier = Modifier.fillMaxWidth(),
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

        Spacer(modifier = Modifier.height(16.dp))

        OutlinedTextField(
            value = bio,
            onValueChange = { if (it.length <= 500) bio = it },
            label = { Text("Bio") },
            modifier = Modifier
                .fillMaxWidth()
                .height(150.dp),
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
                    }
                    Text(
                        text = "${bio.length}/500",
                        color = if (bio.length > 500) {
                            MaterialTheme.colorScheme.error
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                }
            }
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = {
                viewModel.updateProfile(
                    userId = userId,
                    displayName = displayName.ifEmpty { null },
                    bio = bio.ifEmpty { null }
                )
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = !uiState.isLoading && bio.length <= 500
        ) {
            if (uiState.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(24.dp),
                    color = MaterialTheme.colorScheme.onPrimary
                )
            } else {
                Text("Save Profile")
            }
        }
    }
}
```

---

This platform architecture guide provides comprehensive patterns for building scalable, maintainable multi-platform applications with shared business logic, platform-specific optimizations, and offline-first capabilities.

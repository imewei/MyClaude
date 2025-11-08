# Architecture Patterns Library

> **Reference**: Full-stack architecture patterns for database, backend, frontend, and integration design

---

## Database Architecture Patterns

### Pattern 1: Normalized Relational Schema

**Use Case**: Transactional systems with complex relationships

**Design Principles**:
- 3NF normalization for data integrity
- Foreign key constraints for referential integrity
- Composite indexes for query optimization
- Soft deletes with timestamp tracking

**Example Schema**:
```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL
);

-- Posts table with foreign key
CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    content TEXT,
    status VARCHAR(20) DEFAULT 'draft',
    published_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL
);

-- Composite index for common queries
CREATE INDEX idx_posts_user_status ON posts(user_id, status) WHERE deleted_at IS NULL;
CREATE INDEX idx_posts_published ON posts(published_at DESC) WHERE status = 'published' AND deleted_at IS NULL;
```

**Migration Strategy**:
```python
# Alembic migration example
def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('username', sa.String(100), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )

def downgrade():
    op.drop_table('users')
```

---

### Pattern 2: Document-Oriented Schema

**Use Case**: Flexible schema requirements, hierarchical data

**Design Principles**:
- Embed related data for read performance
- Use references for many-to-many relationships
- Index frequently queried fields
- Schema validation with JSON Schema

**Example Schema (MongoDB)**:
```javascript
// User document with embedded profile
{
  _id: ObjectId("..."),
  email: "user@example.com",
  username: "johndoe",
  passwordHash: "...",
  profile: {
    firstName: "John",
    lastName: "Doe",
    avatar: "https://...",
    bio: "Software engineer",
    socialLinks: {
      twitter: "@johndoe",
      github: "johndoe"
    }
  },
  preferences: {
    theme: "dark",
    notifications: {
      email: true,
      push: false
    }
  },
  createdAt: ISODate("2024-01-01T00:00:00Z"),
  updatedAt: ISODate("2024-01-15T10:30:00Z")
}

// Post document with user reference
{
  _id: ObjectId("..."),
  userId: ObjectId("..."),  // Reference to User
  title: "My First Post",
  content: "Lorem ipsum...",
  tags: ["javascript", "mongodb"],
  metadata: {
    views: 1250,
    likes: 45,
    commentsCount: 12
  },
  status: "published",
  publishedAt: ISODate("2024-01-10T14:00:00Z"),
  createdAt: ISODate("2024-01-10T13:45:00Z"),
  updatedAt: ISODate("2024-01-10T14:00:00Z")
}
```

**Indexes**:
```javascript
db.users.createIndex({ email: 1 }, { unique: true });
db.users.createIndex({ username: 1 }, { unique: true });
db.posts.createIndex({ userId: 1, status: 1 });
db.posts.createIndex({ publishedAt: -1 });
db.posts.createIndex({ tags: 1 });
db.posts.createIndex({ "metadata.views": -1 });
```

---

## Backend Service Architecture Patterns

### Pattern 3: Layered Service Architecture

**Use Case**: Clean separation of concerns, testable code

**Layers**:
1. **API Layer** (Controllers/Routes): HTTP request handling
2. **Service Layer**: Business logic
3. **Repository Layer**: Data access
4. **Model Layer**: Data structures

**Example (FastAPI + SQLAlchemy)**:
```python
# models/user.py - Model Layer
from sqlalchemy import Column, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from database import Base
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

# repositories/user_repository.py - Repository Layer
from sqlalchemy.orm import Session
from models.user import User
from typing import Optional

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def create(self, user: User) -> User:
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

# services/user_service.py - Service Layer
from repositories.user_repository import UserRepository
from schemas.user import UserCreate, UserResponse
from security import hash_password, verify_password
import uuid

class UserService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo

    def create_user(self, user_data: UserCreate) -> UserResponse:
        # Business logic: validate, hash password, create user
        if self.user_repo.get_by_email(user_data.email):
            raise ValueError("Email already registered")

        hashed_password = hash_password(user_data.password)
        user = User(
            email=user_data.email,
            username=user_data.username,
            password_hash=hashed_password
        )

        created_user = self.user_repo.create(user)
        return UserResponse.from_orm(created_user)

# api/users.py - API Layer
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from dependencies import get_db
from services.user_service import UserService
from repositories.user_repository import UserRepository
from schemas.user import UserCreate, UserResponse

router = APIRouter(prefix="/api/users", tags=["users"])

@router.post("/", response_model=UserResponse, status_code=201)
def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    user_repo = UserRepository(db)
    user_service = UserService(user_repo)

    try:
        user = user_service.create_user(user_data)
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

### Pattern 4: API Contract Design (OpenAPI)

**Use Case**: Contract-first development, frontend-backend coordination

**OpenAPI Specification**:
```yaml
openapi: 3.0.3
info:
  title: User Management API
  version: 1.0.0
  description: RESTful API for user operations

servers:
  - url: https://api.example.com/v1
    description: Production server
  - url: https://staging-api.example.com/v1
    description: Staging server

paths:
  /users:
    post:
      summary: Create new user
      operationId: createUser
      tags:
        - users
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserCreate'
      responses:
        '201':
          description: User created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '400':
          description: Invalid input
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '409':
          description: Email already exists
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /users/{userId}:
    get:
      summary: Get user by ID
      operationId: getUserById
      tags:
        - users
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '404':
          description: User not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  schemas:
    UserCreate:
      type: object
      required:
        - email
        - username
        - password
      properties:
        email:
          type: string
          format: email
          example: user@example.com
        username:
          type: string
          minLength: 3
          maxLength: 100
          example: johndoe
        password:
          type: string
          minLength: 8
          format: password
          example: SecureP@ssw0rd

    UserResponse:
      type: object
      properties:
        id:
          type: string
          format: uuid
          example: 550e8400-e29b-41d4-a716-446655440000
        email:
          type: string
          format: email
          example: user@example.com
        username:
          type: string
          example: johndoe
        createdAt:
          type: string
          format: date-time
          example: 2024-01-01T00:00:00Z
        updatedAt:
          type: string
          format: date-time
          example: 2024-01-15T10:30:00Z

    Error:
      type: object
      properties:
        code:
          type: string
          example: VALIDATION_ERROR
        message:
          type: string
          example: Invalid email format
        details:
          type: object
          additionalProperties: true
```

---

## Frontend Architecture Patterns

### Pattern 5: Component-Based Architecture

**Use Case**: Reusable UI components, maintainable frontend

**Component Hierarchy**:
```
App
├── Layout
│   ├── Header
│   │   ├── Logo
│   │   ├── Navigation
│   │   └── UserMenu
│   ├── Sidebar (optional)
│   └── Footer
├── Pages
│   ├── HomePage
│   ├── UserProfilePage
│   ├── PostListPage
│   └── PostDetailPage
└── Shared Components
    ├── Button
    ├── Input
    ├── Card
    ├── Modal
    └── LoadingSpinner
```

**Example Implementation (React + TypeScript)**:
```typescript
// components/UserProfile/UserProfile.tsx
import React from 'react';
import { Card } from '../shared/Card';
import { Button } from '../shared/Button';
import { useUser } from '../../hooks/useUser';
import styles from './UserProfile.module.css';

interface UserProfileProps {
  userId: string;
}

export const UserProfile: React.FC<UserProfileProps> = ({ userId }) => {
  const { user, isLoading, error, updateUser } = useUser(userId);

  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!user) return <NotFound />;

  return (
    <Card className={styles.profileCard}>
      <div className={styles.header}>
        <img
          src={user.profile.avatar}
          alt={user.username}
          className={styles.avatar}
        />
        <div className={styles.info}>
          <h2>{user.username}</h2>
          <p>{user.profile.bio}</p>
        </div>
      </div>

      <div className={styles.stats}>
        <Stat label="Posts" value={user.stats.postsCount} />
        <Stat label="Followers" value={user.stats.followersCount} />
        <Stat label="Following" value={user.stats.followingCount} />
      </div>

      <div className={styles.actions}>
        <Button onClick={() => navigate(`/messages/${userId}`)}>
          Message
        </Button>
        <Button variant="secondary" onClick={handleFollow}>
          {user.isFollowing ? 'Unfollow' : 'Follow'}
        </Button>
      </div>
    </Card>
  );
};
```

---

### Pattern 6: State Management Architecture

**Use Case**: Complex state, global data sharing

**Zustand Store Example**:
```typescript
// stores/userStore.ts
import create from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface User {
  id: string;
  email: string;
  username: string;
  profile: UserProfile;
}

interface UserState {
  currentUser: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  updateProfile: (data: Partial<UserProfile>) => Promise<void>;
  fetchCurrentUser: () => Promise<void>;
}

export const useUserStore = create<UserState>()(
  devtools(
    persist(
      (set, get) => ({
        currentUser: null,
        isAuthenticated: false,
        isLoading: false,
        error: null,

        login: async (email, password) => {
          set({ isLoading: true, error: null });
          try {
            const response = await api.post('/auth/login', { email, password });
            const user = response.data.user;
            const token = response.data.token;

            localStorage.setItem('token', token);
            set({
              currentUser: user,
              isAuthenticated: true,
              isLoading: false
            });
          } catch (error) {
            set({
              error: error.message,
              isLoading: false
            });
          }
        },

        logout: () => {
          localStorage.removeItem('token');
          set({
            currentUser: null,
            isAuthenticated: false
          });
        },

        updateProfile: async (data) => {
          const { currentUser } = get();
          if (!currentUser) return;

          set({ isLoading: true });
          try {
            const response = await api.patch(`/users/${currentUser.id}/profile`, data);
            set({
              currentUser: { ...currentUser, profile: response.data },
              isLoading: false
            });
          } catch (error) {
            set({ error: error.message, isLoading: false });
          }
        },

        fetchCurrentUser: async () => {
          const token = localStorage.getItem('token');
          if (!token) return;

          set({ isLoading: true });
          try {
            const response = await api.get('/users/me');
            set({
              currentUser: response.data,
              isAuthenticated: true,
              isLoading: false
            });
          } catch (error) {
            localStorage.removeItem('token');
            set({
              error: error.message,
              isLoading: false,
              isAuthenticated: false
            });
          }
        }
      }),
      {
        name: 'user-storage',
        partialize: (state) => ({
          currentUser: state.currentUser,
          isAuthenticated: state.isAuthenticated
        })
      }
    )
  )
);
```

---

## Integration Patterns

### Pattern 7: API Client Architecture

**Use Case**: Centralized API communication, error handling

**Axios Instance Configuration**:
```typescript
// api/client.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import { useUserStore } from '../stores/userStore';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://api.example.com';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: Add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: Handle errors globally
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const { response } = error;

    // Handle 401 Unauthorized
    if (response?.status === 401) {
      const { logout } = useUserStore.getState();
      logout();
      window.location.href = '/login';
    }

    // Handle 403 Forbidden
    if (response?.status === 403) {
      // Show permission denied message
      console.error('Permission denied');
    }

    // Handle 500 Server Error
    if (response?.status && response.status >= 500) {
      // Show generic server error
      console.error('Server error occurred');
    }

    return Promise.reject(error);
  }
);

// API methods
export const api = {
  get: <T>(url: string, config?: AxiosRequestConfig) =>
    apiClient.get<T>(url, config),

  post: <T>(url: string, data?: any, config?: AxiosRequestConfig) =>
    apiClient.post<T>(url, data, config),

  put: <T>(url: string, data?: any, config?: AxiosRequestConfig) =>
    apiClient.put<T>(url, data, config),

  patch: <T>(url: string, data?: any, config?: AxiosRequestConfig) =>
    apiClient.patch<T>(url, data, config),

  delete: <T>(url: string, config?: AxiosRequestConfig) =>
    apiClient.delete<T>(url, config),
};
```

---

## Technology Stack Decision Matrix

| Criteria | React/FastAPI/PostgreSQL | Next.js/Django/MongoDB | Vue/NestJS/MySQL |
|----------|-------------------------|------------------------|------------------|
| **Type Safety** | 90% (TypeScript + Pydantic) | 85% (TypeScript + Django ORM) | 88% (TypeScript + TypeORM) |
| **Development Speed** | High (FastAPI auto docs) | Very High (Django admin) | High (NestJS CLI) |
| **Scalability** | Excellent (async FastAPI) | Good (sync Django) | Excellent (NestJS microservices) |
| **Ecosystem** | Large (React + Python) | Largest (Next.js + Django) | Large (Vue + Node.js) |
| **Learning Curve** | Moderate | Moderate-High | Moderate |
| **Best For** | APIs, microservices | Full-stack apps, CMS | Enterprise apps, real-time |

---

## Architecture Decision Records (ADR) Template

```markdown
# ADR-001: Choose Database Technology

## Status
Accepted

## Context
We need to select a primary database for our application that handles user data, posts, and relationships.

**Requirements**:
- ACID transactions for data integrity
- Complex queries with joins
- ~100K users, ~1M posts expected
- Read-heavy workload (80% reads, 20% writes)

## Decision
We will use PostgreSQL as the primary database.

## Consequences

**Positive**:
- ACID compliance ensures data integrity
- Advanced indexing (B-tree, GiST, GIN) for complex queries
- JSON/JSONB support for flexible schema
- Full-text search built-in
- Large ecosystem and community support

**Negative**:
- Vertical scaling limitations (requires sharding for >10M rows)
- More complex replication setup than MySQL
- Requires careful index management for performance

**Neutral**:
- Team has PostgreSQL experience
- Cloud providers offer managed PostgreSQL (RDS, CloudSQL)
```

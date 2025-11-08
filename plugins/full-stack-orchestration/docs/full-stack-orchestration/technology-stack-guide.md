# Technology Stack Guide

> **Reference**: Stack-specific implementation guidance for common full-stack combinations

## Stack 1: React + FastAPI + PostgreSQL

### Architecture Overview
- **Frontend**: React 18 + Vite + TypeScript + Tailwind CSS
- **Backend**: FastAPI + SQLAlchemy + Pydantic
- **Database**: PostgreSQL 15 + Redis (caching)
- **Deployment**: Docker + Kubernetes

### Project Structure
```
project/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── api/
│   │   └── stores/
│   ├── package.json
│   └── vite.config.ts
├── backend/
│   ├── app/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── api/
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── alembic/
└── docker-compose.yml
```

### Backend Setup (FastAPI)
```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import users, posts
from app.database import engine, Base

app = FastAPI(title="My API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables
Base.metadata.create_all(bind=engine)

# Include routers
app.include_router(users.router, prefix="/api", tags=["users"])
app.include_router(posts.router, prefix="/api", tags=["posts"])

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

### Frontend Setup (React + Vite)
```typescript
// frontend/vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
```

---

## Stack 2: Next.js + Django + MongoDB

### Architecture Overview
- **Frontend**: Next.js 14 + App Router + TypeScript + Tailwind
- **Backend**: Django 5 + Django REST Framework + Celery
- **Database**: MongoDB + Redis (caching + Celery)
- **Deployment**: Vercel (frontend) + Railway (backend)

### Django REST Framework Setup
```python
# backend/api/views.py
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import User, Post
from .serializers import UserSerializer, PostSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    @action(detail=True, methods=['post'])
    def follow(self, request, pk=None):
        user = self.get_object()
        request.user.following.add(user)
        return Response({'status': 'following'})
```

### Next.js API Routes
```typescript
// app/api/users/route.ts
import { NextResponse } from 'next/server';

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const page = searchParams.get('page') || '1';

  const response = await fetch(`${API_URL}/api/users/?page=${page}`, {
    headers: {
      'Content-Type': 'application/json',
    },
  });

  const data = await response.json();
  return NextResponse.json(data);
}
```

---

## Stack 3: Vue + NestJS + MySQL

### Architecture Overview
- **Frontend**: Vue 3 + Composition API + Pinia + Vuetify
- **Backend**: NestJS + TypeORM + Bull (queues)
- **Database**: MySQL 8 + Redis
- **Deployment**: Docker + AWS ECS

### NestJS Module Structure
```typescript
// backend/src/users/users.module.ts
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { UsersService } from './users.service';
import { UsersController } from './users.controller';
import { User } from './entities/user.entity';

@Module({
  imports: [TypeOrmModule.forFeature([User])],
  controllers: [UsersController],
  providers: [UsersService],
  exports: [UsersService],
})
export class UsersModule {}
```

### Vue 3 Composition API
```typescript
// frontend/src/composables/useUsers.ts
import { ref, computed } from 'vue';
import { api } from '@/api/client';
import type { User } from '@/types';

export function useUsers() {
  const users = ref<User[]>([]);
  const loading = ref(false);
  const error = ref<Error | null>(null);

  const fetchUsers = async () => {
    loading.value = true;
    try {
      const response = await api.get<User[]>('/users');
      users.value = response.data;
    } catch (err) {
      error.value = err as Error;
    } finally {
      loading.value = false;
    }
  };

  const activeUsers = computed(() =>
    users.value.filter(u => u.isActive)
  );

  return {
    users,
    loading,
    error,
    fetchUsers,
    activeUsers,
  };
}
```

---

## Technology Decision Matrix

| Feature | React/FastAPI/PostgreSQL | Next.js/Django/MongoDB | Vue/NestJS/MySQL |
|---------|-------------------------|----------------------|------------------|
| **Type Safety** | Excellent (TS + Pydantic) | Excellent (TS + Django) | Excellent (TS + TypeORM) |
| **SSR/SSG** | Client-side only | Built-in Next.js | Nuxt.js optional |
| **Real-time** | Add WebSockets | Django Channels | Socket.io built-in |
| **Admin Panel** | Build custom | Django Admin | NestJS Admin |
| **ORM Quality** | SQLAlchemy (excellent) | Django ORM (excellent) | TypeORM (good) |
| **Testing** | Pytest + Testing Library | Django Test + Jest | Jest + Vue Test Utils |
| **Deployment** | Docker + K8s | Vercel + Railway | Docker + AWS |
| **Learning Curve** | Moderate | Moderate-High | Moderate |
| **Best For** | APIs, microservices | Content sites, CMS | Enterprise apps |

---

## Environment Configuration

### Development `.env` Template
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/myapp_dev
REDIS_URL=redis://localhost:6379/0

# API
API_SECRET_KEY=your-secret-key-change-in-production
API_DEBUG=true
API_PORT=8000

# Frontend
VITE_API_URL=http://localhost:8000
VITE_APP_ENV=development

# Third-party
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
STRIPE_SECRET_KEY=sk_test_xxx
SENDGRID_API_KEY=SG.xxx
```

### Production Environment Variables
```bash
# NEVER commit these to git
DATABASE_URL=postgresql://user:pass@prod-db:5432/myapp
REDIS_URL=redis://prod-redis:6379/0
API_SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
ALLOWED_ORIGINS=https://myapp.com,https://www.myapp.com
```

---

## Package Management

### Python (backend)
```bash
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
alembic==1.12.1
pydantic==2.5.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
redis==5.0.1
celery==5.3.4

# Development
pytest==7.4.3
pytest-cov==4.1.0
black==23.11.0
ruff==0.1.6
```

### JavaScript (frontend)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "zustand": "^4.4.7",
    "axios": "^1.6.2",
    "@tanstack/react-query": "^5.12.2"
  },
  "devDependencies": {
    "@types/react": "^18.2.42",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.3.2",
    "vite": "^5.0.5",
    "tailwindcss": "^3.3.6",
    "eslint": "^8.54.0",
    "vitest": "^1.0.4",
    "@testing-library/react": "^14.1.2",
    "playwright": "^1.40.1"
  }
}
```

---

## Quick Start Commands

### React + FastAPI + PostgreSQL
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev

# Docker
docker-compose up
```

### Next.js + Django + MongoDB
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# Frontend
cd frontend
npm install
npm run dev
```

### Vue + NestJS + MySQL
```bash
# Backend
cd backend
npm install
npm run start:dev

# Frontend
cd frontend
npm install
npm run dev
```

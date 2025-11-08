# Django Project Structure

Production-ready Django 5.x project structure with async views, Django REST Framework, Celery task queue, and modern deployment patterns.

## Directory Structure

```
django-project/
├── pyproject.toml
├── README.md
├── .gitignore
├── .env.example
├── manage.py
├── config/
│   ├── __init__.py
│   ├── asgi.py
│   ├── wsgi.py
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── development.py
│   │   ├── production.py
│   │   └── test.py
│   └── urls.py
├── apps/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── admin.py
│   │   ├── urls.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_models.py
│   │       └── test_views.py
│   └── api/
│       ├── __init__.py
│       ├── apps.py
│       ├── serializers.py
│       ├── views.py
│       ├── urls.py
│       └── tests/
│           └── test_api.py
├── static/
├── media/
├── templates/
│   └── base.html
└── requirements/
    ├── base.txt
    ├── development.txt
    └── production.txt
```

## pyproject.toml

```toml
[project]
name = "django-project"
version = "0.1.0"
description = "Django 5.x project"
requires-python = ">=3.12"
dependencies = [
    "django>=5.0.0",
    "django-environ>=0.11.0",
    "djangorestframework>=3.14.0",
    "django-filter>=24.1",
    "psycopg[binary]>=3.1.0",
    "gunicorn>=21.2.0",
    "celery>=5.3.0",
    "redis>=5.0.0",
]

[project.optional-dependencies]
dev = [
    "django-debug-toolbar>=4.3.0",
    "django-extensions>=3.2.0",
    "pytest-django>=4.8.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
    "django-stubs>=4.2.0",
]

[tool.ruff]
line-length = 100
target-version = "py312"
extend-exclude = ["migrations"]

[tool.pytest.ini_options]
DJANGO_SETTINGS_MODULE = "config.settings.test"
python_files = ["test_*.py", "*_test.py"]
addopts = "-v --cov=apps --cov-report=term-missing"

[tool.mypy]
plugins = ["mypy_django_plugin.main"]
python_version = "3.12"
```

## config/settings/base.py

```python
import environ
from pathlib import Path

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Environment variables
env = environ.Env(
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, []),
)
environ.Env.read_env(BASE_DIR / ".env")

# Security
SECRET_KEY = env("SECRET_KEY")
DEBUG = env("DEBUG")
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS")

# Applications
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third party
    "rest_framework",
    "django_filters",
    # Local apps
    "apps.core",
    "apps.api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# Database
DATABASES = {
    "default": env.db("DATABASE_URL", default="postgres://localhost/mydb")
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

# Default primary key
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# REST Framework
REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 100,
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
}

# Celery Configuration
CELERY_BROKER_URL = env("CELERY_BROKER_URL", default="redis://localhost:6379/0")
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default="redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE
```

## apps/api/views.py (with DRF)

```python
from rest_framework import viewsets, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend

from apps.core.models import Item
from .serializers import ItemSerializer


class ItemViewSet(viewsets.ModelViewSet):
    """
    API endpoint for items with filtering and search
    """

    queryset = Item.objects.all()
    serializer_class = ItemSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["status", "created_at"]
    search_fields = ["name", "description"]
    ordering_fields = ["created_at", "name"]

    @action(detail=False, methods=["get"])
    def recent(self, request):
        """Get recently created items"""
        recent_items = self.queryset.order_by("-created_at")[:10]
        serializer = self.get_serializer(recent_items, many=True)
        return Response(serializer.data)
```

## Celery Tasks (apps/core/tasks.py)

```python
from celery import shared_task
from django.core.mail import send_mail


@shared_task
def send_email_task(subject: str, message: str, recipient_list: list[str]):
    """Send email asynchronously"""
    send_mail(
        subject=subject,
        message=message,
        from_email="noreply@example.com",
        recipient_list=recipient_list,
        fail_silently=False,
    )
    return f"Email sent to {len(recipient_list)} recipients"


@shared_task
def process_data_task(data_id: int):
    """Process data asynchronously"""
    # Your processing logic here
    return f"Processed data {data_id}"
```

## Management Commands

Create custom management command in `apps/core/management/commands/`:

```python
# apps/core/management/commands/seed_data.py
from django.core.management.base import BaseCommand
from apps.core.models import Item


class Command(BaseCommand):
    help = "Seed database with sample data"

    def handle(self, *args, **options):
        Item.objects.create(name="Sample", description="Sample item")
        self.stdout.write(self.style.SUCCESS("Successfully seeded data"))
```

## Running the Project

```bash
# Install dependencies
uv sync

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run development server
python manage.py runserver

# Run Celery worker (separate terminal)
celery -A config worker -l info

# Run tests
pytest
```

## Best Practices

### 1. Split Settings
- Use separate settings files for dev/prod/test
- Use `django-environ` for environment variables
- Never commit `.env` files

### 2. Async Views (Django 5.x)
```python
from django.http import JsonResponse


async def async_view(request):
    """Async view for I/O-bound operations"""
    result = await some_async_operation()
    return JsonResponse({"result": result})
```

### 3. Database Optimization
- Use `select_related()` for foreign keys
- Use `prefetch_related()` for reverse relations
- Add indexes for frequently queried fields
- Use database transactions for data integrity

### 4. Security Checklist
- Set `DEBUG = False` in production
- Configure `ALLOWED_HOSTS`
- Use strong `SECRET_KEY`
- Enable HTTPS with `SECURE_SSL_REDIRECT`
- Configure CSRF protection
- Use `django-ratelimit` for rate limiting

### 5. Testing Strategy
```python
import pytest
from django.test import Client


@pytest.mark.django_db
def test_item_creation():
    """Test item can be created"""
    from apps.core.models import Item

    item = Item.objects.create(name="Test")
    assert item.name == "Test"
```

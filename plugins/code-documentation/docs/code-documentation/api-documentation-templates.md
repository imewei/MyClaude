# API Documentation Templates

**Version**: 1.0.3
**Category**: code-documentation
**Purpose**: OpenAPI specifications, endpoint extraction, and schema generation templates

## Overview

Comprehensive templates and patterns for generating API documentation including OpenAPI specs, endpoint extraction from code, and interactive documentation.

## API Endpoint Extraction

### Python APIDocExtractor

```python
import ast
from typing import Dict, List

class APIDocExtractor:
    def extract_endpoints(self, code_path):
        """Extract API endpoints and their documentation from source code"""
        endpoints = []

        with open(code_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if self._is_route_decorator(decorator):
                        endpoint = {
                            'method': self._extract_method(decorator),
                            'path': self._extract_path(decorator),
                            'function': node.name,
                            'docstring': ast.get_docstring(node),
                            'parameters': self._extract_parameters(node),
                            'returns': self._extract_returns(node)
                        }
                        endpoints.append(endpoint)
        return endpoints

    def _extract_parameters(self, func_node):
        """Extract function parameters with types"""
        params = []
        for arg in func_node.args.args:
            param = {
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation else None,
                'required': True
            }
            params.append(param)
        return params

    def _is_route_decorator(self, decorator):
        """Check if decorator is a route decorator"""
        if isinstance(decorator, ast.Call):
            if hasattr(decorator.func, 'attr'):
                return decorator.func.attr in ['get', 'post', 'put', 'delete', 'patch']
            if hasattr(decorator.func, 'id'):
                return decorator.func.id in ['route', 'app']
        return False

    def _extract_method(self, decorator):
        """Extract HTTP method from decorator"""
        if isinstance(decorator, ast.Call):
            if hasattr(decorator.func, 'attr'):
                return decorator.func.attr.upper()
        return 'GET'

    def _extract_path(self, decorator):
        """Extract path from decorator"""
        if isinstance(decorator, ast.Call) and decorator.args:
            return ast.literal_eval(decorator.args[0])
        return '/'
```

### Schema Extraction from Pydantic

```python
def extract_pydantic_schemas(file_path):
    """Extract Pydantic model definitions for API documentation"""
    schemas = []

    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(base.id == 'BaseModel' for base in node.bases if hasattr(base, 'id')):
                schema = {
                    'name': node.name,
                    'description': ast.get_docstring(node),
                    'fields': []
                }

                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        field = {
                            'name': item.target.id,
                            'type': ast.unparse(item.annotation),
                            'required': item.value is None
                        }
                        schema['fields'].append(field)
                schemas.append(schema)
    return schemas
```

## OpenAPI Specification Templates

### Complete OpenAPI 3.0 Template

```yaml
openapi: 3.0.0
info:
  title: ${API_TITLE}
  version: ${VERSION}
  description: |
    ${DESCRIPTION}

    ## Authentication
    ${AUTH_DESCRIPTION}

    ## Rate Limiting
    - 100 requests per minute per API key
    - 10,000 requests per day per API key

servers:
  - url: https://api.example.com/v1
    description: Production server
  - url: https://staging-api.example.com/v1
    description: Staging server

security:
  - bearerAuth: []

paths:
  /users:
    get:
      summary: List all users
      operationId: listUsers
      tags:
        - Users
      parameters:
        - name: page
          in: query
          description: Page number for pagination
          schema:
            type: integer
            default: 1
            minimum: 1
        - name: limit
          in: query
          description: Number of items per page
          schema:
            type: integer
            default: 20
            minimum: 1
            maximum: 100
        - name: sort
          in: query
          description: Sort field and order
          schema:
            type: string
            enum: [created_asc, created_desc, name_asc, name_desc]
            default: created_desc
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
              examples:
                success:
                  value:
                    data:
                      - id: "123e4567-e89b-12d3-a456-426614174000"
                        email: "user@example.com"
                        name: "John Doe"
                        createdAt: "2024-01-01T00:00:00Z"
                    pagination:
                      page: 1
                      limit: 20
                      total: 100
                      pages: 5
        '401':
          $ref: '#/components/responses/Unauthorized'
        '429':
          $ref: '#/components/responses/RateLimitExceeded'

    post:
      summary: Create a new user
      operationId: createUser
      tags:
        - Users
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
            examples:
              newUser:
                value:
                  email: "newuser@example.com"
                  name: "Jane Smith"
                  password: "SecureP@ssw0rd"
      responses:
        '201':
          description: User created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          description: Email already exists
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /users/{userId}:
    get:
      summary: Get user by ID
      operationId: getUser
      tags:
        - Users
      parameters:
        - name: userId
          in: path
          required: true
          description: User ID
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          $ref: '#/components/responses/NotFound'

    put:
      summary: Update user
      operationId: updateUser
      tags:
        - Users
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
              $ref: '#/components/schemas/UpdateUserRequest'
      responses:
        '200':
          description: User updated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          $ref: '#/components/responses/NotFound'

    delete:
      summary: Delete user
      operationId: deleteUser
      tags:
        - Users
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '204':
          description: User deleted
        '404':
          $ref: '#/components/responses/NotFound'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  schemas:
    User:
      type: object
      required:
        - id
        - email
        - createdAt
      properties:
        id:
          type: string
          format: uuid
          description: Unique user identifier
        email:
          type: string
          format: email
          description: User's email address
        name:
          type: string
          description: User's full name
        createdAt:
          type: string
          format: date-time
          description: Account creation timestamp
        updatedAt:
          type: string
          format: date-time
          description: Last update timestamp

    CreateUserRequest:
      type: object
      required:
        - email
        - password
      properties:
        email:
          type: string
          format: email
        name:
          type: string
          minLength: 1
          maxLength: 100
        password:
          type: string
          format: password
          minLength: 8
          description: Must contain uppercase, lowercase, number, and special character

    UpdateUserRequest:
      type: object
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        email:
          type: string
          format: email

    Pagination:
      type: object
      properties:
        page:
          type: integer
          description: Current page number
        limit:
          type: integer
          description: Items per page
        total:
          type: integer
          description: Total number of items
        pages:
          type: integer
          description: Total number of pages

    Error:
      type: object
      required:
        - error
        - message
      properties:
        error:
          type: string
          description: Error code
        message:
          type: string
          description: Human-readable error message
        details:
          type: object
          description: Additional error details

  responses:
    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "unauthorized"
            message: "Valid authentication credentials required"

    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "not_found"
            message: "The requested resource was not found"

    BadRequest:
      description: Invalid request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "bad_request"
            message: "Invalid input parameters"
            details:
              email: "Invalid email format"

    RateLimitExceeded:
      description: Rate limit exceeded
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "rate_limit_exceeded"
            message: "Too many requests. Try again later."
      headers:
        X-RateLimit-Limit:
          schema:
            type: integer
          description: Request limit per window
        X-RateLimit-Remaining:
          schema:
            type: integer
          description: Remaining requests in current window
        X-RateLimit-Reset:
          schema:
            type: integer
          description: Timestamp when rate limit resets
```

## Code Example Generation

### Multi-Language Code Examples

```python
def generate_code_examples(endpoint):
    """Generate code examples for API endpoints in multiple languages"""
    examples = {}

    # Python requests
    examples['python'] = f'''
import requests

url = "https://api.example.com{endpoint['path']}"
headers = {{"Authorization": "Bearer YOUR_API_KEY"}}

response = requests.{endpoint['method'].lower()}(url, headers=headers)
print(response.json())
'''

    # Python httpx (async)
    examples['python_async'] = f'''
import httpx
import asyncio

async def call_api():
    async with httpx.AsyncClient() as client:
        response = await client.{endpoint['method'].lower()}(
            "https://api.example.com{endpoint['path']}",
            headers={{"Authorization": "Bearer YOUR_API_KEY"}}
        )
        return response.json()

result = asyncio.run(call_api())
print(result)
'''

    # JavaScript fetch
    examples['javascript'] = f'''
const response = await fetch('https://api.example.com{endpoint['path']}', {{
    method: '{endpoint['method']}',
    headers: {{
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json'
    }}
}});

const data = await response.json();
console.log(data);
'''

    # cURL
    examples['curl'] = f'''
curl -X {endpoint['method']} https://api.example.com{endpoint['path']} \\
    -H "Authorization: Bearer YOUR_API_KEY" \\
    -H "Content-Type: application/json"
'''

    # Go
    examples['go'] = f'''
package main

import (
    "fmt"
    "net/http"
    "io/ioutil"
)

func main() {{
    client := &http.Client{{}}
    req, _ := http.NewRequest("{endpoint['method']}", "https://api.example.com{endpoint['path']}", nil)
    req.Header.Set("Authorization", "Bearer YOUR_API_KEY")

    resp, _ := client.Do(req)
    defer resp.Body.Close()

    body, _ := ioutil.ReadAll(resp.Body)
    fmt.Println(string(body))
}}
'''

    return examples
```

## Interactive Documentation Setup

### Swagger UI HTML Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Documentation</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@latest/swagger-ui.css">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        .topbar {
            display: none;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>

    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@latest/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: "/api/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                layout: "BaseLayout",
                defaultModelsExpandDepth: 1,
                defaultModelExpandDepth: 1,
                docExpansion: "list",
                filter: true,
                showRequestHeaders: true,
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                tryItOutEnabled: true,
                requestInterceptor: (request) => {
                    // Add custom headers
                    request.headers['X-Custom-Header'] = 'value';
                    return request;
                }
            });
            window.ui = ui;
        }
    </script>
</body>
</html>
```

### Redoc HTML Template

```html
<!DOCTYPE html>
<html>
<head>
    <title>API Reference</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
    </style>
</head>
<body>
    <redoc spec-url='/api/openapi.json' lazy-rendering></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>
</body>
</html>
```

## FastAPI Auto-Documentation

### FastAPI with Pydantic Models

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

app = FastAPI(
    title="User API",
    description="API for managing users",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

class User(BaseModel):
    """User model with full details"""
    id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User's email address")
    name: Optional[str] = Field(None, description="User's full name")
    created_at: datetime = Field(..., description="Account creation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "user@example.com",
                "name": "John Doe",
                "created_at": "2024-01-01T00:00:00Z"
            }
        }

class CreateUserRequest(BaseModel):
    """Request body for creating a new user"""
    email: EmailStr = Field(..., description="User's email address")
    name: str = Field(..., min_length=1, max_length=100, description="User's full name")
    password: str = Field(..., min_length=8, description="User password")

@app.get("/users", response_model=List[User], tags=["Users"])
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
):
    """
    List all users with pagination.

    Returns a paginated list of users.
    """
    # Implementation
    pass

@app.post("/users", response_model=User, status_code=201, tags=["Users"])
async def create_user(user: CreateUserRequest):
    """
    Create a new user.

    Creates a new user account with the provided information.
    """
    # Implementation
    pass
```

## Usage Examples

### Generate OpenAPI from Code

```python
# Extract endpoints
extractor = APIDocExtractor()
endpoints = extractor.extract_endpoints('app/routes.py')

# Generate OpenAPI spec
openapi_spec = generate_openapi_spec(
    title="My API",
    version="1.0.0",
    endpoints=endpoints
)

# Save to file
with open('openapi.json', 'w') as f:
    json.dump(openapi_spec, f, indent=2)
```

### Generate Code Examples

```python
endpoint = {
    'method': 'POST',
    'path': '/users',
    'parameters': [...]
}

examples = generate_code_examples(endpoint)
print(examples['python'])
print(examples['curl'])
```

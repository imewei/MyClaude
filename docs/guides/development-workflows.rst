Development Workflows
=====================

This guide demonstrates how to combine development plugins for building modern applications, from API development to full-stack systems. Learn best practices for integrating frontend, backend, testing, and deployment tools.

Overview
--------

Development workflows in this marketplace span the complete software development lifecycle:

- **Backend Development**: :term:`REST API`, :term:`Microservices`, database integration
- **Frontend Development**: Modern web and mobile applications
- **Testing**: :term:`TDD`, integration testing, and quality assurance
- **Integration**: Combining multiple plugins for full-stack development

Multi-Plugin Workflow: Building a REST API with Testing
--------------------------------------------------------

This workflow combines :doc:`/plugins/python-development`, :doc:`/plugins/backend-development`, and :doc:`/plugins/unit-testing` to create a well-tested RESTful API.

Prerequisites
~~~~~~~~~~~~~

Before starting, ensure you have:

- Python 3.12+ installed
- Understanding of :term:`REST API` principles
- Basic knowledge of :term:`TDD` methodology
- Familiarity with FastAPI or Flask frameworks
- PostgreSQL or similar database

See :term:`REST API`, :term:`TDD`, and :term:`Microservices` in the :doc:`/glossary` for background.

Step 1: Initialize Python Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :doc:`/plugins/python-development` to set up your project structure:

.. code-block:: bash

   # Create project structure
   mkdir api-project && cd api-project

   # Initialize virtual environment
   python3.12 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install fastapi uvicorn sqlalchemy pytest pytest-cov

Create project structure:

.. code-block:: text

   api-project/
   ├── app/
   │   ├── __init__.py
   │   ├── main.py
   │   ├── models/
   │   ├── routes/
   │   └── services/
   ├── tests/
   │   ├── __init__.py
   │   ├── test_api.py
   │   └── test_services.py
   └── requirements.txt

Step 2: Design API with Backend Plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :doc:`/plugins/backend-development` patterns for clean architecture:

.. code-block:: python

   # app/main.py
   from fastapi import FastAPI, Depends, HTTPException
   from sqlalchemy.orm import Session
   from app.database import get_db
   from app.models import User
   from app.schemas import UserCreate, UserResponse

   app = FastAPI(title="Example API", version="1.0.0")

   @app.post("/users/", response_model=UserResponse)
   def create_user(user: UserCreate, db: Session = Depends(get_db)):
       """Create a new user."""
       db_user = User(**user.dict())
       db.add(db_user)
       db.commit()
       db.refresh(db_user)
       return db_user

   @app.get("/users/{user_id}", response_model=UserResponse)
   def get_user(user_id: int, db: Session = Depends(get_db)):
       """Retrieve a user by ID."""
       user = db.query(User).filter(User.id == user_id).first()
       if user is None:
           raise HTTPException(status_code=404, detail="User not found")
       return user

Step 3: Implement Test-Driven Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :doc:`/plugins/unit-testing` to create comprehensive tests:

.. code-block:: python

   # tests/test_api.py
   import pytest
   from fastapi.testclient import TestClient
   from app.main import app
   from app.database import Base, engine

   @pytest.fixture
   def client():
       Base.metadata.create_all(bind=engine)
       yield TestClient(app)
       Base.metadata.drop_all(bind=engine)

   def test_create_user(client):
       """Test user creation endpoint."""
       response = client.post(
           "/users/",
           json={"name": "Test User", "email": "test@example.com"}
       )
       assert response.status_code == 200
       data = response.json()
       assert data["name"] == "Test User"
       assert "id" in data

   def test_get_user(client):
       """Test user retrieval endpoint."""
       # Create user first
       create_response = client.post(
           "/users/",
           json={"name": "Test User", "email": "test@example.com"}
       )
       user_id = create_response.json()["id"]

       # Get user
       response = client.get(f"/users/{user_id}")
       assert response.status_code == 200
       assert response.json()["name"] == "Test User"

   def test_get_nonexistent_user(client):
       """Test 404 error for missing user."""
       response = client.get("/users/9999")
       assert response.status_code == 404

Run tests with coverage:

.. code-block:: bash

   # Run tests with coverage report
   pytest --cov=app --cov-report=html tests/

   # View coverage report
   open htmlcov/index.html

Step 4: Add Database Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement :term:`ORM` models and database connections:

.. code-block:: python

   # app/models.py
   from sqlalchemy import Column, Integer, String, DateTime
   from sqlalchemy.ext.declarative import declarative_base
   from datetime import datetime

   Base = declarative_base()

   class User(Base):
       __tablename__ = "users"

       id = Column(Integer, primary_key=True, index=True)
       name = Column(String, index=True)
       email = Column(String, unique=True, index=True)
       created_at = Column(DateTime, default=datetime.utcnow)

   # app/database.py
   from sqlalchemy import create_engine
   from sqlalchemy.orm import sessionmaker

   DATABASE_URL = "postgresql://user:password@localhost/dbname"

   engine = create_engine(DATABASE_URL)
   SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

   def get_db():
       db = SessionLocal()
       try:
           yield db
       finally:
           db.close()

Expected Outcomes
~~~~~~~~~~~~~~~~~

After completing this workflow, you will have:

- A fully functional REST API with FastAPI
- Comprehensive test suite with >80% coverage
- Clean separation of concerns (models, routes, services)
- Database integration with SQLAlchemy ORM
- Development best practices applied throughout

Workflow: Full-Stack Application Development
---------------------------------------------

This workflow integrates :doc:`/plugins/frontend-mobile-development`, :doc:`/plugins/backend-development`, and :doc:`/plugins/python-development` for complete application development.

Prerequisites
~~~~~~~~~~~~~

- Node.js 18+ and npm
- Python 3.12+
- Understanding of :term:`Microservices` architecture

Step 1: Set Up Frontend
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create React application
   npx create-react-app frontend
   cd frontend

   # Install dependencies
   npm install axios react-router-dom

Create API service:

.. code-block:: javascript

   // src/services/api.js
   import axios from 'axios';

   const API_BASE = 'http://localhost:8000';

   export const api = {
       async getUsers() {
           const response = await axios.get(`${API_BASE}/users/`);
           return response.data;
       },

       async createUser(userData) {
           const response = await axios.post(`${API_BASE}/users/`, userData);
           return response.data;
       },

       async getUser(id) {
           const response = await axios.get(`${API_BASE}/users/${id}`);
           return response.data;
       }
   };

Step 2: Implement Frontend Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: jsx

   // src/components/UserList.jsx
   import React, { useState, useEffect } from 'react';
   import { api } from '../services/api';

   function UserList() {
       const [users, setUsers] = useState([]);
       const [loading, setLoading] = useState(true);

       useEffect(() => {
           async function fetchUsers() {
               try {
                   const data = await api.getUsers();
                   setUsers(data);
               } catch (error) {
                   console.error('Error fetching users:', error);
               } finally {
                   setLoading(false);
               }
           }
           fetchUsers();
       }, []);

       if (loading) return <div>Loading...</div>;

       return (
           <div>
               <h1>Users</h1>
               <ul>
                   {users.map(user => (
                       <li key={user.id}>{user.name} - {user.email}</li>
                   ))}
               </ul>
           </div>
       );
   }

   export default UserList;

Step 3: Connect Frontend and Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable CORS in the backend:

.. code-block:: python

   # app/main.py
   from fastapi.middleware.cors import CORSMiddleware

   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )

Run both servers:

.. code-block:: bash

   # Terminal 1: Backend
   uvicorn app.main:app --reload

   # Terminal 2: Frontend
   cd frontend && npm start

Workflow: Microservices Architecture
-------------------------------------

Build scalable :term:`Microservices` with :doc:`/plugins/backend-development` and :doc:`/plugins/python-development`.

Prerequisites
~~~~~~~~~~~~~

- Understanding of distributed systems
- Docker for containerization
- API gateway knowledge

Step 1: Design Service Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Architecture:
   ├── API Gateway (Port 8000)
   ├── User Service (Port 8001)
   ├── Order Service (Port 8002)
   └── Notification Service (Port 8003)

Step 2: Implement Individual Services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # user-service/main.py
   from fastapi import FastAPI
   import httpx

   app = FastAPI()

   @app.get("/users/{user_id}")
   async def get_user(user_id: int):
       return {"id": user_id, "name": "User Name"}

   # order-service/main.py
   app = FastAPI()

   @app.post("/orders")
   async def create_order(order_data: dict):
       # Call user service to validate user
       async with httpx.AsyncClient() as client:
           user = await client.get(f"http://user-service:8001/users/{order_data['user_id']}")

       # Create order
       return {"order_id": 1, "status": "created"}

Step 3: Implement Service Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # shared/messaging.py
   import aio_pika
   import json

   async def publish_event(event_type: str, data: dict):
       """Publish event to message queue."""
       connection = await aio_pika.connect_robust("amqp://guest:guest@localhost/")

       async with connection:
           channel = await connection.channel()

           message = aio_pika.Message(
               body=json.dumps({"type": event_type, "data": data}).encode()
           )

           await channel.default_exchange.publish(
               message, routing_key="events"
           )

Integration Patterns
--------------------

Common Development Combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**API + Testing + Documentation**
   :doc:`/plugins/python-development` + :doc:`/plugins/backend-development` + :doc:`/plugins/unit-testing` + :doc:`/plugins/code-documentation`

   Build production-ready APIs with comprehensive testing and documentation.

**Full-Stack with Modern Frontend**
   :doc:`/plugins/frontend-mobile-development` + :doc:`/plugins/backend-development` + :doc:`/plugins/javascript-typescript`

   Create complete web applications with React/Vue and FastAPI/Django.

**Microservices with Deployment**
   :doc:`/plugins/backend-development` + :doc:`/plugins/cicd-automation` + :doc:`/plugins/observability-monitoring`

   Build scalable distributed systems with automated deployment and monitoring.

Best Practices
~~~~~~~~~~~~~~

1. **API Design**: Follow RESTful principles and versioning
2. **Testing**: Maintain >80% code coverage
3. **Documentation**: Keep API docs up-to-date with OpenAPI
4. **Error Handling**: Use consistent error responses
5. **Security**: Implement authentication and input validation
6. **Performance**: Use caching and database indexing

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**CORS Errors**
   - Configure CORS middleware in backend
   - Check allowed origins match frontend URL
   - Verify preflight requests are handled

**Database Connection Issues**
   - Check connection string format
   - Verify database server is running
   - Use connection pooling for production

**Test Failures**
   - Isolate test database from development
   - Use fixtures for consistent test data
   - Mock external dependencies

Next Steps
----------

- Explore :doc:`devops-workflows` for deployment automation
- See :doc:`/plugins/llm-application-dev` for AI integration
- Review :doc:`/plugins/quality-engineering` for advanced testing
- Check :doc:`/categories/development` for all development plugins

Additional Resources
--------------------

- `FastAPI Documentation <https://fastapi.tiangolo.com/>`_
- `pytest Documentation <https://docs.pytest.org/>`_
- `SQLAlchemy ORM Guide <https://docs.sqlalchemy.org/en/20/orm/>`_
- `React Best Practices <https://react.dev/learn>`_

See Also
--------

- :doc:`scientific-workflows` - Research computing patterns
- :doc:`infrastructure-workflows` - Cloud infrastructure setup
- :doc:`/integration-map` - Plugin compatibility reference

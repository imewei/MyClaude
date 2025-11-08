#!/usr/bin/env python3
"""
Test Corpus Generator for Plugin Triggering Pattern Testing

Generates diverse sample projects for each plugin category to test activation accuracy.
Includes edge cases, multi-language projects, and negative test samples.

Usage:
    python3 tools/test-corpus-generator.py
    python3 tools/test-corpus-generator.py --output-dir custom-test-corpus
    python3 tools/test-corpus-generator.py --categories scientific-computing development
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TestCorpusSample:
    """Represents a single test corpus sample."""

    name: str
    category: str
    description: str
    expected_plugins: List[str]
    files: Dict[str, str]  # filename -> content
    expected_trigger: bool  # Should this trigger any plugin?
    is_edge_case: bool = False
    is_negative_test: bool = False
    is_multi_language: bool = False


class TestCorpusGenerator:
    """Generates comprehensive test corpus for plugin triggering validation."""

    def __init__(self, output_dir: str = "test-corpus"):
        self.output_dir = Path(output_dir)
        self.samples: List[TestCorpusSample] = []

    def generate_scientific_computing_samples(self) -> List[TestCorpusSample]:
        """Generate test samples for scientific computing plugins."""
        samples = []

        # Julia + SciML sample
        samples.append(TestCorpusSample(
            name="julia-diffeq-project",
            category="scientific-computing",
            description="Julia project with differential equations",
            expected_plugins=["julia-development"],
            files={
                "Project.toml": '''
name = "DiffEqExample"
version = "0.1.0"

[deps]
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
''',
                "src/model.jl": '''
using DifferentialEquations
using Plots

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
p = [10.0, 28.0, 8/3]
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob)
plot(sol)
'''
            },
            expected_trigger=True,
            is_edge_case=False
        ))

        # Python + JAX sample
        samples.append(TestCorpusSample(
            name="jax-neural-ode-project",
            category="scientific-computing",
            description="Python project with JAX for neural ODEs",
            expected_plugins=["python-development", "jax-implementation"],
            files={
                "requirements.txt": '''
jax>=0.4.0
jaxlib>=0.4.0
optax>=0.1.0
numpy>=1.24.0
''',
                "neural_ode.py": '''
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax

def neural_ode_dynamics(state, t, params):
    """Neural ODE dynamics function."""
    x = state
    for W, b in params:
        x = jnp.tanh(x @ W + b)
    return x

def solve_ode(params, initial_state, t_span):
    """Solve ODE using Euler method."""
    dt = 0.01
    trajectory = [initial_state]
    state = initial_state

    for t in jnp.arange(t_span[0], t_span[1], dt):
        dx = neural_ode_dynamics(state, t, params)
        state = state + dx * dt
        trajectory.append(state)

    return jnp.array(trajectory)
'''
            },
            expected_trigger=True,
            is_multi_language=False
        ))

        # HPC + MPI sample
        samples.append(TestCorpusSample(
            name="mpi-simulation-project",
            category="scientific-computing",
            description="C++ HPC project with MPI parallelization",
            expected_plugins=["hpc-computing"],
            files={
                "Makefile": '''
CXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -fopenmp
LIBS = -lmpi

all: simulation

simulation: simulation.cpp
\t$(CXX) $(CXXFLAGS) -o simulation simulation.cpp $(LIBS)

clean:
\trm -f simulation
''',
                "simulation.cpp": '''
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000;
    std::vector<double> data(N / size);

    #pragma omp parallel for
    for (int i = 0; i < data.size(); ++i) {
        data[i] = rank * (N / size) + i;
    }

    double local_sum = 0.0;
    #pragma omp parallel for reduction(+:local_sum)
    for (int i = 0; i < data.size(); ++i) {
        local_sum += data[i];
    }

    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Global sum: " << global_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
'''
            },
            expected_trigger=True,
            is_edge_case=False
        ))

        # Molecular simulation sample
        samples.append(TestCorpusSample(
            name="molecular-dynamics-project",
            category="scientific-computing",
            description="Molecular dynamics simulation with LAMMPS",
            expected_plugins=["molecular-simulation"],
            files={
                "lammps_input.in": '''
# LAMMPS input script for water simulation
units real
atom_style full

read_data water.data

pair_style lj/cut/coul/long 10.0
pair_coeff * * 0.0 0.0
pair_coeff 1 1 0.1553 3.166  # O-O
kspace_style pppm 1.0e-5

bond_style harmonic
bond_coeff 1 450.0 0.9572

angle_style harmonic
angle_coeff 1 55.0 104.52

fix 1 all nve
fix 2 all temp/csvr 300.0 300.0 100.0 54324

timestep 1.0
thermo 100
dump 1 all custom 100 traj.lammpstrj id type x y z

run 10000
''',
                "analysis.py": '''
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rdf

# Load trajectory
u = mda.Universe("water.data", "traj.lammpstrj")

# Compute radial distribution function
rdf_analysis = rdf.InterRDF(u.select_atoms("type 1"), u.select_atoms("type 1"))
rdf_analysis.run()

print(f"RDF computed: {len(rdf_analysis.results.rdf)} bins")
'''
            },
            expected_trigger=True,
            is_edge_case=False
        ))

        # Deep learning sample
        samples.append(TestCorpusSample(
            name="pytorch-transformer-project",
            category="scientific-computing",
            description="PyTorch transformer model training",
            expected_plugins=["deep-learning", "python-development"],
            files={
                "requirements.txt": '''
torch>=1.0.2
transformers>=4.30.0
datasets>=2.10.0
wandb>=0.15.0
''',
                "train.py": '''
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import wandb

class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        return self.classifier(x)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        wandb.log({"train_loss": loss.item()})

    return total_loss / len(dataloader)
'''
            },
            expected_trigger=True,
            is_multi_language=False
        ))

        return samples

    def generate_development_samples(self) -> List[TestCorpusSample]:
        """Generate test samples for development plugins."""
        samples = []

        # JavaScript/TypeScript sample
        samples.append(TestCorpusSample(
            name="typescript-react-app",
            category="development",
            description="TypeScript React application with modern tooling",
            expected_plugins=["javascript-typescript", "frontend-mobile-development"],
            files={
                "package.json": '''
{
  "name": "react-app",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "test": "vitest"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.0.0",
    "vite": "^4.3.0",
    "vitest": "^0.30.0"
  }
}
''',
                "tsconfig.json": '''
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020", "DOM"],
    "jsx": "react-jsx",
    "strict": true,
    "moduleResolution": "bundler"
  }
}
''',
                "src/App.tsx": '''
import { useState } from 'react';

interface TodoItem {
  id: number;
  text: string;
  completed: boolean;
}

export default function App() {
  const [todos, setTodos] = useState<TodoItem[]>([]);
  const [input, setInput] = useState('');

  const addTodo = () => {
    if (input.trim()) {
      setTodos([...todos, { id: Date.now(), text: input, completed: false }]);
      setInput('');
    }
  };

  const toggleTodo = (id: number) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };

  return (
    <div>
      <h1>Todo App</h1>
      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={addTodo}>Add</button>
      <ul>
        {todos.map(todo => (
          <li key={todo.id} onClick={() => toggleTodo(todo.id)}>
            {todo.completed ? '✓ ' : ''}{todo.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
'''
            },
            expected_trigger=True,
            is_edge_case=False
        ))

        # Rust systems programming sample
        samples.append(TestCorpusSample(
            name="rust-cli-tool",
            category="development",
            description="Rust CLI tool with async I/O",
            expected_plugins=["systems-programming", "cli-tool-design"],
            files={
                "Cargo.toml": '''
[package]
name = "cli-tool"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.28", features = ["full"] }
clap = { version = "4.3", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
''',
                "src/main.rs": '''
use clap::Parser;
use anyhow::Result;
use tokio::fs;
use serde::{Serialize, Deserialize};

#[derive(Parser)]
#[command(name = "cli-tool")]
#[command(about = "A sample CLI tool", long_about = None)]
struct Cli {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,
}

#[derive(Serialize, Deserialize)]
struct Data {
    items: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let content = fs::read_to_string(&cli.input).await?;
    let data: Data = serde_json::from_str(&content)?;

    println!("Processing {} items", data.items.len());

    let output = serde_json::to_string_pretty(&data)?;
    fs::write(&cli.output, output).await?;

    Ok(())
}
'''
            },
            expected_trigger=True,
            is_edge_case=False
        ))

        # Backend API sample
        samples.append(TestCorpusSample(
            name="fastapi-backend",
            category="development",
            description="FastAPI backend with SQLAlchemy and authentication",
            expected_plugins=["python-development", "backend-development"],
            files={
                "requirements.txt": '''
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
sqlalchemy>=1.0.2
pydantic>=1.0.2
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
''',
                "main.py": '''
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()

# Database setup
DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/db"
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Authentication logic here
    return {"access_token": "token", "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    return {"username": "current_user"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
            },
            expected_trigger=True,
            is_edge_case=False
        ))

        return samples

    def generate_devops_samples(self) -> List[TestCorpusSample]:
        """Generate test samples for DevOps plugins."""
        samples = []

        # CI/CD sample
        samples.append(TestCorpusSample(
            name="github-actions-cicd",
            category="devops",
            description="GitHub Actions CI/CD workflow",
            expected_plugins=["cicd-automation"],
            files={
                ".github/workflows/ci.yml": '''
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests

  build:
    needs: test
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Build Docker image
      run: docker build -t myapp:${{ github.sha }} .

    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push myapp:${{ github.sha }}
''',
                "Dockerfile": '''
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
'''
            },
            expected_trigger=True,
            is_edge_case=False
        ))

        # Unit testing sample
        samples.append(TestCorpusSample(
            name="pytest-test-suite",
            category="devops",
            description="Comprehensive pytest test suite",
            expected_plugins=["unit-testing", "python-development"],
            files={
                "conftest.py": '''
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_database():
    """Mock database connection."""
    db = MagicMock()
    db.query.return_value = []
    return db

@pytest.fixture
def sample_data():
    """Sample test data."""
    return {
        "id": 1,
        "name": "Test Item",
        "value": 42
    }

@pytest.fixture(scope="session")
def app():
    """Application instance for testing."""
    from app import create_app
    app = create_app("testing")
    yield app
''',
                "tests/test_api.py": '''
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_item(sample_data):
    response = client.post("/items/", json=sample_data)
    assert response.status_code == 201
    assert response.json()["name"] == sample_data["name"]

@pytest.mark.parametrize("item_id,expected", [
    (1, 200),
    (999, 404),
    (-1, 400),
])
def test_read_item(item_id, expected):
    response = client.get(f"/items/{item_id}")
    assert response.status_code == expected

def test_database_integration(mock_database):
    # Test with mocked database
    result = mock_database.query("SELECT * FROM items")
    assert result == []
    mock_database.query.assert_called_once()
'''
            },
            expected_trigger=True,
            is_edge_case=False
        ))

        return samples

    def generate_edge_case_samples(self) -> List[TestCorpusSample]:
        """Generate edge case test samples."""
        samples = []

        # Empty project (negative test)
        samples.append(TestCorpusSample(
            name="empty-project",
            category="edge-case",
            description="Empty project directory (should not trigger)",
            expected_plugins=[],
            files={
                ".gitkeep": ""
            },
            expected_trigger=False,
            is_edge_case=True,
            is_negative_test=True
        ))

        # Mixed languages but unrelated to scientific computing
        samples.append(TestCorpusSample(
            name="web-frontend-only",
            category="edge-case",
            description="Pure frontend web project (no backend/scientific)",
            expected_plugins=["javascript-typescript", "frontend-mobile-development"],
            files={
                "index.html": '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>My Website</title>
</head>
<body>
    <h1>Hello World</h1>
    <script src="script.js"></script>
</body>
</html>
''',
                "script.js": '''
document.addEventListener('DOMContentLoaded', () => {
    console.log('Page loaded');

    const button = document.querySelector('button');
    button?.addEventListener('click', () => {
        alert('Button clicked!');
    });
});
''',
                "style.css": '''
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f0f0;
}

h1 {
    color: #333;
}
'''
            },
            expected_trigger=True,
            is_edge_case=True,
            is_negative_test=False
        ))

        # Julia file but not scientific
        samples.append(TestCorpusSample(
            name="julia-web-api",
            category="edge-case",
            description="Julia web API (not scientific computing)",
            expected_plugins=["julia-development"],
            files={
                "Project.toml": '''
name = "WebAPI"
version = "0.1.0"

[deps]
Genie = "c43c736e-a2d1-11e8-161f-af95117fbd1e"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
''',
                "routes.jl": '''
using Genie.Router
using JSON3

route("/") do
    JSON3.write(Dict("message" => "Hello from Julia API"))
end

route("/api/users", method = GET) do
    users = [
        Dict("id" => 1, "name" => "Alice"),
        Dict("id" => 2, "name" => "Bob")
    ]
    JSON3.write(users)
end

route("/api/users", method = POST) do
    payload = jsonpayload()
    # Process user creation
    JSON3.write(Dict("status" => "created", "user" => payload))
end
'''
            },
            expected_trigger=True,
            is_edge_case=True,
            is_negative_test=False
        ))

        # Configuration files only
        samples.append(TestCorpusSample(
            name="config-files-only",
            category="edge-case",
            description="Only configuration files, no code",
            expected_plugins=[],
            files={
                ".eslintrc.json": '''
{
  "extends": ["eslint:recommended"],
  "env": {
    "node": true,
    "es6": true
  },
  "rules": {
    "semi": ["error", "always"],
    "quotes": ["error", "single"]
  }
}
''',
                ".prettierrc": '''
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2
}
''',
                "tsconfig.json": '''
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "strict": true
  }
}
'''
            },
            expected_trigger=False,
            is_edge_case=True,
            is_negative_test=True
        ))

        return samples

    def generate_multi_language_samples(self) -> List[TestCorpusSample]:
        """Generate multi-language project samples."""
        samples = []

        # Python + C++ for performance
        samples.append(TestCorpusSample(
            name="python-cpp-extension",
            category="multi-language",
            description="Python with C++ extensions using pybind11",
            expected_plugins=["python-development", "systems-programming"],
            files={
                "setup.py": '''
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fast_math",
        ["src/fast_math.cpp"],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="fast_math",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
''',
                "src/fast_math.cpp": '''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

double compute_sum(py::array_t<double> arr) {
    auto buf = arr.request();
    double *ptr = static_cast<double*>(buf.ptr);
    double sum = 0.0;

    for (size_t i = 0; i < buf.size; ++i) {
        sum += ptr[i];
    }

    return sum;
}

PYBIND11_MODULE(fast_math, m) {
    m.def("compute_sum", &compute_sum, "Fast sum computation");
}
''',
                "example.py": '''
import numpy as np
import fast_math

# Create large array
arr = np.random.randn(1000000)

# Use C++ extension for fast computation
result = fast_math.compute_sum(arr)
print(f"Sum: {result}")
'''
            },
            expected_trigger=True,
            is_edge_case=False,
            is_multi_language=True
        ))

        # Julia + Python interop
        samples.append(TestCorpusSample(
            name="julia-python-workflow",
            category="multi-language",
            description="Julia and Python integration for data science",
            expected_plugins=["julia-development", "python-development"],
            files={
                "Project.toml": '''
name = "JuliaPythonWorkflow"
version = "0.1.0"

[deps]
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
''',
                "workflow.jl": '''
using PythonCall
using DataFrames
using CSV

# Import Python libraries
sklearn = pyimport("sklearn.ensemble")
pd = pyimport("pandas")

# Load data in Julia
df = CSV.read("data.csv", DataFrame)

# Convert to Python pandas
py_df = pytable(df)

# Use Python scikit-learn for modeling
model = sklearn.RandomForestClassifier(n_estimators=100)
X = py_df[["feature1", "feature2"]].values
y = py_df["target"].values

model.fit(X, y)
predictions = model.predict(X)

# Convert predictions back to Julia
julia_predictions = pyconvert(Vector, predictions)
println("Predictions: ", julia_predictions[1:10])
''',
                "python_analysis.py": '''
import pandas as pd
import matplotlib.pyplot as plt
from julia import Main

# Call Julia functions from Python
Main.include("workflow.jl")

# Load results from Julia
results = Main.eval("julia_predictions")

# Visualize in Python
plt.figure(figsize=(10, 6))
plt.hist(results, bins=50)
plt.title("Prediction Distribution")
plt.savefig("results.png")
'''
            },
            expected_trigger=True,
            is_edge_case=False,
            is_multi_language=True
        ))

        return samples

    def generate_all_samples(self) -> None:
        """Generate all test corpus samples."""
        print("Generating test corpus samples...")

        self.samples.extend(self.generate_scientific_computing_samples())
        print(f"  - Generated {len(self.samples)} scientific computing samples")

        dev_samples = self.generate_development_samples()
        self.samples.extend(dev_samples)
        print(f"  - Generated {len(dev_samples)} development samples")

        devops_samples = self.generate_devops_samples()
        self.samples.extend(devops_samples)
        print(f"  - Generated {len(devops_samples)} devops samples")

        edge_samples = self.generate_edge_case_samples()
        self.samples.extend(edge_samples)
        print(f"  - Generated {len(edge_samples)} edge case samples")

        multi_lang_samples = self.generate_multi_language_samples()
        self.samples.extend(multi_lang_samples)
        print(f"  - Generated {len(multi_lang_samples)} multi-language samples")

        print(f"\nTotal samples: {len(self.samples)}")

    def write_samples_to_disk(self) -> None:
        """Write all samples to disk."""
        print(f"\nWriting samples to {self.output_dir}...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create index file
        index_data = {
            "total_samples": len(self.samples),
            "categories": {},
            "samples": []
        }

        for sample in self.samples:
            # Create sample directory
            sample_dir = self.output_dir / sample.name
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Write all files
            for filename, content in sample.files.items():
                file_path = sample_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)

            # Write sample metadata
            metadata = {
                "name": sample.name,
                "category": sample.category,
                "description": sample.description,
                "expected_plugins": sample.expected_plugins,
                "expected_trigger": sample.expected_trigger,
                "is_edge_case": sample.is_edge_case,
                "is_negative_test": sample.is_negative_test,
                "is_multi_language": sample.is_multi_language,
                "files": list(sample.files.keys())
            }

            metadata_path = sample_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))

            # Update index
            if sample.category not in index_data["categories"]:
                index_data["categories"][sample.category] = 0
            index_data["categories"][sample.category] += 1

            index_data["samples"].append({
                "name": sample.name,
                "category": sample.category,
                "path": str(sample_dir.relative_to(self.output_dir))
            })

            print(f"  ✓ {sample.name}")

        # Write index file
        index_path = self.output_dir / "index.json"
        index_path.write_text(json.dumps(index_data, indent=2))

        print(f"\n✓ Test corpus generated successfully!")
        print(f"  Location: {self.output_dir.absolute()}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Categories: {len(index_data['categories'])}")

    def generate_readme(self) -> None:
        """Generate README for test corpus."""
        readme_content = f"""# Test Corpus for Plugin Triggering Pattern Testing

This directory contains {len(self.samples)} test samples for validating plugin activation accuracy.

## Overview

The test corpus is organized into categories representing different plugin types and use cases:

"""

        # Count by category
        category_counts = {}
        for sample in self.samples:
            category_counts[sample.category] = category_counts.get(sample.category, 0) + 1

        for category, count in sorted(category_counts.items()):
            readme_content += f"- **{category}**: {count} samples\n"

        readme_content += f"""
## Sample Types

- **Regular samples**: Typical project structures that should trigger specific plugins
- **Edge cases**: Unusual or boundary-case projects
- **Negative tests**: Projects that should NOT trigger plugins
- **Multi-language**: Projects combining multiple programming languages

## Statistics

- Total samples: {len(self.samples)}
- Edge cases: {sum(1 for s in self.samples if s.is_edge_case)}
- Negative tests: {sum(1 for s in self.samples if s.is_negative_test)}
- Multi-language: {sum(1 for s in self.samples if s.is_multi_language)}
- Expected triggers: {sum(1 for s in self.samples if s.expected_trigger)}

## Usage

Each sample directory contains:
- **metadata.json**: Sample information and expected behavior
- **Source files**: Sample code files representing the project type

Use these samples with the triggering pattern analysis tools:
- `activation-tester.py`: Test plugin activation accuracy
- `command-analyzer.py`: Test command suggestion relevance
- `skill-validator.py`: Test skill pattern matching

## Sample List

"""

        for sample in self.samples:
            readme_content += f"### {sample.name}\n"
            readme_content += f"- **Category**: {sample.category}\n"
            readme_content += f"- **Description**: {sample.description}\n"
            readme_content += f"- **Expected Plugins**: {', '.join(sample.expected_plugins) if sample.expected_plugins else 'None'}\n"
            readme_content += f"- **Should Trigger**: {'Yes' if sample.expected_trigger else 'No'}\n"

            flags = []
            if sample.is_edge_case:
                flags.append("Edge Case")
            if sample.is_negative_test:
                flags.append("Negative Test")
            if sample.is_multi_language:
                flags.append("Multi-Language")

            if flags:
                readme_content += f"- **Flags**: {', '.join(flags)}\n"

            readme_content += "\n"

        readme_path = self.output_dir / "README.md"
        readme_path.write_text(readme_content)
        print(f"  ✓ README.md generated")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test corpus for plugin triggering pattern testing"
    )
    parser.add_argument(
        "--output-dir",
        default="test-corpus",
        help="Output directory for test corpus (default: test-corpus)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["scientific-computing", "development", "devops", "edge-case", "multi-language", "all"],
        default=["all"],
        help="Categories to generate (default: all)"
    )

    args = parser.parse_args()

    generator = TestCorpusGenerator(args.output_dir)
    generator.generate_all_samples()
    generator.write_samples_to_disk()
    generator.generate_readme()

    return 0


if __name__ == "__main__":
    sys.exit(main())

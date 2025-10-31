---
name: airflow-scientific-workflows
description: Design and implement Apache Airflow DAGs for scientific data pipelines, workflow orchestration, and computational task automation. Use when creating or modifying Airflow DAG files, implementing scientific data processing pipelines, orchestrating experimental workflows, scheduling batch computations, managing time-series data ingestion, coordinating distributed simulations, integrating with databases like PostgreSQL or TimescaleDB, building ETL pipelines for scientific instruments, implementing data quality validation workflows, or automating scientific computation tasks across multiple workers.
---

# Airflow Scientific Workflows

## When to use this skill

- Creating or modifying Airflow DAG files (*.py files in airflow/dags/ directories)
- Implementing scientific data processing pipelines for experimental data
- Orchestrating multi-step scientific workflows with dependencies
- Scheduling batch computations and simulations
- Managing time-series data ingestion from scientific instruments
- Coordinating distributed scientific computations across workers
- Integrating Airflow with databases (PostgreSQL, TimescaleDB) for scientific data
- Building ETL (Extract, Transform, Load) pipelines for laboratory or sensor data
- Implementing data quality validation and gating logic in workflows
- Automating scientific computation tasks with retry logic and error handling
- Processing multi-dimensional array data in distributed workflows
- Setting up periodic analysis jobs for experimental results

**Purpose**: Apache Airflow integration patterns for scientific data pipelines and workflows

**Use Instead Of**: `database-optimizer` agent (removed in Week 2-3)

**Recommended**: Use marketplace `observability-monitoring:database-optimizer` + this skill

---

## Airflow Patterns for Scientific Computing

### 1. Time-Series Data Pipeline

**Use Case**: Process experimental time-series data with PostgreSQL storage

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

default_args = {
    'owner': 'scientific-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'scientific_timeseries_pipeline',
    default_args=default_args,
    description='Process experimental time-series data',
    schedule=timedelta(hours=6),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def extract_experimental_data(**context):
        """Extract data from instrument/sensor"""
        # Your data extraction logic
        data = pd.read_csv('/data/experiments/latest.csv')
        # Store in XCom for next task
        context['ti'].xcom_push(key='raw_data', value=data.to_json())

    def process_timeseries(**context):
        """Apply scientific processing"""
        ti = context['ti']
        raw_data = pd.read_json(ti.xcom_pull(key='raw_data'))

        # Scientific processing
        # - Noise filtering
        # - Baseline correction
        # - Peak detection
        processed = apply_scientific_processing(raw_data)

        ti.xcom_push(key='processed_data', value=processed.to_json())

    def store_to_postgres(**context):
        """Store processed data in PostgreSQL with TimescaleDB"""
        ti = context['ti']
        data = pd.read_json(ti.xcom_pull(key='processed_data'))

        pg_hook = PostgresHook(postgres_conn_id='scientific_db')
        conn = pg_hook.get_conn()

        # Use COPY for bulk insert (faster)
        data.to_sql(
            'timeseries_data',
            conn,
            if_exists='append',
            index=False,
            method='multi'
        )

    # Define task dependencies
    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_experimental_data,
    )

    process = PythonOperator(
        task_id='process_timeseries',
        python_callable=process_timeseries,
    )

    store = PythonOperator(
        task_id='store_to_postgres',
        python_callable=store_to_postgres,
    )

    extract >> process >> store
```

### 2. Array Data Processing with PostgreSQL

**Use Case**: Store multi-dimensional scientific arrays efficiently

```python
from airflow import DAG
from airflow.decorators import task
import numpy as np
from sqlalchemy import create_engine

@task
def process_array_data():
    """Process multi-dimensional experimental data"""
    # Load 3D array from experiment
    data_3d = np.load('/data/experiment_3d.npy')  # Shape: (100, 200, 50)

    # Apply scientific transformations
    fourier_transform = np.fft.fftn(data_3d)
    magnitude = np.abs(fourier_transform)

    return magnitude.tobytes(), magnitude.shape

@task
def store_array_postgres(array_bytes, shape):
    """Store array in PostgreSQL with metadata"""
    engine = create_engine('postgresql://user:pass@localhost/scientific_db')

    with engine.connect() as conn:
        # Store array as bytea with shape metadata
        conn.execute("""
            INSERT INTO array_data (experiment_id, data, shape, dtype)
            VALUES (%s, %s, %s, %s)
        """, (
            'exp_001',
            array_bytes,
            shape,
            'complex128'
        ))

with DAG('array_data_pipeline', ...) as dag:
    array_data = process_array_data()
    store_array_postgres(array_data[0], array_data[1])
```

### 3. Distributed Scientific Computation

**Use Case**: Parallelize scientific computations across workers

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
import jax
import jax.numpy as jnp

def run_simulation(simulation_id, parameters):
    """Run JAX simulation on worker"""
    # Enable GPU if available
    devices = jax.devices()

    @jax.jit
    def simulate(params):
        # Your scientific simulation
        result = complex_simulation(params)
        return result

    result = simulate(parameters)
    # Store result
    save_simulation_result(simulation_id, result)

with DAG('distributed_simulations', ...) as dag:
    # Create dynamic tasks for each simulation
    with TaskGroup('parallel_simulations') as sim_group:
        for sim_id in range(100):
            PythonOperator(
                task_id=f'simulation_{sim_id}',
                python_callable=run_simulation,
                op_kwargs={
                    'simulation_id': sim_id,
                    'parameters': get_parameters(sim_id)
                }
            )

    # Aggregate results after all simulations complete
    aggregate = PythonOperator(
        task_id='aggregate_results',
        python_callable=aggregate_simulation_results,
    )

    sim_group >> aggregate
```

### 4. Database Optimization for Scientific Data

**PostgreSQL Configuration for Scientific Workloads**:

```sql
-- Enable TimescaleDB for time-series
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create hypertable for time-series data
SELECT create_hypertable('timeseries_data', 'timestamp');

-- Optimize for array operations
CREATE EXTENSION IF NOT EXISTS cube;  -- For multi-dimensional data

-- Create efficient indexes
CREATE INDEX idx_experiment_time ON timeseries_data (experiment_id, timestamp DESC);
CREATE INDEX idx_array_metadata ON array_data USING GIN (metadata);

-- Set performance parameters
ALTER SYSTEM SET shared_buffers = '4GB';  -- For large datasets
ALTER SYSTEM SET work_mem = '256MB';      -- For complex queries
ALTER SYSTEM SET effective_cache_size = '12GB';
```

**Airflow Task for Database Optimization**:

```python
@task
def optimize_scientific_db():
    """Run database optimization for scientific queries"""
    pg_hook = PostgresHook(postgres_conn_id='scientific_db')

    # Analyze tables for query planner
    pg_hook.run("ANALYZE timeseries_data;")
    pg_hook.run("ANALYZE array_data;")

    # Vacuum for performance
    pg_hook.run("VACUUM ANALYZE;")

    # Reindex if needed
    pg_hook.run("REINDEX TABLE timeseries_data;")
```

### 5. Data Quality Checks for Scientific Pipelines

```python
from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.email import EmailOperator

@task
def validate_experimental_data(data_path):
    """Validate data quality for scientific standards"""
    data = load_data(data_path)

    checks = {
        'missing_values': check_missing_values(data),
        'outliers': check_outliers(data, n_sigma=5),
        'calibration': check_calibration(data),
        'metadata': validate_metadata(data),
    }

    if all(checks.values()):
        return 'data_valid'
    else:
        return 'data_invalid'

def branch_on_validation(**context):
    """Branch based on validation result"""
    result = context['ti'].xcom_pull(task_ids='validate_data')
    return 'process_data' if result == 'data_valid' else 'alert_team'

with DAG('data_quality_pipeline', ...) as dag:
    validate = validate_experimental_data('/data/latest')

    branch = BranchPythonOperator(
        task_id='check_quality',
        python_callable=branch_on_validation,
    )

    process = PythonOperator(
        task_id='process_data',
        python_callable=process_valid_data,
    )

    alert = EmailOperator(
        task_id='alert_team',
        to=['team@example.com'],
        subject='Data Quality Alert',
        html_content='Invalid experimental data detected',
    )

    validate >> branch >> [process, alert]
```

---

## Integration Patterns

### With PostgreSQL (Use Marketplace Database Optimizer)

For database optimization, use the marketplace plugin:
```python
# Use observability-monitoring:database-optimizer for:
# - Query optimization
# - Index recommendations
# - Performance tuning

# Use this skill for:
# - Airflow DAG patterns
# - Scientific data workflows
# - Pipeline orchestration
```

### With JAX (Use Custom jax-pro Agent)

```python
# Combine Airflow orchestration with JAX computation
@task
def jax_computation_task(data):
    """Leverage custom jax-pro agent for optimization"""
    # This task can invoke jax-pro agent for:
    # - JIT compilation advice
    # - GPU optimization
    # - Gradient computation

    import jax
    @jax.jit
    def compute(x):
        return x ** 2

    return compute(data)
```

---

## Best Practices

1. **Use PostgresHook** for database connections (connection pooling)
2. **Parallelize independent tasks** with TaskGroups
3. **Store large arrays as bytea** with metadata for shape/dtype
4. **Use TimescaleDB** for time-series data
5. **Implement data quality checks** before processing
6. **Enable retry logic** for scientific computations
7. **Monitor DAG performance** with Airflow metrics
8. **Use XCom sparingly** (store large data in shared storage)

---

## Common Scientific Workflow Patterns

| Pattern | Use Case | Airflow Features |
|---------|----------|------------------|
| **ETL Pipeline** | Experimental data processing | PythonOperator, PostgresOperator |
| **Distributed Compute** | Parallel simulations | TaskGroup, Dynamic tasks |
| **Time-Series** | Sensor data ingestion | TimescaleDB, continuous aggregates |
| **Array Processing** | Multi-dimensional data | PostgreSQL bytea, JAX |
| **Data Quality** | Validation gates | BranchOperator, Sensors |
| **Batch ML** | Model training pipelines | SubDagOperator, KubernetesPodOperator |

---

## Example: Complete Scientific Pipeline

```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta

@task
def extract_sensor_data():
    """Extract from scientific instruments"""
    return read_sensor('/dev/instrument0')

@task
def calibrate_data(raw_data):
    """Apply calibration curves"""
    return apply_calibration(raw_data)

@task
def detect_peaks(calibrated_data):
    """Scientific peak detection"""
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(calibrated_data, height=threshold)
    return peaks

@task
def fit_model(peaks_data):
    """Fit theoretical model"""
    from scipy.optimize import curve_fit
    params, cov = curve_fit(model_function, peaks_data)
    return params

@task
def store_results(params):
    """Store in PostgreSQL"""
    pg_hook = PostgresHook('scientific_db')
    pg_hook.run("""
        INSERT INTO experiment_results (timestamp, parameters)
        VALUES (NOW(), %s)
    """, parameters=(params,))

with DAG(
    'scientific_experiment_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule=timedelta(hours=1),
    catchup=False,
) as dag:

    # Define workflow
    raw = extract_sensor_data()
    calibrated = calibrate_data(raw)
    peaks = detect_peaks(calibrated)
    params = fit_model(peaks)
    store_results(params)
```

---

**Replaced Agent**: `database-optimizer`
**For Database Optimization**: Use `observability-monitoring:database-optimizer`
**For Airflow Patterns**: Use this skill
**Maintenance**: Update for Airflow version changes

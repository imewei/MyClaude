---
name: airflow-scientific-workflows
version: "1.0.7"
maturity: "5-Expert"
specialization: Scientific Pipeline Orchestration
description: Design Apache Airflow DAGs for scientific data pipelines, batch computations, distributed simulations, and time-series data ingestion with PostgreSQL/TimescaleDB integration. Use when orchestrating experimental workflows or coordinating scientific computations.
---

# Airflow Scientific Workflows

Apache Airflow patterns for scientific data pipelines and computation orchestration.

---

## Pattern Selection

| Pattern | Use Case | Airflow Features |
|---------|----------|------------------|
| ETL Pipeline | Experimental data processing | PythonOperator, PostgresOperator |
| Distributed Compute | Parallel simulations | TaskGroup, dynamic tasks |
| Time-Series | Sensor data ingestion | TimescaleDB, continuous aggregates |
| Array Processing | Multi-dimensional data | PostgreSQL bytea, JAX |
| Data Quality | Validation gates | BranchOperator, Sensors |
| Batch ML | Model training | SubDagOperator, KubeOperator |

---

## Time-Series Pipeline

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta

default_args = {
    'owner': 'scientific-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG('timeseries_pipeline', default_args=default_args,
         schedule=timedelta(hours=6), start_date=datetime(2025, 1, 1),
         catchup=False) as dag:

    def extract(**context):
        data = pd.read_csv('/data/experiments/latest.csv')
        context['ti'].xcom_push(key='raw_data', value=data.to_json())

    def process(**context):
        raw = pd.read_json(context['ti'].xcom_pull(key='raw_data'))
        processed = apply_scientific_processing(raw)
        context['ti'].xcom_push(key='processed', value=processed.to_json())

    def store(**context):
        data = pd.read_json(context['ti'].xcom_pull(key='processed'))
        hook = PostgresHook(postgres_conn_id='scientific_db')
        data.to_sql('timeseries_data', hook.get_conn(), if_exists='append')

    extract_task = PythonOperator(task_id='extract', python_callable=extract)
    process_task = PythonOperator(task_id='process', python_callable=process)
    store_task = PythonOperator(task_id='store', python_callable=store)

    extract_task >> process_task >> store_task
```

---

## Distributed Simulations

```python
from airflow.utils.task_group import TaskGroup

with DAG('distributed_simulations', ...) as dag:
    with TaskGroup('parallel_simulations') as sim_group:
        for sim_id in range(100):
            PythonOperator(
                task_id=f'simulation_{sim_id}',
                python_callable=run_simulation,
                op_kwargs={'simulation_id': sim_id, 'params': get_params(sim_id)}
            )

    aggregate = PythonOperator(
        task_id='aggregate_results',
        python_callable=aggregate_simulation_results
    )

    sim_group >> aggregate
```

---

## Data Quality Validation

```python
from airflow.operators.python import BranchPythonOperator

@task
def validate_data(data_path):
    data = load_data(data_path)
    checks = {
        'missing_values': check_missing_values(data),
        'outliers': check_outliers(data, n_sigma=5),
        'calibration': check_calibration(data)
    }
    return 'process_data' if all(checks.values()) else 'alert_team'

def branch_on_validation(**context):
    return context['ti'].xcom_pull(task_ids='validate_data')

with DAG('data_quality_pipeline', ...) as dag:
    validate = validate_data('/data/latest')
    branch = BranchPythonOperator(task_id='check', python_callable=branch_on_validation)
    process = PythonOperator(task_id='process_data', python_callable=process_valid_data)
    alert = EmailOperator(task_id='alert_team', to=['team@example.com'],
                          subject='Data Quality Alert')

    validate >> branch >> [process, alert]
```

---

## TimescaleDB Integration

```sql
-- Enable TimescaleDB for time-series
CREATE EXTENSION IF NOT EXISTS timescaledb;
SELECT create_hypertable('timeseries_data', 'timestamp');

-- Indexes for scientific queries
CREATE INDEX idx_experiment_time ON timeseries_data (experiment_id, timestamp DESC);

-- Performance settings
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET work_mem = '256MB';
```

```python
@task
def optimize_scientific_db():
    hook = PostgresHook(postgres_conn_id='scientific_db')
    hook.run("ANALYZE timeseries_data;")
    hook.run("VACUUM ANALYZE;")
```

---

## Array Data Storage

```python
from airflow.decorators import task
import numpy as np

@task
def process_array_data():
    data_3d = np.load('/data/experiment_3d.npy')
    magnitude = np.abs(np.fft.fftn(data_3d))
    return magnitude.tobytes(), magnitude.shape

@task
def store_array(array_bytes, shape):
    engine = create_engine('postgresql://user:pass@localhost/db')
    with engine.connect() as conn:
        conn.execute("""
            INSERT INTO array_data (experiment_id, data, shape, dtype)
            VALUES (%s, %s, %s, %s)
        """, ('exp_001', array_bytes, shape, 'float64'))
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| PostgresHook | Connection pooling, reuse |
| TaskGroups | Parallelize independent tasks |
| bytea for arrays | Store with shape/dtype metadata |
| TimescaleDB | Time-series optimization |
| Data validation | Before processing |
| XCom sparingly | Large data in shared storage |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Large XCom payloads | Use shared storage (S3, NFS) |
| No retry logic | Set retries in default_args |
| Sequential when parallel OK | Use TaskGroup |
| Missing dependencies | Use proper task ordering |
| No data validation | Add BranchOperator gates |

---

## Checklist

- [ ] DAG default_args configured (retries, owner)
- [ ] Task dependencies correctly ordered
- [ ] Data validation gates implemented
- [ ] PostgreSQL connections use hooks
- [ ] Large data stored externally (not XCom)
- [ ] Parallel tasks use TaskGroup

---

**Version**: 1.0.5

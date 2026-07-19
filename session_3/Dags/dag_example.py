# dags/retrain_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    dag_id="ride_duration_retrain",
    schedule="@weekly",              # every Monday 00:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,                   # don't backfill missed runs
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "owner": "ml-team",
    },
)

def extract(**ctx):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("mlops-ride-duration")
    bucket.blob("rides/latest.parquet").download_to_filename("data/raw/rides.parquet")

def train(**ctx):
    import mlflow, subprocess
    with mlflow.start_run() as run:
        subprocess.run(["python", "src/train.py"], check=True)
        ctx["ti"].xcom_push(key="run_id", value=run.info.run_id)

def evaluate(**ctx):
    run_id = ctx["ti"].xcom_pull(key="run_id", task_ids="train")
    # compare MAE against current Production model

t_extract  = PythonOperator(task_id="extract",  python_callable=extract,  dag=dag)
t_train    = PythonOperator(task_id="train",    python_callable=train,    dag=dag)
t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate, dag=dag)

t_extract >> t_train >> t_evaluate

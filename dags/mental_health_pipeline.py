import sys
from datetime import datetime, timedelta

from airflow.decorators import dag, task

sys.path.insert(0, '/opt/airflow')

from src.data_preparation import prepare_data
from src.train import train_model
from src.evaluate import evaluate_model


@dag(
    dag_id='mental_health_pipeline',
    start_date=datetime(2026, 11, 1),
    schedule='@weekly',
    catchup=False,
    default_args={'retries': 1, 'retry_delay': timedelta(minutes=5)},
    tags=['mental-health'],
)
def mental_health_pipeline():

    @task
    def prepare():
        prepare_data()
        return 'data/processed'

    @task
    def train(data_path):
        train_model()
        return 'models/pipeline.joblib'

    @task
    def evaluate(model_path):
        evaluate_model()

    data_path = prepare()
    model_path = train(data_path)
    evaluate(model_path)


mental_health_pipeline()
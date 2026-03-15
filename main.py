import yaml
import joblib
import argparse
import time
import mlflow

from src.utils.logger import get_logger
from src.data.preprocess import load_data, preprocess
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate
from src.models.predict_model import load_model, predict

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ADAS_Phantom_Braking")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/config.yaml"
)
args = parser.parse_args()


def main():
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger = get_logger(config["paths"]["log_file"])

    try:
        logger.info("Pipeline started")
        mode = config["mode"]
        # start timer
        start_time = time.time()
        if mode == "train":
            logger.info("Running training pipeline")
            df = load_data(config["paths"]["raw_data"])
            X, y = preprocess(df, config["data"]["target_column"])
            mlflow.set_tracking_uri("http://127.0.0.1:5000")

            with mlflow.start_run():
                # log configuration parameters
                mlflow.log_param("model_type", config["model"]["selected_model"])
                mlflow.log_param("test_size", config["data"]["test_size"])
                mlflow.log_param("random_state", config["data"]["random_state"])
                model, X_test, y_test = train_model(X, y, config)
                metrics = evaluate(model, X_test, y_test)
                logger.info(f"Model metrics: {metrics}")
                # log metrics to MLflow
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                # log model artifact
                mlflow.sklearn.log_model(model, "model")
                joblib.dump(model, config["paths"]["model_output"])
                logger.info("Model saved successfully")

        elif mode == "test":
            logger.info("Running prediction")
            model = load_model(config["paths"]["model_output"])
            sample = config["input_features"]
            prob = predict(model, sample)
            logger.info(f"Prediction probability: {prob}")
        end_time = time.time()
        logger.info(f"Total pipeline time: {round(end_time - start_time, 2)} seconds")
        logger.info("Pipeline finished")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
import yaml
import joblib
import argparse
import time

from src.utils.logger import get_logger
from src.data.preprocess import load_data, preprocess
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate
from src.models.predict_model import load_model, predict


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
            if mode == "train":
                logger.info("Running training pipeline")
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logger.info(f"Pipeline started at: {start_time}")
                df = load_data(config["paths"]["raw_data"])
                X, y = preprocess(df, config["data"]["target_column"])
                model, X_test, y_test = train_model(X, y, config)
                metrics = evaluate(model, X_test, y_test)
                logger.info(f"Model metrics: {metrics}")
                joblib.dump(model, config["paths"]["model_output"])
                logger.info("Model saved successfully")
                end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logger.info(f"Pipeline finished successfully at: {end_time}")
                logger.info(f"Total pipeline time: {end_time - start_time}")

            elif mode == "test":
                logger.info("Running prediction")
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logger.info(f"Pipeline started at: {start_time}")
                model = load_model(config["paths"]["model_output"])
                sample = config["input_features"]
                prob = predict(model, sample)
                logger.info(f"Prediction probability: {prob}")
                end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logger.info(f"Pipeline finished successfully at: {end_time}")
                logger.info(f"Total pipeline time: {end_time - start_time}")


        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
if __name__ == "__main__":
    main()
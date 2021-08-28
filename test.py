import argparse
from os import makedirs
from os.path import isdir, join
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from time import time

from utils import read_yaml, read_object, visualize_classification_2d, write_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument("--config", type=str, default="params.yaml",
                        help="Absolute path to the configuration file (params.yaml)")
    return parser.parse_args()


def run_testing(args):
    config = read_yaml(args.config)["testing"]
    print(f"Running test according settings from {args.config}:\n{config}\n")
    report = {
        "config": config
    }

    test_df = pd.read_csv(config["dataset"])
    x_test = test_df.loc[:, test_df.columns != "label"].to_numpy()
    y_test = test_df.loc[:, test_df.columns == "label"].to_numpy()
    print(f"Test dataset contains {x_test.shape[0]} samples")

    preprocessor_path = join(config["model"], "preprocessor.pickle")
    preprocessor = read_object(preprocessor_path)
    print(f"Preprocessor '{preprocessor}' has been loaded from {preprocessor_path}")
    model_path = join(config["model"], "model.pickle")
    model = read_object(model_path)
    print(f"Model '{model}' has been loaded from {model_path}")
    x_test_norm = preprocessor.transform(x_test)

    start = time()
    y_pred = model.predict(x_test_norm)
    prediction_time = time() - start
    report["prediction_time"] = prediction_time
    print(f"Testing completed. Prediction time (full test dataset): {prediction_time} seconds")

    report["quality"] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }
    print(f"Quality on the test dataset:\n{report['quality']}\n")

    if not isdir(config["output"]):
        makedirs(config["output"])
    report_path = join(config["output"], "test_report.yml")
    write_yaml(report_path, report)
    print(f"Test report has been saved to {report_path}")
    feature_names = test_df.columns[:-1].values.tolist()
    visualization_path = join(config["output"], "visualization_test.png")
    visualize_classification_2d(x_test_norm, y_test.flatten(), model, path=visualization_path, hint="Test subset", feature_names=feature_names)
    print(f"Prediction visualization has been saved to {visualization_path}")


if __name__ == "__main__":
    run_testing(parse_args())

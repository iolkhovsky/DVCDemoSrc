import argparse
from os import makedirs
from os.path import isdir, join
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
import sklearn.neural_network
from time import time

from utils import read_yaml, save_object, visualize_classification_2d, write_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, default="params.yaml",
                        help="Absolute path to the configuration file (params.yaml)")
    return parser.parse_args()


def run_training(args):
    config = read_yaml(args.config)["training"]
    print(f"Running model training. Training configuration:\n{config}\n")
    report = {
        "config": config
    }

    train_df = pd.read_csv(config["dataset"])
    x_train = train_df.loc[:, train_df.columns != "label"].to_numpy()
    y_train = train_df.loc[:, train_df.columns == "label"].to_numpy()
    print(f"Training dataset size: {x_train.shape[0]} samples")

    preprocessor = getattr(sklearn.preprocessing, config["preprocessor"]["type"])()
    print(f"Preprocessor (normalization): {preprocessor}")
    preprocessor.fit(x_train)
    x_train_norm = preprocessor.transform(x_train)

    model_pars = config["model"]["parameters"]
    if model_pars != "None":
        model = getattr(getattr(sklearn, config["model"]["method"]), config["model"]["type"])(**model_pars)
    else:
        model = getattr(getattr(sklearn, config["model"]["method"]), config["model"]["type"])()
    print(f"Model: {model}\n")

    start = time()
    model.fit(x_train_norm, y_train.flatten())
    training_time = time() - start
    report["training_time"] = training_time
    print(f"Training completed. Duration: {training_time} seconds")

    start = time()
    y_pred = model.predict(x_train_norm)
    prediction_time = time() - start
    report["prediction_time"] = prediction_time
    print(f"Prediction total (full training dataset) time: {prediction_time} seconds")

    report["quality"] = {
        "accuracy": float(accuracy_score(y_train, y_pred)),
        "precision": float(precision_score(y_train, y_pred)),
        "recall": float(recall_score(y_train, y_pred)),
        "f1": float(f1_score(y_train, y_pred))
    }
    print(f"Quality metrics on training dataset:\n{report['quality']}\n")

    if not isdir(config["output"]):
        makedirs(config["output"])
    model_path = join(config["output"], "model.pickle")
    save_object(model, model_path)
    print(f"Trained model has been saved to '{model_path}'")
    preprocessor_path = join(config["output"], "preprocessor.pickle")
    save_object(preprocessor, preprocessor_path)
    print(f"Data preprocessor has been saved to '{preprocessor_path}'")
    report_path = join(config["output"], "train_report.yml") 
    write_yaml(report_path, report)
    print(f"Training report has been saved to '{report_path}'")
    feature_names = train_df.columns[:-1].values.tolist()
    visualization_path = join(config["output"], "visualization_train.png")
    visualize_classification_2d(x_train_norm, y_train.flatten(), model, path=visualization_path, hint="Train subset", feature_names=feature_names)
    print(f"Prediction visualization has been saved to {visualization_path}")


if __name__ == "__main__":
    run_training(parse_args())

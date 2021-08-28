import argparse
import numpy as np
from os import makedirs
from os.path import isdir, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from utils import read_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument("--config", type=str, default="params.yaml",
                        help="Absolute path to the configuration file (params.yaml)")
    return parser.parse_args()


def preprocess_data(args):
    print(f"Running data preprocessing according settings from {args.config}:\n")
    config = read_yaml(args.config)["preprocessing"]
    print(config, "\n")
    output = config["output"]
    data = config["data"]
    test_share = config["test_share"]
    random_state = config["random_state"]

    if not isdir(output):
        makedirs(output)

    df = pd.read_csv(data, index_col=False)
    features = df.loc[:, df.columns != "label"].to_numpy()
    labels = df.loc[:, df.columns == "label"].to_numpy()
    print(f"Dataset includes {features.shape[0]} samples total.")
    print(f"Features: {df.columns[:-1].values.tolist()}")

    target_features_cnt = 2
    norm_features = StandardScaler().fit_transform(features, labels.flatten())
    feature_selector = SelectKBest(f_classif, k=target_features_cnt)
    feature_selector.fit(norm_features, labels.flatten())
    best_features_mask = feature_selector.get_support()
    best_features_labels = df.columns[:-1][best_features_mask].values.tolist()
    print(f"Best {target_features_cnt} features: {best_features_labels}\n")
    best_features = df[best_features_labels].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        best_features, labels, test_size=test_share, random_state=random_state
    )
    
    print("Preprocessed dataset:\n")
    print(f"Train subset: {x_train.shape[0]} samples")
    train_dataset_path = join(output, "train.csv")
    train_df = pd.DataFrame(data=np.hstack([x_train, y_train]), columns=best_features_labels + ["label"])
    train_df.to_csv(train_dataset_path, index=False)
    print(f"Train dataset has been saved to {train_dataset_path}")
    print(train_df.head(), "\n")
    
    print(f"Test subset: {x_test.shape[0]} samples")
    test_dataset_path = join(output, "test.csv")
    test_df = pd.DataFrame(data=np.hstack([x_test, y_test]), columns=best_features_labels + ["label"])
    test_df.to_csv(test_dataset_path, index=False)
    print(f"Test dataset has been saved to {test_dataset_path}")
    print(test_df.head(), "\n")


if __name__ == "__main__":
    preprocess_data(parse_args())

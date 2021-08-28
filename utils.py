import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from os.path import dirname, isfile, isdir
from os import makedirs
import pickle
import seaborn as sns
import yaml


CMAP_LIGHT = ListedColormap(['lightyellow', 'cyan'])
CMAP_BOLD = ['orange', 'darkblue']


def read_yaml(path):
    assert isfile(path), f"{path} doesnt exist"
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        return data


def write_yaml(path, data):
    parent_folder = dirname(path)
    if len(parent_folder) and not isdir(parent_folder):
        makedirs(parent_folder)
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)


def save_object(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_object(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
        return obj


def visualize_classification_2d(x, y, model, path=None, grid_step=None, feature_names=None, hint=""):
    if grid_step is None:
        grid_step = 0.02
    if feature_names is None:
        feature_names = ["feature_#0", "feature_#1"]

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    z = z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, cmap=CMAP_LIGHT)

    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y,
                    palette=CMAP_BOLD, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"Binary classification. Model: {model}. {hint}")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

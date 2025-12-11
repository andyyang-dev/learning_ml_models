from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def show_linear_data(df: pd.DataFrame, parameter: Tuple[float, float], hide_line: bool = False) -> None:
    b0, b1 = parameter
    _, ax = plt.subplots()
    if not hide_line:
        x_line = np.linspace(0, 1, 100)
        y_line = b1*x_line + b0
        ax.plot(x_line, y_line)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.scatter(df["Intelligence"], df["Mating Probability"], marker="x")
    ax.set_title("Mating Possibility in terms of Intelligence from Sarah's Perspective", pad=20)
    ax.set_xlabel("Intelligence")
    ax.set_ylabel("Mating Posibility")
    plt.tight_layout()
    plt.show()

def show_logistic_data(df: pd.DataFrame, parameter: np.ndarray) -> None:
    x_line = np.linspace(0, 1, 100)
    thetaTx = parameter[1]*x_line + parameter[0]
    y_line = 1 / (1 + math.e ** (-thetaTx))

    fig, ax = plt.subplots()
    ax.scatter(df["Attractiveness"], df["Mating Probability"])
    ax.plot(x_line, y_line)
    ax.plot(x_line, [0.5] * len(x_line))
    plt.show()

def show_svm_data(df, w: Optional[np.ndarray] = None, b: Optional[float]= None):
    _, ax = plt.subplots(figsize=(15, 6))
    df_ideal = df[df["Class"] == 1]
    df_not_ideal = df[df["Class"] == -1]
    if w is not None:
        w1, w2 = w
        x1_line = np.linspace(df_ideal["Normalized Height (cm)"].min() * 2, df_ideal["Normalized Height (cm)"].max() * 2, 100)
        x2_line = (w1 * x1_line + b) / (-w2)
        ax.plot(x1_line, x2_line)
    ax.scatter(x=df_ideal["Normalized Height (cm)"], y=df_ideal["Transformed X"], label="Ideal", marker='o')
    ax.scatter(x=df_not_ideal["Normalized Height (cm)"], y=df_not_ideal["Transformed X"], label="Not Ideal", marker='x')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    plt.show()

def show_knn_kmeans_data(df: pd.DataFrame, center_clusters: Optional[np.ndarray] = None):
    _, ax = plt.subplots()
    if center_clusters is not None:
        for cluster in center_clusters:
            ax.scatter(cluster[0], cluster[1], s=200)
    for career in df["Career"].unique():
        df_one_career = df[df["Career"] == career]
        ax.scatter(df_one_career["Attractiveness Scale"], df_one_career["Creativity Scale"], label=career)
    ax.set_title("Attractiveness and Creativity Role on Woman's Career")
    ax.set_xlabel("Attractiveness Scale")
    ax.set_ylabel("Creative Scale")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.legend()
    plt.show()
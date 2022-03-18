from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_outlier_scores(
    y_true: List[int], scores: List[float], title: str = "", **kdeplot_options
) -> pd.DataFrame:
    """
    Plots the distribution of scores conditional on the real labels in y_true,
    and returns a DataFrame with classification results for further analysis.

    y_true: the actual labels (0/1)
    scores: the computed outlier scores (the higher the score,
            the higher the probability of an outlier).
    **kdeplot_options (such as bw for kde kernel width) are passed to
            sns.kdeplot()

    Returns a pandas.DataFrame with classification results
    """
    if len(y_true) != len(scores):
        msg = "Error: Expecting y_true and scores to be 1-D and equal-length"
        raise ValueError(msg)
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(scores, pd.Series):
        scores = scores.values

    aucroc_score = roc_auc_score(y_true, scores)
    aucpr_score = average_precision_score(y_true, scores)

    classify_results = pd.DataFrame(
        data=pd.concat((pd.Series(y_true), pd.Series(scores)), axis=1)
    )

    classify_results.rename(columns={0: "true", 1: "score"}, inplace=True)
    sns.kdeplot(
        classify_results.loc[classify_results.true == 0, "score"],
        label="negatives",
        shade=True,
        **kdeplot_options,
    )

    sns.kdeplot(
        classify_results.loc[classify_results.true == 1, "score"],
        label="positives",
        shade=True,
        **kdeplot_options,
    )

    plt.title(
        "{} AUC-ROC: {:.3f}, AUC-PR: {:.3f}".format(title, aucroc_score, aucpr_score)
    )

    plt.title("{} AUC-ROC: {:.3f}".format(title, aucroc_score))
    plt.xlabel("Predicted outlier score")
    plt.legend()
    return classify_results


def plot_roc_averageprecision_curves(y_true: List[int], scores: List[float]):
    LW = 2  # Linewidth of ROC and APR curves

    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, recall, _ = precision_recall_curve(y_true, scores, pos_label=1)

    roc_auc = auc(fpr, tpr)
    ap_auc = average_precision_score(y_true, scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(fpr, tpr, color="darkorange", lw=LW, label="ROC curve")
    ax1.plot([0, 1], [0, 1], color="navy", lw=LW, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=14)
    ax1.set_ylabel("True Positive Rate", fontsize=14)
    ax1.set_title(f"ROC curve (AUC = {roc_auc:.2f})", fontsize=15)
    ax1.legend(loc="lower right")

    ax2.plot(recall, prec, color="darkorange", lw=LW, label="Precision-recall curve")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("True Positive Rate (Recall)", fontsize=14)
    ax2.set_ylabel("Precision", fontsize=14)
    ax2.set_title(f"AP curve (AUC = {ap_auc:.2f})", fontsize=15)
    ax2.legend(loc="upper right")


def plot_top_N(y_true: List[int], scores: List[float], N: int = 100) -> pd.DataFrame:
    """
    Plots the actual binary labels (Positive versus Negative) of the N points
    with the highest outlier scores.

    y_true are the actual labels (0/1)
    scores are the computed outlier scores (the higher the score,
    the higher the probability of an outlier).
    N: number of points with highest outlier scores.

    Returns: a pandas DataFrame with classification results
    """
    assert len(y_true) == len(scores), (
        "Error: " "Expecting y_true and scores to be 1-D and of equal length"
    )
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(scores, pd.Series):
        scores = scores.values
    N = min(N, len(scores))
    classify_results = pd.DataFrame(
        data=pd.concat((pd.Series(y_true), pd.Series(scores)), axis=1)
    )
    classify_results.rename(columns={0: "true", 1: "score"}, inplace=True)
    classify_results = classify_results.sort_values(by="score", ascending=False)[:N]
    Npos_in_N = classify_results["true"].sum()

    fig, ax = plt.subplots(1, 1, figsize=(16, 2))
    ims = ax.imshow(
        np.reshape(classify_results.true.values, [1, -1]),
        extent=[-0.5, N, N / 50, -0.5],
        vmin=0,
        vmax=1,
    )
    ax.yaxis.set_visible(False)
    # ax.xaxis.set_ticklabels
    plt.colorbar(ims)
    plt.xlabel("Outlier rank [-]")
    plt.title(
        f"Yellow: positive, Purple:Negative. Number of positives found: "
        f"{Npos_in_N} (P@Rank{N}: {Npos_in_N/N:.1%})"
    )
    return classify_results

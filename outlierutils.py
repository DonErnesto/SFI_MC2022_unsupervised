import functools
import json
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (roc_auc_score,
                            roc_curve,
                            precision_recall_curve,
                            auc,
                            average_precision_score
                            )


def plot_outlier_scores(
    y_true: List[int], scores: List[float], title: str = "", **kdeplot_options
) -> pd.DataFrame:
    """
    Plots the distribution of scores conditional on the real labels in y_true,
    and returns a DataFrame with classification results for further analysis.

    y_true: the actual labels (0/1)
    scores: the computed outlier scores (the higher the score, the higher the probability of an outlier).
    **kdeplot_options (such as bw for kde kernel width) are passed to sns.kdeplot()

    Returns a pandas.DataFrame with classification results
    """
    if len(y_true) != len(scores):
        msg = "Error: " "Expecting y_true and scores to be 1-D and of equal length"
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


def plot_roc_averageprecision_curves(y_true: List[int], y_pred: List[float]):
    fpr, tpr, _ = roc_curve(y, homemade_outlier_scores)
    prec, recall, _ = precision_recall_curve(y, homemade_outlier_scores, pos_label=1)

    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y, homemade_outlier_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve")
    ax1.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=14)
    ax1.set_ylabel("True Positive Rate", fontsize=14)
    ax1.set_title(f"ROC curve (AUC = {roc_auc:.2f})", fontsize=15)
    ax1.legend(loc="lower right")

    ax2.plot(
        recall,
        prec,
        color="darkorange",
        lw=lw,
        label="Precision-recall curve")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("True Positive Rate (Recall)", fontsize=14)
    ax2.set_ylabel("Precision", fontsize=14)
    ax2.set_title(f"AP curve (AUC = {ap:.2f})", fontsize=15)
    ax2.legend(loc="lower right")
    plt.display()



def plot_top_N(y_true: List[int], scores: List[float], N: int = 100
) -> pd.DataFrame:
    """
    Plots the actual binary labels (Positive versus Negative) of the N points
    with the highest outlier scores.

    y_true are the actual labels (0/1)
    scores are the computed outlier scores (the higher the score, the higher the probability of an outlier).
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
        f"Yellow: positive, Purple:Negative. Number of positives found: {Npos_in_N} (P@Rank{N}: {Npos_in_N/N:.1%})"
    )
    # plt.show()
    return classify_results


def median_imputation(
    df: pd.DataFrame, median_impute_limit: float = 0.95, impute_val: int = -999
) -> pd.DataFrame:
    """inf/nan Values that occur more often than median_impute_limit are imputed with the median
    when less often, they are imputed by impute_val.
    Set median_impute_limit to 0 to always do median imputation
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if not df[col].dtype == "object":
            mean_nan = df[col].isna().mean()
            if mean_nan > median_impute_limit:  # then, impute by median
                df[col] = df[col].fillna(df[col].median())
            elif mean_nan > 0 and mean_nan <= median_impute_limit:
                df[col] = df[col].fillna(impute_val)

    return df


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """
    Converts the numeric data types in dataframe df to the smallest possible representation.
    Function taken from Kaggle.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                # if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                #    df[col] = df[col].astype(np.int8)
                # elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                #    df[col] = df[col].astype(np.int16)
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased from {:5.2f} to {:5.2f} Mb ({:.1f}% reduction)".format(
                start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

"""Module providing plotting functionality."""

import os

import metallurgy as mg
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)
import numpy as np
from adjustText import adjust_text
from scipy.cluster import hierarchy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import cerebral as cb

plt.rc("axes", axisbelow=True)


def plot_training(history):
    """Plot metrics over the course of training.

    :group: plots

    """

    image_directory = None
    if cb.conf.save:
        if not os.path.exists(cb.conf.image_directory):
            os.makedirs(cb.conf.image_directory)

        image_directory = cb.conf.image_directory

    for metric in history.history:

        if "val_" not in metric:
            plt.plot(history.history[metric], "-b", label="Train")

            if "val_" + metric in history.history:
                plt.plot(history.history["val_" + metric], "-r", label="Test")
                plt.legend(loc="best")

            # plt.grid(alpha=.4)
            plt.xlabel("Epochs")
            if (
                "loss" in metric
                or metric == "lr"
                or "MSE" in metric
                or "MAE" in metric
                or "Huber" in metric
            ):
                plt.yscale("log")

            if "GFA_" in metric and "_loss" not in metric:
                plt.ylim(0, 1)

            if len(cb.conf.targets) == 1:
                metric = cb.conf.targets[0].name + "_" + metric

            plt.ylabel(metric)

            plt.tight_layout()

            if cb.conf.save:
                if "_" in metric:
                    feature = metric.split("_")[0]
                    if not os.path.exists(image_directory + feature):
                        os.makedirs(image_directory + feature)
                    plt.savefig(
                        image_directory + feature + "/" + metric + ".png"
                    )
                else:
                    plt.savefig(image_directory + metric + ".png")
            else:
                plt.show()

            plt.cla()
            plt.clf()
            plt.close()


def write_errors(compositions, labels, predictions, suffix=None):
    for target_index, target in enumerate(cb.conf.targets):
        if target["type"] != "numerical":
            continue

        target_name = target["name"]

        results_file_path = (
            cb.conf.output_directory + "/" + target_name + "_error"
        )
        if suffix is not None:
            results_file_path += "_" + suffix
        results_file_path += ".dat"

        with open(results_file_path, "w") as results_file:
            for j in range(len(compositions)):
                results_file.write(
                    compositions.iloc[j].to_string()
                    + " "
                    + str(labels[target["name"]].iloc[j])
                    + " "
                    + str(predictions[target_index][j])
                    + " "
                    + str(
                        predictions[target_index][j]
                        - labels[target["name"]].iloc[j]
                    )
                    + "\n"
                )


def gather_labelled_errors(labels, predictions, compositions, errors):
    labelled_errors = []
    errors = list(errors)
    top_errors = sorted(errors, reverse=True)[:3]
    for e in top_errors:
        index = errors.index(e)
        labelled_errors.append(
            [
                labels[index],
                predictions[index],
                compositions[index],
            ]
        )
    return labelled_errors


def plot_results_regression(
    train_labels,
    train_predictions,
    train_errors,
    test_labels=None,
    test_predictions=None,
    test_errors=None,
    train_compositions=None,
    test_compositions=None,
    train_errorbars=None,
    test_errorbars=None,
    metrics=None,
):
    """Plot true versus prediction for regression outputs.

    :group: plots

    """

    if cb.conf.save:
        image_directory = cb.conf.image_directory
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)

    for i, target in enumerate(cb.conf.targets):
        if target["type"] != "numerical":
            continue

        target_name = target["name"]

        (
            masked_train_labels,
            masked_train_predictions,
        ) = cb.features.filter_masked(
            train_labels[target_name], train_predictions[i]
        )
        if train_compositions is not None:
            _, masked_train_compositions = cb.features.filter_masked(
                train_labels[target_name], train_compositions
            )
        if train_errorbars is not None:
            _, masked_train_errorbars = cb.features.filter_masked(
                train_labels[target_name], train_errorbars[i]
            )

        _, masked_train_errors = cb.features.filter_masked(
            train_labels[target_name], train_errors[i]
        )
        abs_train_errors = np.abs(masked_train_errors).tolist()
        abs_test_errors = None

        if test_labels is not None:
            (
                masked_test_labels,
                masked_test_predictions,
            ) = cb.features.filter_masked(
                test_labels[target_name], test_predictions[i]
            )
            if test_compositions is not None:
                _, masked_test_compositions = cb.features.filter_masked(
                    test_labels[target_name], test_compositions
                )
            if test_errorbars is not None:
                _, masked_test_errorbars = cb.features.filter_masked(
                    test_labels[target_name], test_errorbars[i]
                )

            abs_test_errors = np.abs(test_errors).tolist()
            _, masked_test_errors = cb.features.filter_masked(
                test_labels[target_name], test_errors[i]
            )

        labelled_errors = []
        if train_compositions is not None:
            labelled_errors.extend(
                gather_labelled_errors(
                    masked_train_labels,
                    masked_train_predictions,
                    masked_train_compositions,
                    masked_train_errors,
                )
            )

        if test_compositions is not None:
            labelled_errors.extend(
                gather_labelled_errors(
                    masked_test_labels,
                    masked_test_predictions,
                    masked_test_compositions,
                    masked_test_errors,
                )
            )

        fig, ax = plt.subplots(figsize=(5, 5))
        # plt.grid(alpha=.4)

        if train_errorbars is not None:
            ax.errorbar(
                masked_train_labels,
                masked_train_predictions,
                yerr=masked_train_errorbars,
                fmt="rx",
                ms=5,
                alpha=0.8,
                capsize=1,
                elinewidth=0.75,
            )
        else:
            if test_labels is not None:
                ax.scatter(
                    masked_train_labels,
                    masked_train_predictions,
                    label="Train",
                    s=5,
                    alpha=0.8,
                    marker="x",
                )
            else:
                ax.scatter(
                    masked_train_labels,
                    masked_train_predictions,
                    s=10,
                    alpha=0.8,
                    marker="x",
                )

        if test_labels is not None:
            if test_errorbars is not None:
                ax.errorbar(
                    masked_test_labels,
                    masked_test_predictions,
                    yerr=masked_test_errorbars,
                    fmt="bx",
                    ms=5,
                    alpha=0.8,
                    capsize=1,
                    elinewidth=0.75,
                )
            else:
                ax.scatter(
                    masked_test_labels,
                    masked_test_predictions,
                    label="Test",
                    s=10,
                    alpha=0.8,
                    marker="o",
                )

            min_point = np.min(
                [np.min(masked_train_labels), np.min(masked_test_labels)]
            )
            max_point = np.max(
                [np.max(masked_train_labels), np.max(masked_test_labels)]
            )
        else:
            min_point = np.min(masked_train_labels)
            max_point = np.max(masked_train_labels)

        lims = [min_point - 0.1 * min_point, max_point + 0.1 * min_point]
        ax.plot(lims, lims, "--k")

        annotations = []
        for e in labelled_errors:
            annotations.append(
                plt.text(
                    e[0],
                    e[1],
                    mg.Alloy(e[2]).to_pretty_string(),
                    fontsize=8,
                )
            )

        plt.xlabel(
            "True "
            + cb.features.prettyName(target_name)
            + " ("
            + cb.features.units[target_name]
            + ")"
        )

        plt.ylabel(
            "Predicted "
            + cb.features.prettyName(target_name)
            + " ("
            + cb.features.units[target_name]
            + ")"
        )

        if metrics is not None:
            descriptionStr = (
                r"$R^2$"
                + ": "
                + str(round(metrics[target_name]["train"]["R_sq"], 3))
                + "\nRMSE: "
                + str(round(metrics[target_name]["train"]["RMSE"], 3))
                + " "
                + cb.features.units[target_name]
                + "\nMAE: "
                + str(round(metrics[target_name]["train"]["MAE"], 3))
                + " "
                + cb.features.units[target_name]
            )

            if test_labels is not None:
                descriptionStr = (
                    r"$R^2$"
                    + ": Train: "
                    + str(round(metrics[target_name]["train"]["R_sq"], 3))
                    + ", Test: "
                    + str(round(metrics[target_name]["test"]["R_sq"], 3))
                    + "\nRMSE ("
                    + cb.features.units[target_name]
                    + "): Train: "
                    + str(round(metrics[target_name]["train"]["RMSE"], 3))
                    + ", Test: "
                    + str(round(metrics[target_name]["test"]["RMSE"], 3))
                    + "\nMAE ("
                    + cb.features.units[target_name]
                    + "): Train: "
                    + str(round(metrics[target_name]["train"]["MAE"], 3))
                    + ", Test: "
                    + str(round(metrics[target_name]["test"]["MAE"], 3))
                )
            ob = mpl.offsetbox.AnchoredText(descriptionStr, loc="upper left")
            ob.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(ob)

        legend = plt.legend(loc="lower right")
        ax.set_aspect("equal", "box")

        X = list(masked_train_labels)
        if test_labels is not None:
            X.extend(masked_test_labels)
        Y = list(masked_train_predictions)
        if test_predictions is not None:
            Y.extend(masked_test_predictions)

        adjust_text(
            annotations,
            X,
            Y,
            add_objects=[legend, ob],
            arrowprops=dict(
                arrowstyle="-|>",
                color="k",
                alpha=0.8,
                lw=0.5,
                mutation_scale=5,
            ),
            lim=10000,
            precision=0.001,
            expand_text=(1.05, 2.5),
            expand_points=(1.05, 1.8),
        )
        # force_text=0.005, force_points=0.005) masked_train_labels,
        # masked_train_predictions,

        plt.tight_layout()

        if cb.conf.save:
            if not os.path.exists(image_directory + target_name):
                os.makedirs(image_directory + target_name)
            plt.savefig(
                image_directory
                + target_name
                + "/"
                + target_name
                + "_TrueVsPredicted.png"
            )
        else:
            plt.show()

        plt.cla()
        plt.clf()
        plt.close()

        if test_labels is not None:
            plt.hist(masked_train_errors, label="Train", density=True, bins=40)
            plt.hist(masked_test_errors, label="Test", density=True, bins=40)
            plt.legend(loc="best")
            plt.ylabel("Density")
        else:
            plt.hist(masked_train_errors, bins="auto")
            plt.ylabel("Count")

        # plt.grid(alpha=.4)
        plt.xlabel(
            cb.features.prettyName(target_name)
            + " prediction error ("
            + cb.features.units[target_name]
            + ")"
        )

        plt.tight_layout()

        if cb.conf.save:
            plt.savefig(
                image_directory
                + target_name
                + "/"
                + target_name
                + "_PredictionError.png"
            )
        else:
            plt.show()

        plt.cla()
        plt.clf()
        plt.close()


def plot_results_regression_heatmap(train_labels, train_predictions):
    """Plot true versus prediction heatmaps for multiple regression models.

    :group: plots

    """

    i = 0
    for feature in train_labels[0]:
        if feature != "GFA":

            MAEs = []
            RMSEs = []

            min_point = np.Inf
            max_point = -np.Inf

            labels = []
            predictions = []

            for j in range(len(train_labels)):

                t, p = cb.features.filter_masked(
                    train_labels[j][feature], train_predictions[j][i]
                )

                MAEs.append(cb.metrics.calc_MAE(t, p))
                RMSEs.append(cb.metrics.calc_RMSE(t, p))

                labels.extend(t)
                predictions.extend(p)

            extent = [
                np.min(labels),
                np.max(labels),
                np.min(predictions),
                np.max(predictions),
            ]

            # labels = np.ma.masked_where(labels == 0, labels)
            # predictions = np.ma.masked_where(predictions == 0, predictions)

            hist, xbins, ybins = np.histogram2d(labels, predictions, bins=30)

            cmap = mpl.cm.get_cmap("viridis").copy()
            cmap.set_bad(color="white")

            fig, ax = plt.subplots(figsize=(5, 5))
            # plt.grid(alpha=.4)

            plt.imshow(
                np.ma.masked_where(hist == 0, hist).T,
                interpolation="none",
                aspect="equal",
                extent=extent,
                cmap=cmap,
                origin="lower",
                norm=mpl.colors.LogNorm(),
            )

            min_point = np.min(labels)
            max_point = np.max(labels)

            lims = [min_point - 0.1 * min_point, max_point + 0.1 * min_point]
            plt.plot(lims, lims, "--k")

            # plt.autoscale()
            ax.set_aspect("equal", "box")

            description_str = (
                "RMSE: "
                + str(round(np.mean(RMSEs), 3))
                + " "
                + r"$\pm$"
                + " "
                + str(round(np.std(RMSEs), 3))
                + " "
                + cb.features.units[feature]
                + "\nMAE: "
                + str(round(np.mean(MAEs), 3))
                + " "
                + r"$\pm$"
                + " "
                + str(round(np.std(MAEs), 3))
                + " "
                + cb.features.units[feature]
            )
            ob = mpl.offsetbox.AnchoredText(description_str, loc="upper left")
            ob.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(ob)

            plt.xlabel(
                "True "
                + cb.features.prettyName(feature)
                + " ("
                + cb.features.units[feature]
                + ")"
            )
            plt.ylabel(
                "Predicted "
                + cb.features.prettyName(feature)
                + " ("
                + cb.features.units[feature]
                + ")"
            )

            plt.tight_layout()
            if cb.conf.save:
                if not os.path.exists(cb.conf.image_directory + feature):
                    os.makedirs(cb.conf.image_directory + feature)
                plt.savefig(
                    cb.conf.image_directory
                    + feature
                    + "/"
                    + feature
                    + "_TrueVsPredicted_heatmap.png"
                )
            else:
                plt.show()

            plt.cla()
            plt.clf()
            plt.close()

        i += 1


def plot_results_classification(
    train_labels, train_predictions, test_labels=None, test_predictions=None
):
    """Plot a confusion matrix for a classifier.

    :group: plots

    """

    i = 0
    for target in cb.conf.targets:
        if target["type"] == "categorical":
            target_name = target["name"]
            classes = target["classes"]

            image_directory = None
            if cb.conf.save:
                image_directory = cb.conf.image_directory
                if not os.path.exists(image_directory):
                    os.makedirs(image_directory)
                if not os.path.exists(image_directory + target_name):
                    os.makedirs(image_directory + target_name)

            if len(train_labels.columns) > 1:
                (
                    masked_train_labels,
                    masked_train_predictions,
                ) = cb.features.filter_masked(
                    train_labels[target_name], train_predictions[i]
                )
                if test_labels is not None:
                    (
                        masked_test_labels,
                        masked_test_predictions,
                    ) = cb.features.filter_masked(
                        test_labels[target_name], test_predictions[i]
                    )
            else:
                (
                    masked_train_labels,
                    masked_train_predictions,
                ) = cb.features.filter_masked(
                    train_labels[target_name], train_predictions
                )
                if test_labels is not None:
                    (
                        masked_test_labels,
                        masked_test_predictions,
                    ) = cb.features.filter_masked(
                        test_labels[target_name], test_predictions
                    )

            if test_labels is not None:
                sets = ["train", "test"]
            else:
                sets = ["train"]

            for set_name in sets:

                if set_name == "train":
                    raw_predictions = masked_train_predictions
                    labels = masked_train_labels
                else:
                    raw_predictions = masked_test_predictions
                    labels = masked_test_labels

                plot_multiclass_roc(
                    labels,
                    raw_predictions,
                    target_name,
                    set_name,
                    image_directory,
                )

                predictions = []
                for prediction in raw_predictions:
                    predictions.append(np.argmax(prediction))

                fig = plt.figure()
                ax = fig.add_subplot()
                plt.grid(False)

                confusion = confusion_matrix(labels, predictions)
                confusionPlot = ConfusionMatrixDisplay(
                    confusion_matrix=confusion, display_labels=classes
                )
                confusionPlot.plot(colorbar=False, cmap=plt.cm.Blues, ax=ax)
                ax.set_xlabel("Predicted class")
                ax.set_ylabel("True class")

                specificities = []
                markednesses = []
                matthewsCorrelations = []

                for c in range(len(classes)):

                    p = 0
                    n = 0
                    pp = 0
                    pn = 0
                    tp = 0
                    fp = 0
                    tn = 0
                    fn = 0
                    for i in range(len(labels)):
                        if labels[i] == c:
                            p += 1
                            if predictions[i] == c:
                                tp += 1
                        else:
                            n += 1
                            if predictions[i] != c:
                                tn += 1

                        if predictions[i] == c:
                            pp += 1
                            if labels[i] != c:
                                fp += 1
                        else:
                            pn += 1
                            if labels[i] == c:
                                fn += 1

                    accuracy = (tp + tn) / (p + n)
                    recall = tp / p
                    if pp > 0:
                        precision = tp / pp
                    else:
                        precision = 0
                    specificity = tn / n
                    f1 = (2 * tp) / (2 * tp + fp + fn)
                    informedness = recall + specificity - 1

                    if tn + fn > 0:
                        markedness = precision + (tn / (tn + fn)) - 1
                    else:
                        markedness = 0

                    if pn > 0:
                        npv = tn / pn
                    else:
                        npv = 0

                    matthewsCorrelation = np.sqrt(
                        recall * specificity * precision * npv
                    ) - np.sqrt(
                        (1 - recall)
                        * (1 - specificity)
                        * (1 - npv)
                        * (1 - precision)
                    )

                    specificities.append(specificity)
                    markednesses.append(markedness)
                    matthewsCorrelations.append(matthewsCorrelation)

                    ax.text(
                        1.55,
                        1.05 - c * 0.35,
                        "\nAccuracy: "
                        + str(round(accuracy, 3))
                        + "\nRecall: "
                        + str(round(recall, 3))
                        + "\nPrecision: "
                        + str(round(precision, 3))
                        + "\nSpecificity: "
                        + str(round(specificity, 3))
                        + "\nF1: "
                        + str(round(f1, 3))
                        + "\nInformedness: "
                        + str(round(informedness, 3))
                        + "\nMarkedness: "
                        + str(round(markedness, 3))
                        + "\nMatthews Correlation: "
                        + str(round(matthewsCorrelation, 3)),
                        transform=ax.transAxes,
                        verticalalignment="top",
                        horizontalalignment="right",
                        fontsize=8,
                    )

                plt.title(
                    "Accuracy: "
                    + str(
                        round(cb.metrics.calc_accuracy(labels, predictions), 3)
                    )
                    + " Recall: "
                    + str(
                        round(cb.metrics.calc_recall(labels, predictions), 3)
                    )
                    + " Precision: "
                    + str(
                        round(
                            cb.metrics.calc_precision(labels, predictions), 3
                        )
                    )
                    + "\nSpecificity: "
                    + str(round(np.mean(specificities), 3))
                    + " F1: "
                    + str(round(cb.metrics.calc_f1(labels, predictions), 3))
                    + "\nInformedness: "
                    + str(
                        round(
                            cb.metrics.calc_recall(labels, predictions)
                            + np.mean(specificities)
                            - 1,
                            3,
                        )
                    )
                    + " Markedness: "
                    + str(round(np.mean(markednesses), 3))
                    + "\nMatthews Correlation: "
                    + str(round(np.mean(matthewsCorrelations), 3))
                )

                plt.tight_layout()

                if cb.conf.save:
                    plt.savefig(
                        image_directory
                        + target_name
                        + "/confusion_"
                        + set_name
                        + ".png"
                    )
                else:
                    plt.show()

                plt.cla()
                plt.clf()
                plt.close()

        i += 1


def plot_multiclass_roc(true, pred, feature_name, set_name, image_directory):
    """Plot a reciever-operator characteristic graph for multiple classes.

    :group: plots

    """

    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()

    for i in range(3):
        classPred = []
        classTrue = []
        for j in range(len(pred)):

            if true[j] == i:
                classTrue.append(1)
            else:
                classTrue.append(0)

            classPred.append(pred[j][i])

        fpr[i], tpr[i], thresholds[i] = roc_curve(classTrue, classPred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_aspect(aspect="equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    for i in range(3):
        ax.plot(
            fpr[i],
            tpr[i],
            label=["Crystal", "Ribbon", "BMG"][i]
            + " (area = "
            + str(round(roc_auc[i], 3))
            + ") ",
        )
    ax.legend(loc="best")
    plt.tight_layout()

    if cb.conf.save:
        plt.savefig(
            image_directory + feature_name + "/ROC_" + set_name + ".png"
        )
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()


def plot_distributions(data):
    """Plot distributions of input features in the training data set.

    :group: plots

    """

    if cb.conf.save:
        if not os.path.exists(cb.conf.image_directory + "distributions"):
            os.makedirs(cb.conf.image_directory + "distributions")

    for feature in data.columns:
        if feature == "composition" or feature not in cb.conf.target_names:
            continue

        ax1 = plt.subplot(311)

        crystalData = cb.features.filter_masked(
            data[data["GFA"] == 0][feature]
        )
        bins = "auto"
        if len(crystalData) == 0:
            bins = 1

        plt.hist(crystalData, bins=bins, color="b")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set(yticklabels=[])
        ax1.tick_params(left=False)
        ax1.set_title("Crystals")

        ribbonData = cb.features.filter_masked(data[data["GFA"] == 1][feature])
        bins = "auto"
        if len(ribbonData) == 0:
            bins = 1
        ax2 = plt.subplot(312, sharex=ax1)
        plt.hist(ribbonData, bins=bins, color="r")
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set(yticklabels=[])
        ax2.tick_params(left=False)
        ax2.set_title("Ribbons")

        bmgData = cb.features.filter_masked(data[data["GFA"] == 2][feature])
        bins = "auto"
        if len(bmgData) == 0:
            bins = 1
        ax3 = plt.subplot(313, sharex=ax1)
        plt.hist(bmgData, bins=bins, color="g")
        plt.setp(ax3.get_xticklabels())
        ax3.set(yticklabels=[])
        ax3.tick_params(left=False)
        ax3.set_title("BMGs")

        #        ax4 = plt.subplot(414, sharex=ax1)
        #        plt.hist(filter_masked(data[data['GFA'] == cb.features.mask_value]
        #                               [feature]), bins="auto", color="c")
        # plt.setp(ax4.get_xticklabels())
        #        ax4.set(yticklabels=[])
        #        ax4.tick_params(left=False)
        #        ax4.set_title('Unknown')

        label = cb.features.prettyName(feature)
        if feature in cb.features.units:
            label += " (" + cb.features.units[feature] + ")"

        plt.xlabel(label)
        # plt.gca().xaxis.grid(True)

        plt.tight_layout()

        if cb.conf.save:
            plt.savefig(
                cb.conf.image_directory + "distributions/" + feature + ".png"
            )
        else:
            plt.show()

        plt.cla()
        plt.clf()
        plt.close()

        plt.hist(cb.features.filter_masked(data[feature]), bins=25)
        plt.xlabel(label)
        plt.ylabel("Count")
        # plt.grid(alpha=.4)

        plt.yscale("log")

        plt.tight_layout()

        if cb.conf.save:
            plt.savefig(
                cb.conf.image_directory
                + "distributions/"
                + feature
                + "_all.png"
            )
        else:
            plt.show()

        plt.cla()
        plt.clf()
        plt.close()


def plot_feature_variation(data, suffix=None):
    """Plot a the quartile coefficient of dispersion for each feature in the
    training data.

    :group: plots

    """

    if cb.conf.save:
        if not os.path.exists(cb.conf.image_directory):
            os.makedirs(cb.conf.image_directory)

    tmpData = data.copy()
    tmpData = tmpData.replace(cb.features.mask_value, np.nan)

    if "composition" in tmpData.columns:
        tmpData = tmpData.drop("composition", axis="columns")

    featureNames = []
    coefficients = []
    for feature in tmpData.columns:
        if feature == "composition" or feature in cb.conf.target_names:
            continue

        featureNames.append(cb.features.prettyName(feature))

        Q1 = np.percentile(tmpData[feature], 25)
        Q3 = np.percentile(tmpData[feature], 75)
        if np.abs(Q1 + Q3) > 0:
            coefficients.append(np.abs((Q3 - Q1) / (Q3 + Q1)))
        else:
            coefficients.append(0)

    coefficients, featureNames = zip(*sorted(zip(coefficients, featureNames)))

    fig, ax = plt.subplots(figsize=(10, 0.15 * len(featureNames)))
    # plt.grid(axis='x', alpha=.4)
    plt.barh(featureNames, coefficients)
    plt.ylim(-1, len(featureNames))
    plt.xlabel("Quartile coefficient of dispersion")
    plt.tight_layout()

    if cb.conf.save:
        if suffix is None:
            plt.savefig(cb.conf.image_directory + "variance.png")
        else:
            plt.savefig(
                cb.conf.image_directory + "variance_" + suffix + ".png"
            )
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()


def plot_correlation(data, suffix=None):
    """Plot correlations between pairs of features, using a correlation matrix
    and a dendrogram.

    :group: plots

    """

    if cb.conf.save:
        if not os.path.exists(cb.conf.image_directory + "correlations"):
            os.makedirs(cb.conf.image_directory + "correlations")

    tmpData = data.copy()
    tmpData = tmpData.replace(cb.features.mask_value, np.nan)

    if "composition" in tmpData.columns:
        tmpData = tmpData.drop("composition", axis="columns")

    correlation = tmpData.corr()
    correlation = np.array(correlation.replace(np.nan, 0))

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    corr_linkage = hierarchy.ward(correlation)
    dendro = hierarchy.dendrogram(
        corr_linkage,
        labels=[cb.features.prettyName(f) for f in tmpData.columns],
        ax=ax,
        orientation="right",
    )
    # plt.grid(alpha=.4)
    plt.xlabel("Feature distance")
    plt.ylabel("Features")
    plt.tight_layout()

    if cb.conf.save:
        if suffix is not None:
            plt.savefig(
                cb.conf.image_directory
                + "correlations/dendrogram_"
                + suffix
                + ".png"
            )
        else:
            plt.savefig(
                cb.conf.image_directory + "correlations/dendrogram.png"
            )
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    mask = np.triu(np.ones_like(correlation, dtype=bool))

    plt.figure(figsize=(50, 50))
    hmap = sns.heatmap(
        correlation[dendro["leaves"], :][:, dendro["leaves"]],
        mask=mask,
        cmap="Spectral",
        vmax=1,
        vmin=-1,
        square=True,
        annot=True,
        center=0,
        cbar=False,
        yticklabels=[
            cb.features.prettyName(f)
            for f in tmpData.columns[dendro["leaves"]]
        ],
        xticklabels=[
            cb.features.prettyName(f)
            for f in tmpData.columns[dendro["leaves"]]
        ],
    )

    plt.tight_layout()
    if cb.conf.save:
        if suffix is not None:
            hmap.figure.savefig(
                cb.conf.image_directory
                + "correlations/all_correlation_"
                + suffix
                + ".png",
                format="png",
            )
        else:
            hmap.figure.savefig(
                cb.conf.image_directory + "correlations/all_correlation.png",
                format="png",
            )
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()

    for feature in cb.conf.target_names:
        if feature not in tmpData:
            continue

        featureCorrelation = np.abs(
            correlation[tmpData.columns.get_loc(feature)]
        )
        featureNames = tmpData.columns
        featureCorrelation, featureNames = zip(
            *sorted(zip(featureCorrelation, featureNames), reverse=True)
        )

        significantCorrelations = []
        significantCorrelationFeatures = []
        colors = []

        i = 0
        while len(significantCorrelations) < 20 and i < len(featureNames):
            if (
                featureNames[i] != feature
                and featureNames[i] not in cb.conf.target_names
            ):
                significantCorrelations.append(featureCorrelation[i])
                significantCorrelationFeatures.append(
                    cb.features.prettyName(featureNames[i])
                )

                correlationValue = correlation[
                    tmpData.columns.get_loc(feature)
                ][tmpData.columns.get_loc(featureNames[i])]

                if correlationValue < 0:
                    colors.append("r")
                else:
                    colors.append("b")

            i += 1

        significantCorrelationFeatures.reverse()
        significantCorrelations.reverse()
        colors.reverse()

        # plt.grid(axis='x', alpha=.4)
        plt.barh(
            significantCorrelationFeatures,
            significantCorrelations,
            color=colors,
        )
        plt.ylim(-1, len(significantCorrelationFeatures))
        plt.xlabel("Correlation with " + feature)
        plt.xlim((0, 1))
        plt.tight_layout()

        if cb.conf.save:
            if suffix is not None:
                plt.savefig(
                    cb.conf.image_directory
                    + "correlations/"
                    + feature
                    + "_correlation_"
                    + suffix
                    + ".png"
                )
            else:
                plt.savefig(
                    cb.conf.image_directory
                    + "correlations/"
                    + feature
                    + "_correlation.png"
                )
        else:
            plt.show()

        plt.cla()
        plt.clf()
        plt.close()


def plot_feature_permutation(data):
    """Plot the results of feature permutation, ranking the most important features.

    :group: plots

    """

    if cb.conf.save:
        if not os.path.exists(cb.conf.image_directory + "permutation"):
            os.makedirs(cb.conf.image_directory + "permutation")

    for target in cb.conf.targets:
        tmp_data = []
        tmp_features = []
        tmp_means = []
        for feature in data:
            if target.name in data[feature]:
                if target.type == "numerical":
                    tmp_data.append(data[feature][target.name])
                else:
                    tmp_data.append(data[feature][target.name] * 100)
                tmp_means.append(np.mean(tmp_data[-1]))
                tmp_features.append(feature)

            sorted_indices = np.array(tmp_means).argsort()

        fig, ax = plt.subplots(figsize=(10, 0.14 * len(tmp_features)))

        plt.barh(
            [
                cb.features.prettyName(f)
                for f in np.array(tmp_features)[sorted_indices]
            ],
            np.array(tmp_data)[sorted_indices].T,
        )
        plt.ylim(-1, len(tmp_features))

        if target.type == "categorical":
            plt.xlabel("Decrease in " + target.name + " Accuracy (%)")
        else:
            plt.xlabel(
                "Increase in "
                + cb.features.prettyName(target.name)
                + " Mean Absolute Error ("
                + cb.features.units[target.name]
                + ")"
            )

        plt.ylabel("Permuted Feature")

        plt.tight_layout()
        if cb.conf.save:
            plt.savefig(
                cb.conf.image_directory
                + "/permutation/"
                + target.name
                + "_permutation.png"
            )
        else:
            plt.show()

        plt.clf()
        plt.cla()
        plt.close()

        topN = 10
        plt.barh(
            [
                cb.features.prettyName(f)
                for f in np.array(tmp_features)[sorted_indices]
            ][-topN:],
            np.array(tmp_data)[sorted_indices].T[-topN:],
        )

        if target.type == "categorical":
            plt.xlabel("Decrease in " + target.name + " Accuracy (%)")
        else:
            plt.xlabel(
                "Increase in "
                + cb.features.prettyName(target.name)
                + " Mean Absolute Error ("
                + cb.features.units[target.name]
                + ")"
            )

        plt.ylabel("Permuted Feature")

        plt.tight_layout()

        if cb.conf.save:
            plt.savefig(
                cb.conf.image_directory
                + "/permutation/"
                + target.name
                + "_permutation_top"
                + str(topN)
                + ".png"
            )
        else:
            plt.show()

        plt.clf()
        plt.cla()
        plt.close()

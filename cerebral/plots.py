import os

import cerebral as cb
import metallurgy as mg
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from sklearn import manifold, decomposition, random_projection, discriminant_analysis, neighbors, cluster
import seaborn as sns  # pylint: disable=import-error
import matplotlib.ticker as ticker  # pylint: disable=import-error
import matplotlib.pyplot as plt  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import matplotlib as mpl  # pylint: disable=import-error
from mpl_toolkits import axes_grid1
import matplotlib.collections as mcoll

from . import metrics
from . import features

mpl.use('Agg')
plt.style.use('ggplot')
plt.rc('axes', axisbelow=True)


def plot_training(history, model_name=None):

    if not os.path.exists(cb.conf.image_directory):
        os.makedirs(cb.conf.image_directory)

    image_directory = cb.conf.image_directory
    if model_name is not None:
        image_directory += str(model_name) + "/"
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)

    for metric in history.history:

        if 'val_' not in metric:
            plt.plot(
                history.history[metric], '-b', label='Train')

            if 'val_' + metric in history.history:
                plt.plot(
                    history.history['val_' + metric], '-r', label='Test')
                plt.legend(loc="best")

            # plt.grid(alpha=.4)
            plt.xlabel('Epochs')
            if 'loss' in metric or metric == 'lr' or "MSE" in metric or "MAE" in metric or "Huber" in metric:
                plt.yscale('log')

            if "GFA_" in metric and "_loss" not in metric:
                plt.ylim(0, 1)

            if len(cb.conf.targets) == 1:
                metric = cb.conf.targets[0].name + "_" + metric

            plt.ylabel(metric)

            if '_' in metric:
                feature = metric.split('_')[0]
                if not os.path.exists(image_directory + feature):
                    os.makedirs(image_directory + feature)
                plt.savefig(image_directory +
                            feature + '/' + metric + '.png')
            else:
                plt.savefig(image_directory + metric + '.png')
            plt.cla()
            plt.clf()
            plt.close()


def plot_results_regression(train_labels, train_predictions,
                            test_labels=None, test_predictions=None,
                            train_compositions=None,
                            test_compositions=None,
                            train_errorbars=None, test_errorbars=None,
                            model_name=None):

    image_directory = cb.conf.image_directory
    if model_name is not None:
        image_directory += str(model_name)+'/'
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    i = 0
    for feature in train_labels:
        if feature != 'GFA':

            tmp_train_labels, tmp_train_predictions = features.filter_masked(
                train_labels[feature], train_predictions[i])
            if train_compositions is not None:
                _, tmp_train_compositions = features.filter_masked(
                    train_labels[feature], train_compositions)
            if train_errorbars is not None:
                _, tmp_train_errorbars = features.filter_masked(
                    train_labels[feature], train_errorbars[i])

            train_R_sq = metrics.calc_R_sq(
                tmp_train_labels, tmp_train_predictions)
            train_RMSE = metrics.calc_RMSE(
                tmp_train_labels, tmp_train_predictions)
            train_MAE = metrics.calc_MAE(
                tmp_train_labels, tmp_train_predictions)

            train_error = list(tmp_train_predictions - tmp_train_labels)
            abs_train_error = np.abs(train_error).tolist()

            test_error = None
            abs_test_error = None

            if test_labels is not None:
                tmp_test_labels, tmp_test_predictions = features.filter_masked(
                    test_labels[feature], test_predictions[i])
                if test_compositions is not None:
                    _, tmp_test_compositions = features.filter_masked(
                        test_labels[feature], test_compositions)
                if test_errorbars is not None:
                    _, tmp_test_errorbars = features.filter_masked(
                        test_labels[feature], test_errorbars[i])

                test_R_sq = metrics.calc_R_sq(
                    tmp_test_labels, tmp_test_predictions)
                test_RMSE = metrics.calc_RMSE(
                    tmp_test_labels, tmp_test_predictions)
                test_MAE = metrics.calc_MAE(
                    tmp_test_labels, tmp_test_predictions)

                test_error = list(tmp_test_predictions - tmp_test_labels)
                abs_test_error = np.abs(test_error).tolist()

            labelled_errors = []
            if train_compositions is not None:
                top_errors = sorted(abs_train_error, reverse=True)[:3]
                for e in top_errors:
                    index = abs_train_error.index(e)
                    labelled_errors.append([tmp_train_labels[index],
                                            tmp_train_predictions[index],
                                            tmp_train_compositions[index]])
                if model_name is not None:
                    resultsFilePath = cb.conf.output_directory + '/' + \
                        str(model_name) + "_" + feature + '_error.dat'
                else:
                    resultsFilePath = cb.conf.output_directory + '/' + feature + '_error.dat'

                with open(resultsFilePath, 'w') as resultsFile:
                    for j in range(len(tmp_train_compositions)):
                        resultsFile.write(tmp_train_compositions[j] +
                                          ' ' + str(tmp_train_labels[j]) + ' ' +
                                          str(tmp_train_predictions[j]) + ' ' +
                                          str(train_error[j]) + '\n')

            if test_compositions is not None:
                top_errors = sorted(abs_test_error, reverse=True)[:3]
                for e in top_errors:
                    index = abs_test_error.index(e)
                    labelled_errors.append([tmp_test_labels[index],
                                            tmp_test_predictions[index],
                                            tmp_test_compositions[index]])

                if model_name is not None:
                    resultsFilePath = cb.conf.output_directory + '/' + \
                        str(model_name) + "_" + feature + '_error_test.dat'
                else:
                    resultsFilePath = cb.conf.output_directory + '/' + feature + '_error_test.dat'
                with open(resultsFilePath, 'w') as resultsFile:
                    for j in range(len(tmp_test_compositions)):
                        resultsFile.write(tmp_test_compositions[j] + ' ' +
                                          str(tmp_test_labels[j]) + ' ' + str(tmp_test_predictions[j]) +
                                          ' ' + str(test_error[j]) + '\n')

            fig, ax = plt.subplots(figsize=(5, 5))
            # plt.grid(alpha=.4)

            if train_errorbars is not None:
                ax.errorbar(tmp_train_labels, tmp_train_predictions,
                            yerr=tmp_train_errorbars, fmt='rx', ms=5,
                            alpha=0.8, capsize=1, elinewidth=0.75)
            else:
                if test_labels is not None:
                    ax.scatter(tmp_train_labels, tmp_train_predictions,
                               label="Train", s=10, alpha=0.8, marker='x',
                               c='r')
                else:
                    ax.scatter(tmp_train_labels, tmp_train_predictions,
                               s=10, alpha=0.8, marker='x', c='r')

            if test_labels is not None:
                if test_errorbars is not None:
                    ax.errorbar(tmp_test_labels, tmp_test_predictions,
                                yerr=tmp_test_errorbars, fmt='bx',
                                ms=5, alpha=0.8, capsize=1,
                                elinewidth=0.75)
                else:
                    ax.scatter(tmp_test_labels, tmp_test_predictions,
                               label="Test", s=10, alpha=0.8,
                               marker='x')
                minPoint = np.min(
                    [np.min(tmp_train_labels), np.min(tmp_test_labels)])
                maxPoint = np.max(
                    [np.max(tmp_train_labels), np.max(tmp_test_labels)])
            else:
                minPoint = np.min(tmp_train_labels)
                maxPoint = np.max(tmp_train_labels)

            lims = [minPoint - 0.1 * minPoint,
                    maxPoint + 0.1 * minPoint]
            ax.plot(lims, lims, '--k')

            annotations = []
            for e in labelled_errors:
                annotations.append(plt.text(
                    e[0], e[1],
                    mg.Alloy(e[2]).to_pretty_string(),
                    fontsize=8))

            plt.xlabel('True ' + features.prettyName(feature) +
                       ' (' + features.units[feature] + ')')

            plt.ylabel('Predicted ' + features.prettyName(feature) +
                       ' (' + features.units[feature] + ')')

            descriptionStr = r'$R^2$' + ": " + str(round(train_R_sq, 3)) + "\nRMSE: " + str(round(
                train_RMSE, 3)) + " " + features.units[feature] + "\nMAE: " + str(round(train_MAE, 3)) + " " + features.units[feature]

            if test_labels is not None:
                descriptionStr = r'$R^2$' + ": Train: " + str(round(train_R_sq, 3)) + ", Test: " + str(round(test_R_sq, 3)) + "\nRMSE (" + features.units[feature] + "): Train: " + str(round(
                    train_RMSE, 3)) + ", Test: " + str(round(test_RMSE, 3)) + "\nMAE (" + features.units[feature] + "): Train: " + str(round(train_MAE, 3)) + ", Test: " + str(round(test_MAE, 3))

            legend = plt.legend(loc="lower right")
            ob = mpl.offsetbox.AnchoredText(descriptionStr, loc="upper left")
            ob.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(ob)

            ax.set_aspect('equal', 'box')

            X = list(tmp_train_labels)
            if test_labels is not None:
                X.extend(tmp_test_labels)
            Y = list(tmp_train_predictions)
            if test_predictions is not None:
                Y.extend(tmp_test_predictions)

            adjust_text(annotations, X, Y, add_objects=[legend, ob],
                        arrowprops=dict(arrowstyle="-|>", color='k',
                                        alpha=0.8, lw=0.5, mutation_scale=5),
                        lim=10000, precision=0.001, expand_text=(1.05,
                                                                 2.5),
                        expand_points=(1.05, 1.8))
            # force_text=0.005, force_points=0.005) tmp_train_labels,
            # tmp_train_predictions,

            plt.tight_layout()

            if not os.path.exists(image_directory + feature):
                os.makedirs(image_directory + feature)
            plt.savefig(image_directory + feature + "/" +
                        feature + "_TrueVsPredicted.png")
            plt.cla()
            plt.clf()
            plt.close()

            train_error = list(tmp_train_predictions - tmp_train_labels)

            if test_labels is not None:
                plt.hist(train_error, label="Train", density=True, bins=40)
                plt.hist(test_error, label="Test", density=True, bins=40)
                plt.legend(loc="best")
                plt.ylabel('Density')
            else:
                plt.hist(train_error, bins="auto")
                plt.ylabel('Count')

            # plt.grid(alpha=.4)
            plt.xlabel(features.prettyName(feature) +
                       ' prediction error (' + features.units[feature] + ')')

            plt.tight_layout()
            plt.savefig(image_directory + feature + "/" +
                        feature + "_PredictionError.png")
            plt.cla()
            plt.clf()
            plt.close()

        i += 1


def plot_results_regression_heatmap(train_labels, train_predictions):

    i = 0
    for feature in train_labels[0]:
        if feature != 'GFA':

            MAEs = []
            RMSEs = []

            minPoint = np.Inf
            maxPoint = -np.Inf

            labels = []
            predictions = []

            for j in range(len(train_labels)):

                t, p = features.filter_masked(
                    train_labels[j][feature], train_predictions[j][i])

                MAEs.append(metrics.calc_MAE(t, p))
                RMSEs.append(metrics.calc_RMSE(t, p))

                labels.extend(t)
                predictions.extend(p)

            extent = [np.min(labels), np.max(labels), np.min(
                predictions), np.max(predictions)]

            # labels = np.ma.masked_where(labels == 0, labels)
            # predictions = np.ma.masked_where(predictions == 0, predictions)

            hist, xbins, ybins = np.histogram2d(labels, predictions, bins=30)

            cmap = mpl.cm.get_cmap("jet").copy()
            cmap.set_bad(color='white')

            fig, ax = plt.subplots(figsize=(5, 5))
            # plt.grid(alpha=.4)

            plt.imshow(np.ma.masked_where(hist == 0, hist).T,
                       interpolation='none', aspect='equal',
                       extent=extent, cmap=cmap, origin='lower',
                       norm=mpl.colors.LogNorm())

            minPoint = np.min(labels)
            maxPoint = np.max(labels)

            lims = [minPoint - 0.1 * minPoint,
                    maxPoint + 0.1 * minPoint]
            plt.plot(lims, lims, '--k')

            # plt.autoscale()
            ax.set_aspect('equal', 'box')

            descriptionStr = "RMSE: " + str(round(np.mean(RMSEs), 3)) + " " + r'$\pm$' + " " + str(round(np.std(RMSEs), 3)) + " " + \
                features.units[feature] + "\nMAE: " + str(round(np.mean(MAEs), 3)) + " " + r'$\pm$' + " " + str(
                    round(np.std(MAEs), 3)) + " " + features.units[feature]
            ob = mpl.offsetbox.AnchoredText(descriptionStr, loc="upper left")
            ob.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(ob)

            plt.xlabel('True ' + features.prettyName(feature) +
                       ' (' + features.units[feature] + ')')
            plt.ylabel('Predicted ' + features.prettyName(feature) +
                       ' (' + features.units[feature] + ')')

            plt.tight_layout()
            if not os.path.exists(cb.conf.image_directory + feature):
                os.makedirs(cb.conf.image_directory + feature)
            plt.savefig(cb.conf.image_directory + feature + "/" +
                        feature + "_TrueVsPredicted_heatmap.png")
            plt.cla()
            plt.clf()
            plt.close()

        i += 1


def plot_results_classification(train_labels, train_predictions,
                                test_labels=None, test_predictions=None,
                                model_name=None):

    i = 0
    for feature in train_labels:
        if feature == 'GFA':

            image_directory = cb.conf.image_directory
            if model_name is not None:
                image_directory += str(model_name) + "/"
            if not os.path.exists(image_directory):
                os.makedirs(image_directory)
            if not os.path.exists(image_directory + feature):
                os.makedirs(image_directory + feature)

            if(len(train_labels.columns) > 1):
                tmp_train_labels, tmp_train_predictions = features.filter_masked(
                    train_labels[feature], train_predictions[i])
                if test_labels is not None:
                    tmp_test_labels, tmp_test_predictions = features.filter_masked(
                        test_labels[feature], test_predictions[i])
            else:
                tmp_train_labels, tmp_train_predictions = features.filter_masked(
                    train_labels[feature], train_predictions)
                if test_labels is not None:
                    tmp_test_labels, tmp_test_predictions = features.filter_masked(
                        test_labels[feature], test_predictions)

            classes = ['Crystal', 'Ribbon', 'BMG']
            if test_labels is not None:
                sets = ['train', 'test']
            else:
                sets = ['train']

            for set in sets:

                if set == 'train':
                    raw_predictions = tmp_train_predictions
                    labels = tmp_train_labels
                else:
                    raw_predictions = tmp_test_predictions
                    labels = tmp_test_labels

                plot_multiclass_roc(labels, raw_predictions,
                                    set, image_directory)

                predictions = []
                for prediction in raw_predictions:
                    predictions.append(np.argmax(prediction))

                fig = plt.figure()
                ax = fig.add_subplot()

                confusion = confusion_matrix(labels, predictions)
                confusionPlot = ConfusionMatrixDisplay(
                    confusion_matrix=confusion, display_labels=classes)
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

                    if tn+fn > 0:
                        markedness = precision + (tn / (tn + fn)) - 1
                    else:
                        markedness = 0

                    if pn > 0:
                        npv = tn / pn
                    else:
                        npv = 0

                    matthewsCorrelation = np.sqrt(recall * specificity * precision * npv) - np.sqrt(
                        (1 - recall) * (1 - specificity) * (1 - npv) * (1 - precision))

                    specificities.append(specificity)
                    markednesses.append(markedness)
                    matthewsCorrelations.append(matthewsCorrelation)

                    ax.text(1.55, 1.05 - c * 0.35, "\nAccuracy: " + str(round(accuracy, 3)) + "\nRecall: " + str(round(recall, 3)) + "\nPrecision: " + str(round(precision, 3)) + "\nSpecificity: " + str(round(specificity, 3)) + "\nF1: " + str(round(f1, 3)) +
                            "\nInformedness: " + str(round(informedness, 3)) + "\nMarkedness: " + str(round(markedness, 3)) + "\nMatthews Correlation: " + str(round(matthewsCorrelation, 3)), transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8)

                plt.title("Accuracy: " + str(round(metrics.calc_accuracy(labels, predictions), 3)) +
                          " Recall: " + str(round(metrics.calc_recall(labels, predictions), 3)) +
                          " Precision: " + str(round(metrics.calc_precision(labels, predictions), 3)) +
                          "\nSpecificity: " + str(round(np.mean(specificities), 3)) +
                          " F1: " + str(round(metrics.calc_f1(labels, predictions), 3)) +
                          "\nInformedness: " + str(round(metrics.calc_recall(labels, predictions) + np.mean(specificities) - 1, 3)) +
                          " Markedness: " + str(round(np.mean(markednesses), 3)) +
                          "\nMatthews Correlation: " + str(round(np.mean(matthewsCorrelations), 3)))

                plt.tight_layout()

                plt.savefig(image_directory + feature +
                            "/GFA_confusion_" + set + ".png")
                plt.cla()
                plt.clf()
                plt.close()

        i += 1


def plot_multiclass_roc(true, pred, set, image_directory):
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

        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(classTrue, classPred)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_aspect(aspect='equal')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    for i in range(3):
        ax.plot(fpr[i], tpr[i], label=['Crystal', 'Ribbon', 'BMG']
                [i] + ' (area = ' + str(round(roc_auc[i], 3)) + ') ')
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(image_directory + "GFA/GFA_ROC_" + set + ".png")
    plt.cla()
    plt.clf()
    plt.close()


def plot_distributions(data):

    if not os.path.exists(cb.conf.image_directory + 'distributions'):
        os.makedirs(cb.conf.image_directory + 'distributions')
    for feature in data.columns:
        if feature == 'composition' or feature not in cb.conf.target_names:
            continue

        ax1 = plt.subplot(311)

        crystalData = features.filter_masked(data[data['GFA'] == 0][feature])
        bins = "auto"
        if(len(crystalData) == 0):
            bins = 1

        plt.hist(crystalData, bins=bins, color="b")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set(yticklabels=[])
        ax1.tick_params(left=False)
        ax1.set_title('Crystals')

        ribbonData = features.filter_masked(data[data['GFA'] == 1][feature])
        bins = "auto"
        if(len(ribbonData) == 0):
            bins = 1
        ax2 = plt.subplot(312, sharex=ax1)
        plt.hist(ribbonData, bins=bins, color="r")
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set(yticklabels=[])
        ax2.tick_params(left=False)
        ax2.set_title('Ribbons')

        bmgData = features.filter_masked(data[data['GFA'] == 2][feature])
        bins = "auto"
        if(len(bmgData) == 0):
            bins = 1
        ax3 = plt.subplot(313, sharex=ax1)
        plt.hist(bmgData, bins=bins, color="g")
        plt.setp(ax3.get_xticklabels())
        ax3.set(yticklabels=[])
        ax3.tick_params(left=False)
        ax3.set_title('BMGs')

#        ax4 = plt.subplot(414, sharex=ax1)
#        plt.hist(filter_masked(data[data['GFA'] == cb.features.maskValue]
#                               [feature]), bins="auto", color="c")
        # plt.setp(ax4.get_xticklabels())
#        ax4.set(yticklabels=[])
#        ax4.tick_params(left=False)
#        ax4.set_title('Unknown')

        label = features.prettyName(feature)
        if feature in features.units:
            label += " ("+features.units[feature]+")"

        plt.xlabel(label)
        # plt.gca().xaxis.grid(True)

        plt.tight_layout()
        plt.savefig(cb.conf.image_directory +
                    'distributions/' + feature + '.png')
        plt.cla()
        plt.clf()
        plt.close()

        plt.hist(features.filter_masked(data[feature]), bins=25)
        plt.xlabel(label)
        plt.ylabel('Count')
        # plt.grid(alpha=.4)

        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(cb.conf.image_directory +
                    'distributions/' + feature + '_all.png')
        plt.cla()
        plt.clf()
        plt.close()


def plot_feature_variation(data, suffix=None):

    if not os.path.exists(cb.conf.image_directory):
        os.makedirs(cb.conf.image_directory)

    tmpData = data.copy()
    tmpData = tmpData.replace(cb.features.maskValue, np.nan)

    if 'composition' in tmpData.columns:
        tmpData = tmpData.drop('composition', axis='columns')

    featureNames = []
    coefficients = []
    for feature in tmpData.columns:
        if feature == 'composition' or feature in cb.conf.target_names:
            continue

        featureNames.append(features.prettyName(feature))

        Q1 = np.percentile(tmpData[feature], 25)
        Q3 = np.percentile(tmpData[feature], 75)
        if np.abs(Q1 + Q3) > 0:
            coefficients.append(np.abs((Q3 - Q1) / (Q3 + Q1)))
        else:
            coefficients.append(0)

    coefficients, featureNames = zip(
        *sorted(zip(coefficients, featureNames)))

    fig, ax = plt.subplots(figsize=(10, 0.15 * len(featureNames)))
    # plt.grid(axis='x', alpha=.4)
    plt.barh(featureNames,
             coefficients)
    plt.ylim(-1, len(featureNames))
    plt.xlabel("Quartile coefficient of dispersion")
    plt.tight_layout()
    if suffix is None:
        plt.savefig(cb.conf.image_directory + 'variance.png')
    else:
        plt.savefig(cb.conf.image_directory +
                    'variance_'+suffix+'.png')
    plt.cla()
    plt.clf()
    plt.close()


def plot_correlation(data, suffix=None):

    if not os.path.exists(cb.conf.image_directory + 'correlations'):
        os.makedirs(cb.conf.image_directory + 'correlations')

    tmpData = data.copy()
    tmpData = tmpData.replace(cb.features.maskValue, np.nan)

    if 'composition' in tmpData.columns:
        tmpData = tmpData.drop('composition', axis='columns')

    correlation = tmpData.corr()
    correlation = np.array(correlation.replace(np.nan, 0))

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    corr_linkage = hierarchy.ward(correlation)
    dendro = hierarchy.dendrogram(
        corr_linkage, labels=[features.prettyName(f) for f in tmpData.columns], ax=ax, orientation="right"
    )
    # plt.grid(alpha=.4)
    plt.xlabel('Feature distance')
    plt.ylabel('Features')
    plt.tight_layout()
    if suffix is not None:
        plt.savefig(cb.conf.image_directory +
                    'correlations/dendrogram_' + suffix + '.png')
    else:
        plt.savefig(cb.conf.image_directory +
                    'correlations/dendrogram.png')
    plt.cla()
    plt.clf()
    plt.close()

    mask = np.triu(np.ones_like(correlation, dtype=bool))

    plt.figure(figsize=(50, 50))
    hmap = sns.heatmap(correlation[dendro['leaves'], :][:, dendro['leaves']], mask=mask, cmap="Spectral",
                       vmax=1, vmin=-1, square=True, annot=True, center=0, cbar=False,
                       yticklabels=[features.prettyName(f) for f in tmpData.columns[dendro['leaves']]], xticklabels=[features.prettyName(f) for f in tmpData.columns[dendro['leaves']]])

    plt.tight_layout()
    if suffix is not None:
        hmap.figure.savefig(
            cb.conf.image_directory + 'correlations/all_correlation_' + suffix + '.png', format='png')
    else:
        hmap.figure.savefig(
            cb.conf.image_directory + 'correlations/all_correlation.png', format='png')
    plt.cla()
    plt.clf()
    plt.close()

    for feature in cb.conf.target_names:
        if feature not in tmpData:
            continue

        featureCorrelation = np.abs(
            correlation[tmpData.columns.get_loc(feature)])
        featureNames = tmpData.columns
        featureCorrelation, featureNames = zip(
            *sorted(zip(featureCorrelation, featureNames), reverse=True))

        significantCorrelations = []
        significantCorrelationFeatures = []
        colors = []

        i = 0
        while(len(significantCorrelations) < 20 and i < len(featureNames)):
            if(featureNames[i] != feature and featureNames[i] not in cb.conf.target_names):
                significantCorrelations.append(featureCorrelation[i])
                significantCorrelationFeatures.append(
                    features.prettyName(featureNames[i]))

                correlationValue = correlation[tmpData.columns.get_loc(
                    feature)][tmpData.columns.get_loc(featureNames[i])]

                if(correlationValue < 0):
                    colors.append('r')
                else:
                    colors.append('b')

            i += 1

        significantCorrelationFeatures.reverse()
        significantCorrelations.reverse()
        colors.reverse()

        # plt.grid(axis='x', alpha=.4)
        plt.barh(significantCorrelationFeatures,
                 significantCorrelations, color=colors)
        plt.ylim(-1, len(significantCorrelationFeatures))
        plt.xlabel("Correlation with " + feature)
        plt.xlim((0, 1))
        plt.tight_layout()
        if suffix is not None:
            plt.savefig(cb.conf.image_directory + "correlations/" +
                        feature + '_correlation_' + suffix + '.png')
        else:
            plt.savefig(cb.conf.image_directory + "correlations/" +
                        feature + '_correlation.png')
        plt.cla()
        plt.clf()
        plt.close()


def plot_feature_permutation(data):

    if not os.path.exists(cb.conf.image_directory + 'permutation'):
        os.makedirs(cb.conf.image_directory + 'permutation')

    for target in cb.conf.targets:
        tmp_data = []
        tmp_features = []
        tmp_means = []
        for feature in data:
            if target.name in data[feature]:
                if target.type == 'numerical':
                    tmp_data.append(data[feature][target.name])
                else:
                    tmp_data.append(data[feature][target.name]*100)
                tmp_means.append(np.mean(tmp_data[-1]))
                tmp_features.append(feature)

            sorted_indices = np.array(tmp_means).argsort()

        fig, ax = plt.subplots(figsize=(10, 0.14*len(tmp_features)))

        plt.barh([features.prettyName(f) for f in np.array(tmp_features)[
                 sorted_indices]], np.array(tmp_data)[sorted_indices].T)
        plt.ylim(-1, len(tmp_features))

        if(target.type == 'categorical'):
            plt.xlabel('Decrease in ' + target.name + ' Accuracy (%)')
        else:
            plt.xlabel("Increase in "+features.prettyName(target.name) + ' Mean Absolute Error (' +
                       features.units[target.name] + ')')

        plt.ylabel('Permuted Feature')

        plt.tight_layout()
        plt.savefig(cb.conf.image_directory + "/permutation/" +
                    target.name + "_permutation.png")
        plt.clf()
        plt.cla()
        plt.close()

        topN = 10
        plt.barh([features.prettyName(f) for f in np.array(tmp_features)[
                 sorted_indices]][-topN:], np.array(tmp_data)[sorted_indices].T[-topN:])

        if(target.type == 'categorical'):
            plt.xlabel('Decrease in ' + target.name + ' Accuracy (%)')
        else:
            plt.xlabel("Increase in "+features.prettyName(target.name) + ' Mean Absolute Error (' +
                       features.units[target.name] + ')')

        plt.ylabel('Permuted Feature')

        plt.tight_layout()
        plt.savefig(cb.conf.image_directory + "/permutation/" +
                    target.name + "_permutation_top"+str(topN)+".png")
        plt.clf()
        plt.cla()
        plt.close()

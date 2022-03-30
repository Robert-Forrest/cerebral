from . import metrics
from . import features
import cerebral as cb
import metallurgy as mg
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from adjustText import adjust_text
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from sklearn import manifold, decomposition, random_projection, discriminant_analysis, neighbors, cluster
import tensorflow as tf  # pylint: disable=import-error
import ternary  # pylint: disable=import-error
import seaborn as sns  # pylint: disable=import-error
import matplotlib.ticker as ticker  # pylint: disable=import-error
import matplotlib.pyplot as plt  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import matplotlib as mpl  # pylint: disable=import-error
from mpl_toolkits import axes_grid1
import matplotlib.collections as mcoll
mpl.use('Agg')
plt.style.use('ggplot')
plt.rc('axes', axisbelow=True)

# mpl.rcParams['text.usetex'] = True


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

            if len(features.predictableFeatures) == 1:
                metric = features.predictableFeatures[0] + "_" + metric

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
            print(train_error)
            print(feature)
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


def plot_binary(elements, model, onlyPredictions=False, originalData=None, inspect_features=["percentage"], additionalFeatures=[]):
    if not os.path.exists(cb.conf.image_directory + "compositions"):
        os.makedirs(cb.conf.image_directory + "compositions")
    binary_dir = cb.conf.image_directory + \
        "compositions/" + "_".join(elements)
    if not os.path.exists(binary_dir):
        os.makedirs(binary_dir)

    realData = []
    requiredFeatures = None
    if originalData is not None:
        for _, row in originalData.iterrows():
            parsedComposition = mg.alloy.parse_composition(row['composition'])
            if set(elements).issuperset(set(parsedComposition.keys())):
                if elements[0] in parsedComposition:
                    row['percentage'] = parsedComposition[elements[0]] * 100
                else:
                    row['percentage'] = 0
                realData.append(row)
        realData = pd.DataFrame(realData)
        realData = realData.reset_index(drop=True)
        requiredFeatures = list(originalData.columns)

    for feature in inspect_features:
        if feature not in requiredFeatures:
            requiredFeatures.append(feature)

    compositions, percentages = mg.binary.generate_alloys(elements)

    all_features = pd.DataFrame(compositions, columns=['composition'])
    all_features = features.calculate_features(all_features,
                                               calculate_extra_features=True,
                                               use_composition_vector=False,
                                               dropCorrelatedFeatures=False,
                                               plot=False,
                                               requiredFeatures=requiredFeatures,
                                               additionalFeatures=additionalFeatures)
    all_features = all_features.drop('composition', axis='columns')
    all_features = all_features.fillna(cb.features.maskValue)
    for feature in features.predictableFeatures:
        all_features[feature] = cb.features.maskValue
    all_features['GFA'] = all_features['GFA'].astype('int64')
    all_features['percentage'] = percentages
    GFA_predictions = []

    prediction_ds = features.df_to_dataset(all_features)
    predictions = model.predict(prediction_ds)
    for i in range(len(features.predictableFeatures)):
        if features.predictableFeatures[i] == 'GFA':
            GFA_predictions = predictions[i]
        else:
            all_features[features.predictableFeatures[i]] = predictions[i]

    for inspect_feature in inspect_features:
        if inspect_feature not in all_features.columns:
            continue
        if not os.path.exists(binary_dir+'/'+inspect_feature):
            os.makedirs(binary_dir + '/'+inspect_feature)
        if not os.path.exists(binary_dir+'/'+inspect_feature + '/predictions'):
            os.makedirs(binary_dir+'/'+inspect_feature + '/predictions')
        if not os.path.exists(binary_dir+'/'+inspect_feature + '/features'):
            os.makedirs(binary_dir+'/'+inspect_feature + '/features')

        for feature in all_features.columns:
            trueFeatureName = feature.split(
                '_linearMix')[0].split('_discrepancy')[0]
            if (onlyPredictions and feature not in features.predictableFeatures and trueFeatureName not in additionalFeatures) or inspect_feature == feature:
                continue
            if feature not in features.predictableFeatures and os.path.exists(
                    binary_dir + "/"+inspect_feature + "/" + feature + ".png"):
                continue

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            # plt.grid(alpha=.4)

            if feature == 'GFA':
                crystal = []
                ribbon = []
                BMG = []
                for prediction in GFA_predictions:
                    crystal.append(prediction[0])
                    ribbon.append(prediction[1])
                    BMG.append(prediction[2])

                ax1.plot(all_features[inspect_feature],
                         crystal, label='Crystal')
                ax1.plot(all_features[inspect_feature], ribbon, label='Ribbon')
                ax1.plot(all_features[inspect_feature], BMG, label='BMG')
                ax1.legend(loc="best")

                if len(realData) > 0 and inspect_feature in realData:
                    ax1.scatter(realData[realData['GFA'] == 0][inspect_feature],
                                [1] * len(realData[realData['GFA'] == 0][inspect_feature]), marker="s", edgecolors='k', zorder=2)
                    ax1.scatter(realData[realData['GFA'] == 1][inspect_feature],
                                [1] * len(realData[realData['GFA'] == 1][inspect_feature]), marker="D", edgecolors='k', zorder=2)
                    ax1.scatter(realData[realData['GFA'] == 2][inspect_feature],
                                [1] * len(realData[realData['GFA'] == 2][inspect_feature]), marker="o", edgecolors='k', zorder=2)

            else:
                if inspect_feature != 'percentage':
                    lc = colorline(
                        all_features[inspect_feature], all_features[feature], ax1, z=all_features['percentage'])
                    cbar = plt.colorbar(lc)
                    cbar.set_label(elements[0] + " %", rotation=270)

                else:
                    ax1.plot(all_features[inspect_feature],
                             all_features[feature])

                if len(realData) > 0 and feature in realData and inspect_feature in realData:
                    crystalData, crystalPercentages = filter_masked(
                        realData[realData['GFA'] == 0][feature], realData[realData['GFA'] == 0][inspect_feature])
                    ribbonData, ribbonPercentages = filter_masked(
                        realData[realData['GFA'] == 1][feature], realData[realData['GFA'] == 1][inspect_feature])
                    bmgData, bmgPercentages = filter_masked(
                        realData[realData['GFA'] == 2][feature], realData[realData['GFA'] == 2][inspect_feature])

                    plotted = False
                    if len(crystalData) > 0:
                        if len(ribbonData) > 0 or len(bmgData) > 0:
                            ax1.scatter(crystalPercentages, crystalData,
                                        marker="s", label="Crystal", edgecolors='k', zorder=20)
                            plotted = True
                    if len(ribbonData) > 0:
                        ax1.scatter(ribbonPercentages, ribbonData,
                                    marker="D", label="Ribbon", edgecolors='k', zorder=20)
                        plotted = True
                    if len(bmgData) > 0:
                        ax1.scatter(bmgPercentages, bmgData,
                                    marker="o", label="BMG", edgecolors='k', zorder=20)
                        plotted = True

                    if plotted:
                        ax1.legend(loc="best")

            ax1.autoscale()
            if inspect_feature == 'percentage':
                ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))

                ax2 = ax1.twiny()
                ax2.set_xticks(ax1.get_xticks())
                ax2.set_xbound(ax1.get_xbound())
                ax2.set_xticklabels([int(100 - x) for x in ax1.get_xticks()])

                ax1.set_xlabel(elements[0] + " %")
                ax2.set_xlabel(elements[1] + " %")
                ax2.grid(False)

            else:
                ax1.set_xlabel(features.prettyName(inspect_feature))

            if feature in features.units:
                ax1.set_ylabel(features.prettyName(feature) +
                               ' (' + features.units[feature] + ')')
            elif feature == 'GFA':
                ax1.set_ylabel("GFA Classification Confidence")
            else:
                ax1.set_ylabel(features.prettyName(feature))

            plt.tight_layout()

            if feature in features.predictableFeatures:
                plt.savefig(binary_dir+'/'+inspect_feature +
                            "/predictions/" + feature + ".png")
            else:
                plt.savefig(binary_dir+'/'+inspect_feature +
                            "/features/" + feature + ".png")

            plt.close()
            plt.cla()
            plt.clf()


def plot_quaternary(elements, model, onlyPredictions=False, originalData=None, additionalFeatures=[]):
    if not os.path.exists(cb.conf.image_directory + "compositions"):
        os.makedirs(cb.conf.image_directory + "compositions")

    quaternary_dir = cb.conf.image_directory + "compositions/" + \
        "_".join(elements)

    if not os.path.exists(quaternary_dir):
        os.makedirs(quaternary_dir)

    if not os.path.exists(quaternary_dir + '/features'):
        os.makedirs(quaternary_dir + '/features')

    if not os.path.exists(quaternary_dir + '/predictions'):
        os.makedirs(quaternary_dir + '/predictions')

    minPercent = 0
    maxPercent = 100
    numPercentages = 6
    step = (maxPercent - minPercent) / float(numPercentages)
    percentages = [round(percentage, 3)
                   for percentage in np.arange(minPercent, maxPercent, step)]

    heatmaps = {}
    allPercentages, all_features, GFA_predictions, realData, step = generate_ternary_compositions(
        elements[:3], model, originalData, quaternary=[elements[3], percentages[0]], additionalFeatures=additionalFeatures)
    for feature in all_features.columns:
        trueFeatureName = feature.split(
            '_linearMix')[0].split('_discrepancy')[0]
        if onlyPredictions and feature not in features.predictableFeatures and trueFeatureName not in additionalFeatures:
            continue
        if feature not in heatmaps:
            if feature != 'GFA':
                heatmaps[feature] = []
            else:
                heatmaps['GFA_crystal'] = []
                heatmaps['GFA_glass'] = []
                heatmaps['GFA_ribbon'] = []
                heatmaps['GFA_bmg'] = []
                heatmaps['GFA_argmax'] = []

    realDatas = {}
    for percentage in percentages:
        allPercentages, all_features, GFA_predictions, realData, step = generate_ternary_compositions(
            elements[:3], model, originalData, quaternary=[elements[3], percentage], additionalFeatures=additionalFeatures)
        realDatas[percentage] = realData

        doneGFA = False
        for feature in heatmaps:
            if "GFA_" in feature:
                if not doneGFA:
                    doneGFA = True
                    heatmap_data_crystal, heatmap_data_glass, heatmap_data_ribbon, heatmap_data_bmg, heatmap_data_argmax = generate_ternary_heatmap_data(
                        "GFA", GFA_predictions, allPercentages, step)

                    heatmaps['GFA_crystal'].append(heatmap_data_crystal)
                    heatmaps['GFA_glass'].append(heatmap_data_glass)
                    heatmaps['GFA_ribbon'].append(heatmap_data_ribbon)
                    heatmaps['GFA_bmg'].append(heatmap_data_bmg)
                    heatmaps['GFA_argmax'].append(heatmap_data_argmax)
            else:
                heatmap_data = generate_ternary_heatmap_data(
                    feature, all_features, allPercentages, step)
                heatmaps[feature].append(heatmap_data)

    for feature in heatmaps:
        if 'GFA_' not in feature:
            vmax = -np.inf
            vmin = np.inf
            label = features.prettyName(feature)
            if feature in features.units:
                label += " (" + features.units[feature] + ")"

            tmpFeature = feature
            suffix = None
            for i in range(len(percentages)):
                for coord in heatmaps[feature][i]:
                    if heatmaps[feature][i][coord] > vmax:
                        vmax = heatmaps[feature][i][coord]
                    if heatmaps[feature][i][coord] < vmin:
                        vmin = heatmaps[feature][i][coord]
        else:
            if feature != 'GFA_argmax':
                vmax = 1
                vmin = 0
                GFA_type = feature.split('_')[1]
                if GFA_type == 'bmg':
                    GFA_type = GFA_type.upper()
                else:
                    GFA_type = GFA_type[0].upper() + GFA_type[1:]

                label = GFA_type + " probability (%)"
                suffix = feature.split('_')[1]
                tmpFeature = "GFA"
            else:
                vmax = 2
                vmin = 0
                GFA_type = "GFA Classification"
                label = GFA_type
                suffix = feature.split('_')[1]
                tmpFeature = "GFA"

        for i in range(len(percentages)):
            ternary_dir = cb.conf.image_directory + "compositions/" + \
                "_".join(elements) + "/" + elements[3] + str(percentages[i])
            if not os.path.exists(ternary_dir):
                os.makedirs(ternary_dir)
                os.makedirs(ternary_dir + '/features')
                os.makedirs(ternary_dir + '/predictions')

            if tmpFeature == 'GFA':
                ternary_heatmap(heatmaps[feature][i], tmpFeature,
                                elements[:3], step, ternary_dir,
                                realData=realDatas[percentages[i]],
                                quaternary=[elements[3],
                                            percentages[i]], label=label,
                                suffix=suffix, vmax=vmax, vmin=vmin)
            else:
                ternary_heatmap(heatmaps[feature][i], tmpFeature,
                                elements[:3], step, ternary_dir,
                                realData=realDatas[percentages[i]],
                                quaternary=[elements[3],
                                            percentages[i]], label=label,
                                suffix=suffix)

        columns = int(np.ceil(np.sqrt(len(percentages))))
        rows = int(np.ceil(len(percentages) / columns))
        numGridCells = columns*rows
        gridExcess = numGridCells - len(percentages)

        fig = plt.figure(figsize=(4 * columns, 4 * rows))

        lastAx = None
        for i in reversed(range(len(percentages))):
            iRow = i // columns
            iCol = i % columns

            ax = plt.subplot2grid(
                (rows, columns), (iRow, iCol))

            if gridExcess != 0 and iRow == (rows-1):
                if gridExcess % 2 == 0:
                    ax = plt.subplot2grid(
                        (rows, columns), (iRow, iCol+1))
                else:
                    ax = plt.subplot2grid(
                        (rows, columns * 2), (iRow, 1+(iCol*2)), colspan=2)

            if lastAx is None:
                lastAx = ax

            ternary_heatmap(heatmaps[feature][i], tmpFeature,
                            elements[:3], step, ternary_dir,
                            realData=realDatas[percentages[i]],
                            quaternary=[elements[3], percentages[i]],
                            label=label, suffix=suffix, vmin=vmin,
                            vmax=vmax, ax=ax, showColorbar=False)

        jet_cmap = mpl.cm.get_cmap('jet')

        cax = fig.add_axes([lastAx.get_position().x1 + 0.01,
                            lastAx.get_position().y0 + 0.03,
                            0.0075,
                            lastAx.get_position().height])

        cb = fig.colorbar(mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=jet_cmap),
            cax=cax)
        cb.set_label(label, labelpad=20, rotation=270)

        if feature in features.predictableFeatures or "GFA_" in feature:
            fig.savefig(quaternary_dir + '/predictions/' +
                        feature + ".png", bbox_inches='tight')
        else:
            fig.savefig(quaternary_dir + '/features/' +
                        feature + ".png", bbox_inches='tight')

        plt.clf()
        plt.cla()
        plt.close()


def generate_ternary_compositions(
        elements, model, originalData, quaternary=None, minPercent=0, maxPercent=100, step=None, additionalFeatures=[]):
    if step is None:
        step = 0.02 * (maxPercent - minPercent)

    realData = []
    for _, row in originalData.iterrows():
        parsedComposition = mg.alloy.parse_composition(row['composition'])
        if quaternary is not None:
            if quaternary[1] > 0:
                if quaternary[0] not in parsedComposition:
                    continue
                elif parsedComposition[quaternary[0]] != quaternary[1]:
                    continue

        if set(elements).issuperset(set(parsedComposition.keys())):

            tmpComposition = []
            for e in elements:
                if e in parsedComposition:
                    tmpComposition.append(parsedComposition[e] * 100 / step)
                else:
                    tmpComposition.append(0)

            row['percentages'] = tuple(tmpComposition)
            realData.append(row)
    realData = pd.DataFrame(realData)

    compositions, allPercentages = mg.ternary.generate_alloys(
        elements, step, minPercent, maxPercent, quaternary)

    all_features = pd.DataFrame(compositions, columns=['composition'])
    all_features = features.calculate_features(all_features,
                                               dropCorrelatedFeatures=False,
                                               plot=False,
                                               use_composition_vector=False,
                                               calculate_extra_features=True,
                                               additionalFeatures=additionalFeatures)
    all_features = all_features.drop('composition', axis='columns')
    all_features = all_features.fillna(cb.features.maskValue)

    for feature in features.predictableFeatures:
        all_features[feature] = cb.features.maskValue
    all_features['GFA'] = all_features['GFA'].astype('int64')
    GFA_predictions = []

    prediction_ds = features.df_to_dataset(all_features)
    predictions = model.predict(prediction_ds)
    for i in range(len(features.predictableFeatures)):
        if features.predictableFeatures[i] == 'GFA':
            GFA_predictions = predictions[i]
        else:
            all_features[features.predictableFeatures[i]
                         ] = predictions[i].flatten()

    return allPercentages, all_features, GFA_predictions, realData, step


def generate_ternary_heatmap_data(feature, data, percentages, step):

    if feature == 'GFA':

        heatmap_data_crystal = dict()
        heatmap_data_ribbon = dict()
        heatmap_data_bmg = dict()
        heatmap_data_glass = dict()
        heatmap_data_argmax = dict()
        for i in range(len(data)):
            heatmap_data_crystal[(percentages[i][0] / step,
                                  percentages[i][1] / step)] = data[i][0]
            heatmap_data_ribbon[(percentages[i][0] / step,
                                 percentages[i][1] / step)] = data[i][1]
            heatmap_data_bmg[(percentages[i][0] / step,
                              percentages[i][1] / step)] = data[i][2]
            heatmap_data_glass[(percentages[i][0] / step,
                                percentages[i][1] / step)] = 1 - data[i][0]
            heatmap_data_argmax[(percentages[i][0] / step,
                                 percentages[i][1] / step)] = np.argmax(data[i])

        return heatmap_data_crystal, heatmap_data_glass, heatmap_data_ribbon, heatmap_data_bmg, heatmap_data_argmax
    else:
        heatmap_data = dict()
        for i, row in data.iterrows():
            heatmap_data[(percentages[i][0] / step,
                          percentages[i][1] / step)] = row[feature]
        return heatmap_data


def plot_ternary(elements, model, onlyPredictions=False,
                 originalData=None, quaternary=None, additionalFeatures=[]):
    if not os.path.exists(cb.conf.image_directory + "compositions"):
        os.makedirs(cb.conf.image_directory + "compositions")

    if quaternary is None:
        ternary_dir = cb.conf.image_directory + \
            "compositions/" + "_".join(elements)
    else:
        ternary_dir = cb.conf.image_directory + "compositions/" + \
            "_".join(elements) + "_" + \
            quaternary[0] + "/" + quaternary[0] + str(quaternary[1])

    if not os.path.exists(ternary_dir):
        os.makedirs(ternary_dir)
    if not os.path.exists(ternary_dir + '/predictions'):
        os.makedirs(ternary_dir + '/predictions')
    if not os.path.exists(ternary_dir + '/features'):
        os.makedirs(ternary_dir + '/features')

    allPercentages, all_features, GFA_predictions, realData, step = generate_ternary_compositions(
        elements, model, originalData, quaternary=quaternary, additionalFeatures=additionalFeatures)

    for feature in all_features.columns:
        trueFeatureName = feature.split(
            '_linearMix')[0].split('_discrepancy')[0]
        if onlyPredictions and feature not in features.predictableFeatures and trueFeatureName not in additionalFeatures:
            continue
        if feature not in features.predictableFeatures and os.path.exists(
                ternary_dir + "/" + feature + ".png"):
            continue

        if feature == 'GFA':
            heatmap_data_crystal, heatmap_data_glass, heatmap_data_ribbon, heatmap_data_bmg, heatmap_data_argmax = generate_ternary_heatmap_data(
                feature, GFA_predictions, allPercentages, step)

            ternary_heatmap(heatmap_data_crystal, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                            label="Crystal probability (%)", suffix="Crystal", vmin=0, vmax=1)

            ternary_heatmap(heatmap_data_glass, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                            label="Glass probability (%)", suffix="Glass", vmin=0, vmax=1)

            ternary_heatmap(heatmap_data_ribbon, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                            label="Ribbon probability (%)", suffix="Ribbon", vmin=0, vmax=1)

            ternary_heatmap(heatmap_data_bmg, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                            label="BMG probability (%)", suffix="BMG", vmin=0, vmax=1)

            ternary_heatmap(heatmap_data_argmax, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                            label="Predicted GFA Class", suffix="Classification", vmin=0, vmax=2)

        else:
            heatmap_data = generate_ternary_heatmap_data(
                feature, all_features, allPercentages, step)

            ternary_heatmap(heatmap_data, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary)


def ternary_heatmap(heatmap_data, feature, elements, step,
                    ternary_dir, realData=None, quaternary=None,
                    label=None, suffix=None, vmin=None, vmax=None,
                    ax=None, showColorbar=True):

    scale = 100 / step
    multiple = 10 / step
    fontsize = 10
    tick_fontsize = 6
    tick_offset = 0.018
    gridline_width = 0.3
    gridline_style = '--'
    gridline_color = 'white'

    if label is None:
        label = features.prettyName(feature)
        if feature in features.units:
            label += " ("+features.units[feature]+")"

    title = None
    if quaternary is not None:
        title = "(" + "".join(elements) + ")$_{" + str(
            round(100 - quaternary[1], 2)) + "}$" + quaternary[0] + "$_{" + str(
                round(quaternary[1], 2)) + "}$"

    if ax is None:
        figure, tax = ternary.figure(scale=scale)
    else:
        figure, tax = ternary.figure(scale=scale, ax=ax)

    tax.get_axes().axis('off')

    tax.gridlines(color=gridline_color, multiple=multiple,
                  linewidth=gridline_width, ls=gridline_style)

    tax.set_axis_limits(
        {'b': [0, 100], 'l': [0, 100], 'r': [0, 100]})
    tax.get_ticks_from_axis_limits(multiple=multiple)
    tax.set_custom_ticks(
        fontsize=tick_fontsize, offset=tick_offset, multiple=multiple)

    tax.left_axis_label(elements[2], fontsize=fontsize, offset=0.12)
    tax.right_axis_label(elements[1], fontsize=fontsize, offset=0.12)
    tax.bottom_axis_label(elements[0], fontsize=fontsize, offset=0.12)
    tax.clear_matplotlib_ticks()

    tax.set_title(title)

    jet_cmap = mpl.cm.get_cmap('jet')
    tax.heatmap(heatmap_data, cmap=jet_cmap, vmax=vmax,
                vmin=vmin, cbarlabel=label, colorbar=showColorbar)
    ternary_scatter(realData, tax)

    tax.get_axes().set_aspect(1)
    tax._redraw_labels()

    filepath = ""
    if feature in features.predictableFeatures:
        filepath += ternary_dir + "/predictions/" + feature
    else:
        filepath += ternary_dir + "/features/" + feature

    if suffix is not None:
        filepath += '_' + suffix

    if ax is None:
        tax.savefig(filepath + '.png')
        tax.close()
        figure.clf()


def ternary_scatter(data, tax):

    if len(data) > 0:
        plotted = False
        if len(data[data['GFA'] == 0]) > 0:
            if len(data[data['GFA'] == 1]) > 0 and len(data[data['GFA'] == 2]) > 0:
                tax.scatter(data[data['GFA'] == 0]['percentages'],
                            marker='s', label="Crystal", edgecolors='k',
                            zorder=2)
                plotted = True

        if len(data[data['GFA'] == 1]) > 0:
            tax.scatter(data[data['GFA'] == 1]['percentages'],
                        label="Ribbon", marker='D', zorder=2, edgecolors='k')
            plotted = True

        if len(data[data['GFA'] == 2]) > 0:
            tax.scatter(data[data['GFA'] == 2]['percentages'],
                        marker='o', label="BMG", zorder=2, edgecolors='k')
            plotted = True

        if plotted:
            tax.legend(loc="upper right", handletextpad=0.1, frameon=False)


def plot_distributions(data):

    if not os.path.exists(cb.conf.image_directory + 'distributions'):
        os.makedirs(cb.conf.image_directory + 'distributions')
    for feature in data.columns:
        if feature == 'composition' or feature not in features.predictableFeatures:
            continue

        ax1 = plt.subplot(311)

        crystalData = filter_masked(data[data['GFA'] == 0][feature])
        bins = "auto"
        if(len(crystalData) == 0):
            bins = 1

        plt.hist(crystalData, bins=bins, color="b")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set(yticklabels=[])
        ax1.tick_params(left=False)
        ax1.set_title('Crystals')

        ribbonData = filter_masked(data[data['GFA'] == 1][feature])
        bins = "auto"
        if(len(ribbonData) == 0):
            bins = 1
        ax2 = plt.subplot(312, sharex=ax1)
        plt.hist(ribbonData, bins=bins, color="r")
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax2.set(yticklabels=[])
        ax2.tick_params(left=False)
        ax2.set_title('Ribbons')

        bmgData = filter_masked(data[data['GFA'] == 2][feature])
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

        plt.hist(filter_masked(data[feature]), bins=25)
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
        if feature == 'composition' or feature in features.predictableFeatures:
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

    for feature in features.predictableFeatures:
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
            if(featureNames[i] != feature and featureNames[i] not in features.predictableFeatures):
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

    for predictFeature in features.predictableFeatures:
        tmp_data = []
        tmp_features = []
        tmp_means = []
        for feature in data:
            if predictFeature in data[feature]:
                if predictFeature != 'GFA':
                    tmp_data.append(data[feature][predictFeature])
                else:
                    tmp_data.append(data[feature][predictFeature]*100)
                tmp_means.append(np.mean(tmp_data[-1]))
                tmp_features.append(feature)

            sorted_indices = np.array(tmp_means).argsort()

        fig, ax = plt.subplots(figsize=(10, 0.14*len(tmp_features)))

        # plt.boxplot(np.array(tmp_data)[sorted_indices].T,
        #             labels=[features.prettyName(f) for f in
        #                     np.array(tmp_features)[sorted_indices]],
        #             vert=False)

        plt.barh([features.prettyName(f) for f in np.array(tmp_features)[
                 sorted_indices]], np.array(tmp_data)[sorted_indices].T)
        plt.ylim(-1, len(tmp_features))

        if(predictFeature == 'GFA'):
            plt.xlabel('Decrease in GFA Accuracy (%)')
        else:
            plt.xlabel("Increase in "+features.prettyName(predictFeature) + ' Mean Absolute Error (' +
                       features.units[predictFeature] + ')')

        plt.ylabel('Permuted Feature')

        # plt.grid(alpha=.4)

        plt.tight_layout()
        plt.savefig(cb.conf.image_directory + "/permutation/" +
                    predictFeature + "_permutation.png")
        plt.clf()
        plt.cla()
        plt.close()

        topN = 10
        plt.barh([features.prettyName(f) for f in np.array(tmp_features)[
                 sorted_indices]][-topN:], np.array(tmp_data)[sorted_indices].T[-topN:])

        if(predictFeature == 'GFA'):
            plt.xlabel('Decrease in GFA Accuracy (%)')
        else:
            plt.xlabel("Increase in "+features.prettyName(predictFeature) + ' Mean Absolute Error (' +
                       features.units[predictFeature] + ')')

        plt.ylabel('Permuted Feature')

        # plt.grid(alpha=.4)

        plt.tight_layout()
        plt.savefig(cb.conf.image_directory + "/permutation/" +
                    predictFeature + "_permutation_top"+str(topN)+".png")
        plt.clf()
        plt.cla()
        plt.close()


def plot_entropy_enthalpy(data):

    crystalEnthalpy = []
    crystalEntropy = []

    ribbonEnthalpy = []
    ribbonEntropy = []

    BMGEnthalpy = []
    BMGEntropy = []

    for _, row in data.iterrows():
        if(row['GFA'] == 0):
            crystalEnthalpy.append(row['mixingEnthalpy'])
            crystalEntropy.append(row['mixingEntropy'])
        elif(row['GFA'] == 1):
            ribbonEnthalpy.append(row['mixingEnthalpy'])
            ribbonEntropy.append(row['mixingEntropy'])
        elif(row['GFA'] == 2):
            BMGEnthalpy.append(row['mixingEnthalpy'])
            BMGEntropy.append(row['mixingEntropy'])

    plt.plot(crystalEnthalpy, crystalEntropy,
             'o', label='Crystal', markersize=1)
    plt.plot(ribbonEnthalpy, ribbonEntropy, 'o', label='Ribbon', markersize=1)
    plt.plot(BMGEnthalpy, BMGEntropy, 'o', label='BMG', markersize=1)
    plt.legend(loc="best")
    plt.ylabel('Mismatch Entropy')
    plt.xlabel('Mixing Enthalpy')
    plt.yscale('log')
    plt.ylim(4e-2, 3)
    # plt.grid(alpha=.4)
    plt.tight_layout()
    plt.savefig(cb.conf.image_directory + 'HvS.png')
    plt.cla()
    plt.clf()
    plt.close()


def plot_data_map(originalData):
    data = originalData.copy()

    colors = []
    for _, row in data.iterrows():
        if(row['GFA'] == 0):
            colors.append('b')
        elif(row['GFA'] == 1):
            colors.append('r')
        elif(row['GFA'] == 2):
            colors.append('g')
        else:
            colors.append('c')

    plt.legend(handles=[
        mpl.patches.Patch(color='b', label='Crystal'),
        mpl.patches.Patch(color='r', label='Ribbon'),
        mpl.patches.Patch(color='g', label='BMG'),
        mpl.patches.Patch(color='c', label='Unknown'),
    ], loc="best")

    scaler = StandardScaler()

    # mds = manifold.TSNE(2, n_jobs=4, init="pca",
    #                     n_iter=1000)
    # mds = decomposition.PCA(n_components=2)
    # mds = manifold.MDS(2, max_iter=10)
    # mds = manifold.Isomap(n_components=2, n_neighbors=10)

    GFA = data.pop("GFA")
    data = data.drop("composition", axis="columns")
    data = data.drop("Dmax", axis="columns")
    data = data.drop("Tl", axis="columns")
    data = data.drop("Tg", axis="columns")
    data = data.drop("Tx", axis="columns")

    # mds = cluster.FeatureAgglomeration(n_clusters=3)
    mds = decomposition.PCA(n_components=min(len(data.columns), 40))
    Y = mds.fit_transform(scaler.fit_transform(data), GFA)

    mds = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2)
    Y = mds.fit_transform(Y, GFA)

    plt.scatter(Y[:, 0], Y[:, 1], s=3, c=colors, alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(cb.conf.image_directory + 'map.png')
    plt.cla()
    plt.clf()
    plt.close()

    pca = decomposition.PCA()
    pca.fit(scaler.fit_transform(data))

    plt.plot(np.arange(1, pca.n_components_ + 1),
             np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    # plt.grid()
    plt.tight_layout()
    plt.savefig(cb.conf.image_directory + 'pca.png')
    plt.cla()
    plt.clf()
    plt.close()


def colorline(x, y, ax, z=None, cmap='jet', norm=plt.Normalize(0.0, 100.0), linewidth=2, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)

    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax.add_collection(lc)
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
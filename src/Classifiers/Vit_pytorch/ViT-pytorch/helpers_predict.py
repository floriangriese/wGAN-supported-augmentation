import json
import logging
import os
import shutil
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy.special import softmax


def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets)**2, axis=1))


def brier_multi_weighted(labels_onehot, probs, labels, classes):
    occurrences, _ = np.histogram(labels, bins=range(len(classes)+1))
    class_weights = (1 / occurrences / (1 / occurrences).sum())
    weight_vector = class_weights[labels]
    weighted_multiclass_brier = (np.average(np.sum((probs - labels_onehot) ** 2, axis=1), weights=weight_vector))
    return weighted_multiclass_brier


def generate_roc_curve(foldername):
    datasets = ["galaxy_GAN_dym"] #"galaxy_mingo_LOFAR_test" "galaxy",  "twin_LOFAR"
    #foldername = "2022-03-18_17_31_30"
    all_dataset_paths = []
    classes = ["FRI", "FRII", "Compact", "Bent"]
    for dataset in datasets:
        paths = []
        paths.append("experiments/{}/{}ViT_galaxy_{}.h5".format(foldername,foldername,dataset))
        for k in range(1,5):
            path = "experiments/{}/{}ViT_galaxy_GAN_1{}_{}.h5".format(foldername,foldername, k, dataset)
            paths.append(path)
        all_dataset_paths.append(paths)
    #file_h5 = "../src/Classifiers/RunClassifierPipeline/experiments/{}/{}ViT_galaxy_{}.h5".format(foldername,foldername, "galaxy")
    for l, d in enumerate(datasets):
        for k, path in enumerate(all_dataset_paths[l]):
            with h5py.File(path, "r") as file:
                for key in file.keys():
                    labels_entry = file[key + "/labels"]
                    pred_label_entry = file[key + "/predicted_labels"]
                    pred_logits_entry = file[key + "/predicted_logits"]
                    labels = np.array(labels_entry)
                    predicted_labels = np.array(pred_label_entry)
                    pred_logits = np.array(pred_logits_entry)
                    pred_logits_softmax = softmax(pred_logits, axis=1)
                    labels_onehot = label_binarize(labels, classes=[0, 1, 2, 3])

                    brier_score = brier_multi(labels_onehot, pred_logits_softmax)
                    # Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(4):
                        fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], pred_logits[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])

                    fpr["micro"], tpr["micro"], _ = roc_curve(labels_onehot.ravel(), pred_logits.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    plt.figure()
                    lw = 2
                    plt.plot(
                        fpr[0],
                        tpr[0],
                        color="darkorange",
                        lw=lw,
                        label="ROC curve (area = %0.2f)" % roc_auc[0],
                    )
                    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Receiver operating characteristic example")
                    plt.legend(loc="lower right")
                    plt.show()

                    # First aggregate all false positive rates
                    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

                    # Then interpolate all ROC curves at this points
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in range(4):
                        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

                    # Finally average it and compute AUC
                    mean_tpr /= 4

                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                    # Plot all ROC curves
                    plt.figure()
                    plt.plot(
                        fpr["micro"],
                        tpr["micro"],
                        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
                        color="deeppink",
                        linestyle=":",
                        linewidth=4,
                    )

                    plt.plot(
                        fpr["macro"],
                        tpr["macro"],
                        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                        color="navy",
                        linestyle=":",
                        linewidth=4,
                    )

                    colors = cycle(["aqua", "darkorange", "cornflowerblue", "springgreen"])
                    for i, color in zip(range(4), colors):
                        plt.plot(
                            fpr[i],
                            tpr[i],
                            color=color,
                            lw=lw,
                            label="ROC curve of class {0} (area = {1:0.2f})".format(classes[i], roc_auc[i]),
                        )

                    plt.plot([0, 1], [0, 1], "k--", lw=lw)
                    plt.xlim([-0.05, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Receiver operating characteristic")
                    plt.legend(loc="lower right")
                    #plt.show()
                    plt.savefig("experiments/{}/{}ViT_galaxy_GAN_1{}_{}_rocCurve.svg".format(foldername, foldername, k, d), dpi=300, format="svg")


def generate_overview_results(foldername, ratios, crosssplits, classes):
    len_classes = len(classes)

    accuracies = np.zeros((len(ratios), len(crosssplits)))
    f1s = np.zeros((len(ratios), len(crosssplits), len_classes))
    precisions = np.zeros((len(ratios), len(crosssplits), len_classes))
    recalls = np.zeros((len(ratios), len(crosssplits), len_classes))
    briers = np.zeros((len(ratios), len(crosssplits)))
    roc_auc_scores = np.zeros((len(ratios), len(crosssplits)))
    if len_classes==4:
        for ind_ratio, ratio in enumerate(ratios):
            for ind_crosssplit, crosssplit in enumerate(crosssplits):
                path = "experiments_predict/{0}/{0}galaxy_data_crossvalid_{1}_h5_ra_{2}.json".format(foldername, crosssplit, ratio)
                with open(path) as json_file:
                    data = json.load(json_file)
                    accuracies[ind_ratio, ind_crosssplit] = data["accuracy"]
                    briers[ind_ratio, ind_crosssplit] = data["brier_score"]
                    roc_auc_scores[ind_ratio, ind_crosssplit] = data["roc_auc_score"]
                    precisions[ind_ratio, ind_crosssplit, :] = [data['0']["precision"], data['1']["precision"], data['2']["precision"],
                                        data['3']["precision"]]
                    recalls[ind_ratio, ind_crosssplit, :] = [data['0']["recall"], data['1']["recall"], data['2']["recall"], data['3']["recall"]]
                    f1s[ind_ratio, ind_crosssplit, :] = [data['0']["f1-score"], data['1']["f1-score"], data['2']["f1-score"], data['3']["f1-score"]]
    elif len_classes==3:
        for ind_ratio, ratio in enumerate(ratios):
            for ind_crosssplit, crosssplit in enumerate(crosssplits):
                path = "experiments_predict/{0}/{0}galaxy_data_crossvalid_{1}_h5_ra_{2}.json".format(foldername, crosssplit, ratio)
                with open(path) as json_file:
                    data = json.load(json_file)
                    accuracies[ind_ratio, ind_crosssplit] = data["accuracy"]
                    briers[ind_ratio, ind_crosssplit] = data["brier_score"]
                    roc_auc_scores[ind_ratio, ind_crosssplit] = data["roc_auc_score"]
                    precisions[ind_ratio, ind_crosssplit, :] = [data['0']["precision"], data['1']["precision"], data['2']["precision"]]
                    recalls[ind_ratio, ind_crosssplit, :] = [data['0']["recall"], data['1']["recall"], data['2']["recall"]]
                    f1s[ind_ratio, ind_crosssplit, :] = [data['0']["f1-score"], data['1']["f1-score"], data['2']["f1-score"]]


    fig = plt.figure(figsize=(60, 60))
    fig, axs = plt.subplots(nrows=len_classes, ncols=3, constrained_layout=True)
    #fig.tight_layout()
    precisions_mean = np.mean(precisions, axis=1)
    precisions_std = np.std(precisions, axis=1)
    recalls_mean = np.mean(recalls, axis=1)
    recalls_std = np.std(recalls, axis=1)
    f1s_mean = np.mean(f1s, axis=1)
    f1_std = np.std(f1s, axis=1)
    accuracies_std = np.std(accuracies, axis=1)
    print("Accuracies Standard deviation per ratio: {}".format(accuracies_std))
    xticklabels = ['1:0', '1:1', '1:2', '1:3', '1:5']
    labels = classes

    for l in range(len_classes):
        axs[l, 0].errorbar(np.arange(len(ratios)), precisions_mean[:, l], yerr=precisions_std[:, l], fmt="-", capsize=4.0, linewidth=1.0)
        axs[l, 1].errorbar(np.arange(len(ratios)), recalls_mean[:, l], yerr=recalls_std[:, l], fmt="-", capsize=4.0, linewidth=1.0)
        axs[l, 2].errorbar(np.arange(len(ratios)), f1s_mean[:, l], yerr=f1_std[:, l], fmt="-", capsize=4.0, linewidth=1.0)
        #axs[0, 0].boxplot(np.transpose(precisions[:,:,k]))
        #axs[0, 1].boxplot(np.transpose(recalls[:,:,k]))
        #axs[1, 0].boxplot(np.transpose(f1s[:,:,k]))

        if l==0:
            axs[l, 0].set_title('Precision')
            axs[l, 1].set_title('Recall')
            axs[l, 2].set_title('F1-Score')
        if l == 3:
            axs[l, 0].set(ylabel="Precision", xlabel="real & fake ratio",
                              xticks=np.arange(len(ratios)),
                              xticklabels=xticklabels, ylim=[np.min(f1s) - 0.3, 1.0]) #ylim=[np.min(precisions) - 0.3, 1.1] #xlim=[-0.2, 4.2]

            axs[l, 1].set(ylabel="Recall", xlabel="real & fake ratio",
                              xticks=np.arange(len(ratios)),
                              xticklabels=xticklabels, ylim=[np.min(f1s) - 0.3, 1.0]) # xlim=[-0.2, 4.2], ylim=[np.min(recalls) - 0.3, 1.0],

            axs[l, 2].set(ylabel="F1-Score", xlabel="real & fake ratio",
                              xticks=np.arange(len(ratios)),
                              xticklabels=xticklabels, ylim=[np.min(f1s) - 0.3, 1.0]) #xlim=[-0.2, 4.2],
        axs[l, 0].set(ylabel="Precision",
                      xticks=np.arange(len(ratios)),
                      xticklabels=xticklabels,
                      ylim=[np.min(f1s) - 0.3, 1.0])  # ylim=[np.min(precisions) - 0.3, 1.1] #xlim=[-0.2, 4.2]

        axs[l, 1].set(ylabel="Recall",
                      xticks=np.arange(len(ratios)),
                      xticklabels=xticklabels,
                      ylim=[np.min(f1s) - 0.3, 1.0])  # xlim=[-0.2, 4.2], ylim=[np.min(recalls) - 0.3, 1.0],

        axs[l, 2].set(ylabel="F1-Score",
                      xticks=np.arange(len(ratios)),
                      xticklabels=xticklabels, ylim=[np.min(f1s) - 0.3, 1.0])  # xlim=[-0.2, 4.2],
        axs[l, 0].legend(loc="lower right", labels=[labels[l]], ncol=1, fontsize="xx-small")
        axs[l, 1].legend(loc="lower right", labels=[labels[l]], ncol=1, fontsize="xx-small")
        axs[l, 2].legend(loc="lower right", labels=[labels[l]], ncol=1, fontsize="xx-small")
    fig.suptitle("{} dataset".format("galaxy_data_crossvalid"), fontsize=16)
    fig.savefig("experiments/{}/{}_{}_dataset_pre_rec_f1.svg".format(foldername, foldername, "galaxy_data_crossvalid"), dpi=300,
                    format="svg")

    #fig = plt.figure(figsize=(60, 120))
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4), constrained_layout=True)
    #fig.tight_layout()
    axs[0].boxplot(np.transpose(accuracies))
    axs[0].set_title('Accuracy')
    axs[0].set(ylabel="Accuracy", xlabel="real & fake ratio",
                  ylim=[np.min(accuracies) - 0.025, np.max(accuracies) + 0.025], xticks=np.arange(1, len(ratios)+1),
                  xticklabels=xticklabels) #xlim=[-0.2, 4.2],

    axs[1].boxplot(np.transpose(briers))
    axs[1].set_title('Brier score')
    axs[1].set(ylabel="Brier score", xlabel="real & fake ratio",
                  ylim=[np.min(briers) - 0.025, np.max(briers) + 0.025], xticks=np.arange(1, len(ratios)+1),
                  xticklabels=xticklabels) #xlim=[-0.2, 4.2],

    axs[2].boxplot(np.transpose(roc_auc_scores))
    axs[2].set_title('roc auc score')
    axs[2].set(ylabel="roc auc score", xlabel="real & fake ratio",
                  ylim=[np.min(roc_auc_scores) - 0.025, np.max(roc_auc_scores) + 0.025], xticks=np.arange(1, len(ratios)+1),
                  xticklabels=xticklabels) #xlim=[-0.2, 4.2],

    fig.suptitle("{} dataset".format("galaxy_data_crossvalid"), fontsize=16)
    fig.savefig("experiments/{}/{}_{}_dataset_acc_brier_roc.svg".format(foldername, foldername, "galaxy_data_crossvalid"), dpi=300, format="svg")
    #plt.show()


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


if __name__ == "__main__":
    foldername = os.path.join("2022-11-28_14_15_55")
    datasetname = "galaxy_data_crossvalid"
    classes = ["FRI", "FRII", "Bent"]
    real_fake_ratios = [0, 1, 2, 3, 5]  # [0, 1, 2, 3]
    datasetsplits = {0: "galaxy_data_crossvalid_0_h5.h5",
                     1: "galaxy_data_crossvalid_1_h5.h5",
                     2: "galaxy_data_crossvalid_2_h5.h5",
                     3: "galaxy_data_crossvalid_3_h5.h5",
                     4: "galaxy_data_crossvalid_4_h5.h5"
                     }
    generate_overview_results(foldername, real_fake_ratios, datasetsplits.keys(), classes)
    print("Finished")
from pathlib import Path
import os
import sys
file_folder = os.path.dirname(__file__)
project_folder = os.path.normpath(os.path.join(file_folder,"..","..","..",".."))
sys.path.append(os.path.normpath(project_folder))
import numpy as np
import json
import logging
import datetime
import time
import shutil
import h5py
import matplotlib
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.special import softmax
from helpers_predict import generate_overview_results, brier_multi_weighted
from sklearn.preprocessing import label_binarize
import argparse
from train import set_seed
from predict import main_prediction

Path("logs_predict").mkdir(parents=True, exist_ok=True)
Path("experiments_predict").mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logfile = os.path.join("logs_predict", "{}-run_all.log".format(datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d_%H_%M_%S')))


if __name__ == "__main__":
    """
        Runs all classifiers defined in the run_all_params.json and writes the result to the experiments folder.
        The parameters of the classifiers are defined in *.json files given in the run_all_params.json under attribute "hyperparameter_file"
    """
    logging.info("Start Classifier inference")
    logging.info("load parameters")
    file_folder = os.path.dirname(__file__)
    datetime_str = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d_%H_%M_%S')
    exp_path = os.path.normpath(os.path.join(file_folder, "experiments_predict", datetime_str))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    experiment_folder = "2022-09-30_18_35_00"
    classes = ["FRI", "FRII", "Compact", "Bent"]
    res_save_paths = []

    datasetname = "galaxy_data_crossvalid"
    real_fake_ratios = [0, 1, 2, 3, 5]
    datasetsplits = {0: "galaxy_data_crossvalid_0_h5.h5",
                     1: "galaxy_data_crossvalid_1_h5.h5",
                     2: "galaxy_data_crossvalid_2_h5.h5",
                     3: "galaxy_data_crossvalid_3_h5.h5",
                     4: "galaxy_data_crossvalid_4_h5.h5"
                    }

    #datasetsplits = {0: "galaxy_data_crossvalid_test_h5.h5",
    #                 1: "galaxy_data_crossvalid_test_h5.h5",
    #                 2: "galaxy_data_crossvalid_test_h5.h5",
    #                 3: "galaxy_data_crossvalid_test_h5.h5",
    #                 4: "galaxy_data_crossvalid_test_h5.h5"}

    predict_dataset = "valid" #  "test" "valid"

    for (k, datasetsplit) in datasetsplits.items():
        for ratio in real_fake_ratios:
            classifier_name = "{}_{}_h5_ra_{}".format(datasetname, k, ratio)
            logging.info("Running {} Classifier".format(classifier_name))
            hyperparameter_file = "experiments/{}/{}_{}_h5_ra_{}.json".format(experiment_folder, datasetname, k, ratio)
            classifier_path = Path(hyperparameter_file)
            # copy files to experiment folder
            shutil.copy(classifier_path, exp_path)
            with open(classifier_path) as f:
                # load hyperparameter file of classifier
                classifier_params = json.load(f)
                classifier_params["--predict_dataset"] = predict_dataset
                classifier_params["--dataset_name"] = datasetsplit

            res_save_path = os.path.join(exp_path, datetime_str + classifier_name + ".h5")
            res_save_paths.append(res_save_path)
            classifier_params["--res_save_path"] = res_save_path
            classifier_params["--classes"] = classes
            classifier_params["--output_dir"] = exp_path

            predict_args = argparse.Namespace()
            predict_args_dict = vars(predict_args)
            for (key, value) in classifier_params.items():
                predict_args_dict[key[2:]] = value

            if predict_args.local_rank == -1:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                predict_args.n_gpu = 1
            predict_args.device = device
            # Set seed
            set_seed(predict_args)
            main_prediction(predict_args)


    # combine results together and provide analysis of results
    for res_save_path in res_save_paths:
        if os.path.isfile(res_save_path):
            with h5py.File(res_save_path, "r") as file:
                for key in file.keys():
                    labels_entry = file[key + "/labels"]
                    pred_label_entry = file[key + "/predicted_labels"]
                    pred_logits_entry = file[key + "/predicted_logits"]
                    labels = np.array(labels_entry)
                    labels_onehot = label_binarize(labels, classes=list(range(len(classes))))
                    predicted_labels = np.array(pred_label_entry)
                    pred_logits = np.array(pred_logits_entry)
                    pred_logits_softmax = softmax(pred_logits, axis=1)
                    save_file_name = os.path.splitext(res_save_path)[0]
                    res_dict = metrics.classification_report(labels, predicted_labels, output_dict=True)

                    brier_score = brier_multi_weighted(labels_onehot, pred_logits_softmax, labels, classes)
                    res_dict["brier_score"] = brier_score
                    roc_auc_score = metrics.roc_auc_score(labels, pred_logits_softmax, multi_class='ovr', average="macro")
                    res_dict["roc_auc_score"] = roc_auc_score
                    # save accuracy to JSON file
                    with open(os.path.join(save_file_name + ".json"), "w") as f:
                        json.dump(res_dict, f, indent=4)

                    metrics_str = metrics.classification_report(labels, predicted_labels)
                    print(metrics_str)
                    # save results
                    file = open(save_file_name + ".txt", "w")
                    file.write(metrics_str)
                    file.close()
                    conf_mat = confusion_matrix(labels, predicted_labels)
                    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
                    disp.plot()
                    plt.title(os.path.basename(save_file_name))
                    plt.savefig(save_file_name + ".svg", dpi=300, format="svg", bbox_inches="tight")
                    #plt.show()

    generate_overview_results(datetime_str, real_fake_ratios, datasetsplits.keys(), classes)
    logging.info("Finished Classifier inference")



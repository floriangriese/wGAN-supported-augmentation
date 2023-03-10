import datetime
import json
import os
import sys
import time
from pathlib import Path
from comet_ml import Experiment

import torch
from torchvision.datasets.utils import download_url

file_folder = os.path.dirname(__file__)
project_folder = os.path.normpath(os.path.join(file_folder, "..", "..", "..", ".."))
sys.path.append(os.path.normpath(project_folder))

import logging
import argparse
from train import train, set_seed


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", default="ViT_pytorch_parameters.json")
    parser.add_argument("--x_val_index", default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--api_key', type=str)
    args = parser.parse_args()

    file_folder = os.path.dirname(__file__)
    project_folder = os.path.normpath(os.path.join(file_folder, "..", "..", "..", ".."))

    # download Vit checkpoint
    checkpoint_vit_path = os.path.join(project_folder, "src", "Classifiers", "Vit_pytorch", "ViT-pytorch", "checkpoint")
    Path(checkpoint_vit_path).mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(os.path.join(checkpoint_vit_path, "ViT-B_16.npz")):
        download_url("https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz", checkpoint_vit_path,
                     "ViT-B_16.npz")

    datetime_str = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d_%H_%M_%S')
    exp_path = os.path.normpath(os.path.join(file_folder, "experiments", datetime_str))
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # change parameter dataset and ratios, and generators

    datasetnames = ["galaxy_data_crossvalid_0_h5.h5", "galaxy_data_crossvalid_1_h5.h5",
                    "galaxy_data_crossvalid_2_h5.h5", "galaxy_data_crossvalid_3_h5.h5",
                    "galaxy_data_crossvalid_4_h5.h5"]
    generator_path_filenames = ["Generator_xval_0_CNN_Histograms_paths.yaml",
                                "Generator_xval_1_CNN_Histograms_paths.yaml",
                                "Generator_xval_2_CNN_Histograms_generator_paths.yaml",
                                "Generator_xval_3_CNN_Histograms_generator_paths.yaml",
                                "Generator_xval_4_CNN_Histograms_generator_paths.yaml"]

    real_fake_ratios = [0, 1, 2, 3, 5]
    for ratio in real_fake_ratios:
        with open(args.params_file) as f:
            params = json.load(f)
            params["--dataset_ratio_real_fake"] = ratio
            params["--dataset_name"] = datasetnames[args.x_val_index]
            params["--generator_path_filename"] = os.path.join(project_folder, "configs",
                                                               generator_path_filenames[args.x_val_index])

        export_filepath = os.path.join(exp_path, "{}_ra_{}.json".format(os.path.splitext(params["--dataset_name"])[0],
                                                                        ratio))
        params["--output_dir"] = exp_path
        train_args = argparse.Namespace()
        train_args_dict = vars(train_args)
        for (k, v) in params.items():
            train_args_dict[k[2:]] = v

        # Create an experiment with your api key
        experiment = Experiment(
            api_key=args.api_key,
            project_name=train_args.project_name,
        )
        experiment.log_parameters(train_args)
        train_args.experiment = experiment

        # Setup CUDA, GPU & distributed training
        if train_args.local_rank == -1:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            train_args.n_gpu = 1  # torch.cuda.device_count()

        train_args.device = device

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN)
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                       (train_args.local_rank, train_args.device, train_args.n_gpu, bool(train_args.local_rank != -1)))
        # Set seed
        set_seed(train_args)

        train(train_args)

        model_checkpoint = os.path.join(exp_path,
                                        "{0}_ra_{1}_checkpoint.bin".format(
                                            os.path.splitext(params["--dataset_name"])[0],
                                            params["--dataset_ratio_real_fake"]))
        params["--pretrained_dir"] = model_checkpoint
        with open(export_filepath, 'w') as f:
            json.dump(params, f, indent=4)

    logging.info("python executable: {}, version: {}".format(sys.executable, sys.version_info))
    logging.info("Finished Classifier training ")

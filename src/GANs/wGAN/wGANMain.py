import argparse
import os
import sys

import comet_ml
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml

import src.GANs.wGAN.wGANGPTrainer as two_player_trainer
import src.GANs.wGAN.wGANModels as models
from firstgalaxydata import FIRSTGalaxyData

DIRNAME = os.path.dirname(__file__)


def main(cmd_args):
    with open(cmd_args['config_path'], 'r') as f:
        metadata = yaml.load(f, yaml.FullLoader)
    print('\n'.join([f'{k}: {v}' for k, v in metadata.items()]))

    if cmd_args['EXP'] is not None:
        metadata['EXP'] = cmd_args['EXP']

    restored_iteration = metadata['restored_iteration'] if 'restored_iteration' in metadata else 0
    restore_exp = metadata['restoreEXP'] if 'restoreEXP' in metadata else 0

    EXPERIMENT_KEY = os.environ.get("COMET_EXPERIMENT_KEY", None)
    comet_resume = False
    if (EXPERIMENT_KEY is not None):  # TODO: This breaks the restore function (overrides it actually)
        api = comet_ml.API()
        try:
            api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
        except Exception:
            api_experiment = None
        if api_experiment is not None:
            comet_resume = True
            step = int(api_experiment.get_parameters_summary("steps")["valueCurrent"])
            epoch = int(api_experiment.get_parameters_summary("epochs")["valueCurrent"])

    if comet_resume:
        experiment = comet_ml.ExistingExperiment(
            previous_experiment=EXPERIMENT_KEY,
            log_env_details=True,
            log_env_gpu=True,
            log_env_cpu=True,
            auto_metric_logging=False
        )
        # Retrieved from above APIExperiment
        experiment.set_step(step)
        experiment.set_epoch(epoch)
    else:
        experiment = comet_ml.Experiment(project_name=cmd_args['comet_project_name'], auto_metric_logging=False)
        experiment.log_parameters(metadata)
        experiment.log_parameters(cmd_args)

    if 'random_seed' in metadata:
        torch.manual_seed(metadata["random_seed"])
        np.random.seed(metadata["random_seed"])

    if cmd_args['no_classical_augmentation']:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5]),
             transforms.CenterCrop([metadata["image_shape"][1], metadata["image_shape"][2]])
             ]
        )
    else:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5]),
             transforms.RandomRotation((-360, 360), fill=-1),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.CenterCrop([metadata["image_shape"][1], metadata["image_shape"][2]])
             ]
        )

    train_data = FIRSTGalaxyData(root=os.path.join(DIRNAME, "../../../data/data_real"),
                                 selected_classes=["FRI", "FRII", "Compact", "Bent"], transform=train_transform,
                                 selected_split='train', input_data_list=cmd_args['input_data_list'])
    print(train_data)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=metadata["batch_size"], shuffle=True,
                                                   num_workers=metadata["workers"])

    device = torch.device("cuda:0" if (torch.cuda.is_available() and metadata["ngpu"] > 0) else "cpu")
    if metadata['ngpu'] > 0 and not torch.cuda.is_available():
        print("Cuda not available but requested. Aborting.")
        sys.exit(1)

    num_classes = len(train_data.class_labels)

    generator = models.Generator(metadata["nz"], metadata["image_shape"][0], metadata["ngf"], num_classes,
                                 cmd_args['sigma'], cmd_args['kernel']).to(device)
    if cmd_args['critic'] == 'FCN':
        critic = models.FCNCritic(metadata['image_shape']).to(device)
    elif cmd_args['critic'] == 'CNN':
        critic = models.CNNCritic(metadata['image_shape'], metadata["image_shape"][0], metadata["ndf"], num_classes).to(
            device)
    else:
        raise ValueError(f'Critic {cmd_args["critic"]} not defined.')
    print(critic)

    print(next(generator.parameters()).device)
    optimiser_G = torch.optim.Adam(generator.parameters(), lr=metadata["lr"], betas=(metadata["beta1"],
                                                                                     metadata["beta2"]))
    optimiser_D = torch.optim.Adam(critic.parameters(), lr=metadata["lr"], betas=(metadata["beta1"],
                                                                                  metadata["beta2"]))
    print(generator)
    print(critic)

    restored_epoch = 0
    if 'restored_iteration' in metadata and metadata['restored_iteration'] > 0:
        raise NotImplementedError()

    else:
        checkpoint_folder = os.path.join(DIRNAME,
                                         f'../../../run/wGAN/{metadata["setup"]}/{cmd_args["comet_project_name"]}/{cmd_args["EXP"]}')

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    two_player_trainer.train(experiment, metadata["num_iterations"], train_dataloader, generator, critic, optimiser_D,
                             optimiser_G, metadata["nz"], metadata["n_critic"], metadata["lambda_gp"], device,
                             metadata["image_shape"], restored_iteration, restored_epoch, metadata["save_model_every"],
                             metadata["validate_model_every"], metadata["validate_stats"],
                             checkpoint_folder=checkpoint_folder)


def get_argparser():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('config_path', type=str, help='Path to config.yaml file.')
    parser.add_argument('--no_classical_augmentation', action='store_true', default=False)
    parser.add_argument('--EXP', type=str)
    parser.add_argument('--comet_project_name', type=str, required=True)
    parser.add_argument('--input_data_list', type=str, nargs="+", default=["galaxy_data_crossvalid_0_h5.h5"])
    parser.add_argument('--sigma', type=float, help='Sigma value for gaussian smearing')
    parser.add_argument('--kernel', nargs=2, type=int, help='Kernel for gaussian smearing')
    parser.add_argument('--critic', type=str, choices=['FCN', 'CNN'], default='CNN')

    return parser


if __name__ == '__main__':
    parser = get_argparser()
    cmd_args = vars(parser.parse_args())
    print('Command Line Arguments\n----------')
    print('\n'.join([f'{k}: {v}' for k, v in cmd_args.items()]))
    print('----------')
    main(cmd_args)

import logging
import os
import sys

import numpy as np

file_folder = os.path.dirname(__file__)
project_folder = os.path.normpath(os.path.join(file_folder, "..", "..", "..", "..", ".."))
sys.path.append(os.path.normpath(project_folder))

from firstgalaxydata import FIRSTGalaxyData
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.transforms import InterpolationMode

from data.data_gen_dynamic.ConcatenatedDataSet import ConcatenatedDataSet
import yaml

logger = logging.getLogger(__name__)


def calc_train_weights(freq_classes):
    freq_pos = freq_classes / np.sum(freq_classes)
    return calc_train_weights_from_probs(freq_pos)


def calc_train_weights_from_probs(freq_pos):
    weight_pos = 1 / freq_pos
    weight_pos_norm = weight_pos / np.max(weight_pos)
    return weight_pos_norm


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train_custom_128 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation((0.0, 359.0), interpolation=InterpolationMode.BILINEAR),
        transforms.Pad((48, 48), fill=0, padding_mode="constant"),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_real_trainset = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
         transforms.RandomRotation((0.0, 359.0), interpolation=InterpolationMode.BILINEAR),
         transforms.CenterCrop([224, 224])
         ]
    )

    train_weights = None

    if args.dataset == "galaxy_GAN_dym":
        print(os.getcwd())
        root_path = os.path.join(project_folder, os.path.join("data", "data_real"))

        generator_filepath = os.path.join(args.generator_path_filename)
        with open(generator_filepath, 'r') as f:
            generator_file = yaml.load(f, yaml.FullLoader)
        generator_paths = generator_file["generator_paths"]

        real_trainset = FIRSTGalaxyData(root=root_path,
                                        input_data_list=[args.dataset_name],
                                        selected_split="train",
                                        selected_classes=args.classes,
                                        is_PIL=True,
                                        is_RGB=True,
                                        transform=transform_real_trainset)

        trainset = ConcatenatedDataSet(real_trainset, generator_paths, args.dataset_ratio_real_fake,
                                       real_trainset.get_class_dict(), balancing_mode=args.balancing_mode,
                                       batch_size=args.bsize, is_RGB=True, transform=transform_train_custom_128,
                                       is_return_real_gen_flag=False, target_transform=None)

        testset = FIRSTGalaxyData(root=root_path,
                                  input_data_list=[args.dataset_name],
                                  selected_split="valid",
                                  selected_classes=args.classes,
                                  is_PIL=True,
                                  is_RGB=True,
                                  transform=transform_real_trainset,
                                  target_transform=None)
        train_weights = trainset.get_probabilities()

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=0,  # 4
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,  # 4
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader, train_weights


def get_loader_predict(args):
    transform_real_testset = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
         transforms.CenterCrop([224, 224])
         ]
    )

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    root_path = os.path.normpath(os.path.join(project_folder, os.path.join("data", "data_real")))
    if args.dataset == "galaxy_GAN_dym":
        predictset = FIRSTGalaxyData(root=root_path,
                                     input_data_list=[args.dataset_name],
                                     selected_split=args.predict_dataset,
                                     selected_classes=args.classes,
                                     is_PIL=True,
                                     is_RGB=True,
                                     transform=transform_real_testset,
                                     target_transform=None)

    predict_weights = calc_train_weights(list(predictset.get_occurrences().values()))
    predict_sampler = SequentialSampler(predictset)
    predict_loader = DataLoader(predictset,
                                sampler=predict_sampler,
                                batch_size=args.eval_batch_size,
                                num_workers=1,
                                pin_memory=True) if predictset is not None else None
    return predict_loader, predict_weights

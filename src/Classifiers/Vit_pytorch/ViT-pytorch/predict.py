# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
import sys
file_folder = os.path.dirname(__file__)
project_folder = os.path.normpath(os.path.join(file_folder,"..","..","..",".."))
sys.path.append(os.path.normpath(project_folder))
import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm

from models.modeling import VisionTransformer, CONFIGS
from utils.data_utils import get_loader_predict
from sklearn import metrics
import h5py
from helper import str2bool


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def predict(args, predict_loader):
    eval_losses = AverageMeter()
    num_classes = args.predict_weights.size()[0]
    config = CONFIGS[args.model_type]
    if args.use_weighting:
        loss_function = torch.nn.CrossEntropyLoss(weight=args.predict_weights)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, loss_fct=loss_function)
    model.load_state_dict(torch.load(os.path.join(args.pretrained_dir)))
    model.to(args.device)
    model.eval()

    logger.info("***** Running prediction *****")
    logger.info("  Num steps = %d", len(predict_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_preds = []
    all_logits = []
    all_label = []
    epoch_iterator = tqdm(predict_loader,
                          desc="Predicting... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_function(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_logits[0] = np.append(
                all_logits[0], logits.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

        epoch_iterator.set_description("Predicting... (loss=%2.5f)" % eval_losses.val)

    all_preds = all_preds[0]
    all_label = all_label[0]

    logger.info("\n")
    logger.info("Prediction Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info(all_preds)
    return all_preds, all_logits, all_label


def main_prediction(args):
    # Prediction
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataset
    predict_loader, predict_weights = get_loader_predict(args)
    args.predict_weights = torch.FloatTensor(predict_weights)

    preds, logits, all_label = predict(args, predict_loader)

    logits = np.squeeze(logits, axis=(0,))
    s_acc = metrics.accuracy_score(all_label, preds)
    f1_score = metrics.f1_score(all_label, preds, average=None)
    print(metrics.classification_report(all_label, preds))
    print(s_acc)
    print(f1_score)

    hf = h5py.File(args.res_save_path, "a")
    g = hf.create_group(args.name)
    g.create_dataset("labels", data=all_label)
    g.create_dataset("predicted_labels", data=preds)
    g.create_dataset("predicted_logits", data=logits)
    hf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default="test_training",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument('--classes', '--list', nargs='+', default=["FRI", "FRII", "Compact", "Bent"],
                        help='<Required> Set flag')
    parser.add_argument("--res_save_path", default="temporary_result.h5")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "galaxy",
                                              "galaxy_LOFAR_test",
                                              "galaxy_FIRST_test", "galaxy_mingo_LOFAR_test",
                                              "galaxy_GAN_dym", "galaxy_LOFAR_dym"],
                        default="galaxy_GAN_dym",
                        help="Which downstream task.")
    parser.add_argument('--dataset_name', type=str, default="galaxy_data_crossvalid_0_h5.h5")
    parser.add_argument('--generator_path_filename', type=str,
                        default="run/Jobs/RunAugmentedClassifierTrainingSendScripts/run_augmented_classifier_training_base_config.yaml")
    parser.add_argument('--predict_dataset', type=str, choices=["test", "valid"], default='valid')
    parser.add_argument('--balancing_mode', type=str, choices=["realistic", "balanced"], default='balanced')
    parser.add_argument('--dataset_ratio_real_fake', type=int, default=1)
    parser.add_argument('--bsize', type=int, default=100)
    parser.add_argument('--project_name', type=str, default="radiogalaxies-vit-crossvalid")
    parser.add_argument("--freeze_parameters", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_weighting", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="experiments/2022-09-30_18_35_00/galaxy_data_crossvalid_0_h5_ra_0_checkpoint.bin",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus") #default=-1
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    main_prediction(args)

    print("Finished Prediction")



# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
from datetime import timedelta

from comet_ml import Experiment

import numpy as np
import torch
import torch.distributed as dist
from scipy.special import softmax
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from helper import str2bool
from models.modeling import CONFIGS, VisionTransformer
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule

file_folder = os.path.dirname(__file__)
project_folder = os.path.normpath(os.path.join(file_folder, "..", "..", "..", ".."))
sys.path.append(os.path.normpath(project_folder))



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


def brier_multi(labels_onehot, probs):
    return np.mean(np.sum((probs - labels_onehot) ** 2, axis=1))


def brier_multi_weighted(args, labels_onehot, probs, labels):
    occurrences, _ = np.histogram(labels, bins=range(len(args.classes) + 1))
    class_weights = (1 / occurrences / (1 / occurrences).sum())
    weight_vector = class_weights[labels]
    weighted_multiclass_brier = (np.average(np.sum((probs - labels_onehot) ** 2, axis=1), weights=weight_vector))
    return weighted_multiclass_brier


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir,
                                    "{0}_ra_{1}_checkpoint.bin".format(os.path.splitext(args.dataset_name)[0],
                                                                       args.dataset_ratio_real_fake))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]


    if args.dataset == "galaxy_GAN_dym":
        num_classes = args.train_weights.size()[0]

    if args.use_weighting:
        loss_function = torch.nn.CrossEntropyLoss(weight=args.train_weights)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, loss_fct=loss_function)
    model.load_from(np.load(args.pretrained_dir))

    if args.freeze_parameters:
        # freeze parameters except classification layer (head)
        for param in model.parameters():
            param.requires_grad = False

        model.head.requires_grad_(requires_grad=True)

    summary(model, input_size=(32, 3, 224, 224))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    all_logits = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
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
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    logits = np.squeeze(all_logits, axis=(0,))
    accuracy = simple_accuracy(all_preds, all_label)
    labels_onehot = label_binarize(all_label, classes=list(range(len(args.classes))))
    pred_logits_softmax = softmax(logits, axis=1)
    brier_score = brier_multi(labels_onehot, pred_logits_softmax)
    brier_score_weighted = brier_multi_weighted(args, labels_onehot, pred_logits_softmax, all_label)
    roc_auc_score = metrics.roc_auc_score(all_label, pred_logits_softmax, multi_class='ovr', average="macro")

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy, eval_losses.avg, brier_score, roc_auc_score, brier_score_weighted


def train(args):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset # get weights
    train_loader, test_loader, train_weights = get_loader(args)

    args.train_weights = torch.FloatTensor(train_weights)
    # Model & Tokenizer Setup
    args, model = setup(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    best_brier = 100000.0
    best_brier_weighted = 100000.0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                    args.experiment.log_metric("global_loss", losses.val, step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy, val_loss, brier_score, roc_auc_score, brier_score_weighted = valid(args, model, writer,
                                                                                                 test_loader,
                                                                                                 global_step)
                    args.experiment.log_metric("val_accurcay", accuracy, step=global_step)
                    args.experiment.log_metric("val_loss", val_loss, step=global_step)
                    args.experiment.log_metric("brier_score", brier_score, step=global_step)
                    args.experiment.log_metric("roc_auc_score", roc_auc_score, step=global_step)
                    args.experiment.log_metric("brier_score_weighted", brier_score_weighted, step=global_step)
                    if best_acc < accuracy:
                        best_acc = accuracy
                        args.experiment.log_metric("val_accurcay_best", best_acc, step=global_step)
                    if best_brier > brier_score:
                        best_brier = brier_score
                        args.experiment.log_metric("brier_score_best", best_brier, step=global_step)
                    if best_brier_weighted > brier_score_weighted:
                        save_model(args, model)
                        best_brier_weighted = brier_score_weighted
                        args.experiment.log_metric("best_brier_weighted", best_brier_weighted, step=global_step)

                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break
    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default="test_training",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["galaxy_GAN_dym"], default="galaxy_GAN_dym",
                        help="Which downstream task.")
    parser.add_argument('--dataset_name', type=str, default="galaxy_data_crossvalid_0_h5.h5")
    parser.add_argument('--generator_path_filename', type=str, default=os.path.join(project_folder, 'configs', 'Generator_xval_0_CNN_Histograms_paths.yaml'))
    parser.add_argument('--predict_dataset', type=str, choices=["test", "valid"], default='valid')
    parser.add_argument('--balancing_mode', type=str, choices=["realistic", "balanced"], default='balanced')
    parser.add_argument('--dataset_ratio_real_fake', type=int, default=1)
    parser.add_argument('--bsize', type=int, default=256)
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--classes', '--list', nargs='+', default=["FRI", "FRII", "Compact", "Bent"],
                        help='<Required> Set flag')
    parser.add_argument("--freeze_parameters", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_weighting", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="./checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=10, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=500, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")  # default=-1
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    # Create an experiment with your api key
    experiment = Experiment(
        api_key=args.api_key,
        project_name=args.project_name,
    )
    experiment.log_parameters(args)
    args.experiment = experiment

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.n_gpu = 1  # torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))
    # Set seed
    set_seed(args)
    train(args)

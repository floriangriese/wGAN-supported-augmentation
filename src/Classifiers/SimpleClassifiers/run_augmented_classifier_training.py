import os
import sys

file_folder = os.path.dirname(__file__)
project_folder = os.path.abspath(os.path.join(file_folder, "..", "..", ".."))
sys.path.append(project_folder)

import argparse
import yaml
import re
from datetime import datetime
import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn import preprocessing
from firstgalaxydata import FIRSTGalaxyData
from src.Classifiers.SimpleClassifiers.Classifiers import SimpleCNN, SimpleFCN
from data.data_gen_dynamic.ConcatenatedDataSet import ConcatenatedDataSet

metadata = {
    'image_shape': [1, 128, 128],
    'random_seed': np.random.randint(1, 9999),
}


def train(args, train_dataloader, classifier, criterion, optimiser, device, experiment, checkpoint_folder):
    val_and_test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5]),
         transforms.CenterCrop([metadata["image_shape"][1], metadata["image_shape"][2]])
         ]
    )
    # plain train is used to evaluate performance on training set and NOT for training itself
    plain_train_data = FIRSTGalaxyData(root=os.path.join(project_folder, "data/data_real"),
                                       selected_classes=args["classes"], transform=val_and_test_transform,
                                       selected_split='train', input_data_list=args['input_data_list'])

    val_data = FIRSTGalaxyData(root=os.path.join(project_folder, "data/data_real"),
                               selected_classes=args["classes"], transform=val_and_test_transform,
                               selected_split='valid', input_data_list=args['input_data_list'])

    test_data = FIRSTGalaxyData(root=os.path.join(project_folder, "data/data_real"),
                                selected_classes=args["classes"], transform=val_and_test_transform,
                                selected_split='test')

    plain_train_dataloader = torch.utils.data.DataLoader(plain_train_data, batch_size=args['bsize'], shuffle=False,
                                                         num_workers=2, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args['bsize'], shuffle=False,
                                                  num_workers=2, pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args['bsize'], shuffle=False,
                                                 num_workers=2, pin_memory=True)

    print(f'Start training {args["n_iterations"]} iterations.')
    n_iter = 0 + args["classifier_step"]
    n_epochs = 0 + args['classifier_epoch']

    with experiment.train():
        while n_iter < args['n_iterations'] + args["classifier_step"]:
            for inner_counter, (data, labels, _) in enumerate(train_dataloader, 0):
                experiment.set_step(n_iter)

                data = data.to(device)
                labels = labels.to(device)
                classifier.train()
                optimiser.zero_grad()

                logits = classifier(data)

                loss = criterion(logits, labels)

                loss.backward()
                optimiser.step()

                if n_iter % args['evaluate_every'] == 0:
                    print(f'Evaluating performance at step {n_iter}')
                    classifier.eval()
                    with experiment.train():
                        experiment.set_step(n_iter)

                        all_predictions = []
                        all_logits = []
                        all_labels = []
                        for (plain_train_iter, plain_train_label) in plain_train_dataloader:
                            with torch.no_grad():
                                plain_train_iter = plain_train_iter.to(device)
                                logits = classifier(plain_train_iter)
                                probs = torch.nn.Softmax(dim=1)(logits)
                                _, predictions = torch.max(probs.data, 1)
                                all_predictions.append(predictions)
                                all_logits.append(logits)
                                all_labels.append(plain_train_label.to(device))
                        training_evaluations(all_predictions, all_labels, all_logits, experiment, criterion, n_iter,
                                             plain_train_data.get_class_dict(), prefix='')
                    with experiment.test():
                        all_predictions = []
                        all_logits = []
                        all_labels = []
                        for (test_data, test_labels) in test_dataloader:
                            with torch.no_grad():
                                test_data = test_data.to(device)
                                logits = classifier(test_data)
                                probs = torch.nn.Softmax(dim=1)(logits)
                                _, predictions = torch.max(probs.data, 1)
                                all_predictions.append(predictions)
                                all_logits.append(logits)

                                all_labels.append(test_labels.to(device))
                        training_evaluations(all_predictions, all_labels, all_logits, experiment, criterion, n_iter,
                                             val_data.get_class_dict(), prefix='')

                    with experiment.validate():
                        all_predictions = []
                        all_logits = []
                        all_labels = []
                        for (val_data_iter, val_labels) in val_dataloader:
                            with torch.no_grad():
                                val_data_iter = val_data_iter.to(device)
                                logits = classifier(val_data_iter)
                                probs = torch.nn.Softmax(dim=1)(logits)
                                _, predictions = torch.max(probs.data, 1)
                                all_predictions.append(predictions)
                                all_logits.append(logits)
                                all_labels.append(val_labels.to(device))
                        training_evaluations(all_predictions, all_labels, all_logits, experiment, criterion, n_iter,
                                             val_data.get_class_dict(), prefix='')

                if n_iter % args['save_every'] == 0:
                    torch.save(classifier.state_dict(),
                               os.path.join(checkpoint_folder, f'classifier_iter_{n_iter}.pt'))

                n_iter += 1
                if (n_iter >= args['n_iterations'] + args["classifier_step"]):
                    break

            n_epochs += 1
            experiment.log_epoch_end(n_epochs)
            print(
                f'[{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}] Epoch: [{n_epochs}] Iteration: [{n_iter}|{args["n_iterations"] + args["classifier_step"]}]')


#
def training_evaluations(prediction_list, label_list, logits, experiment, criterion, n_iter, class_dict, prefix=''):
    all_predictions = torch.cat(prediction_list)
    all_labels = torch.cat(label_list)
    all_logits = torch.cat(logits)
    loss = criterion(all_logits, all_labels)

    all_predictions = all_predictions.cpu()
    all_labels = all_labels.cpu()
    all_logits = all_logits.cpu()
    all_probs = torch.nn.Softmax(dim=1)(all_logits).cpu().numpy()
    experiment.log_confusion_matrix(all_labels.numpy(), all_predictions.numpy(),
                                    file_name=f'{prefix}confusion_matrix_{n_iter}.json')
    conf_mat, _, _ = np.histogram2d(all_labels.numpy(), all_predictions.numpy(),
                                    bins=range(len(class_dict) + 1))
    conf_mat_col_sum = conf_mat.sum(0)
    conf_mat_row_sum = conf_mat.sum(1)

    binary_labels = preprocessing.label_binarize(all_labels, classes=range(args['num_classes']))
    multiclass_brier = np.mean(np.sum((all_probs - binary_labels) ** 2, axis=1))

    metrics = {}
    for i, key in enumerate(class_dict.keys()):
        metrics[f'{prefix}Recall {class_dict[key]}'] = conf_mat[i, i] / conf_mat_row_sum[i] if conf_mat_row_sum[
                                                                                                   i] != 0 else 0
        metrics[f'{prefix}Precision {class_dict[key]}'] = conf_mat[i, i] / conf_mat_col_sum[i] if conf_mat_col_sum[
                                                                                                      i] != 0 else 0
        metrics[f'{prefix}F1Score {class_dict[key]}'] = 2 * conf_mat[i, i] / (
                conf_mat_col_sum[i] + conf_mat_row_sum[i]) if (
                                                                      conf_mat_col_sum[
                                                                          i] +
                                                                      conf_mat_row_sum[
                                                                          i]) != 0 else 0

    metrics[f'{prefix}Multiclass_Brier'] = multiclass_brier
    metrics[f'{prefix}Accuracy'] = conf_mat.trace() / conf_mat.sum()
    metrics[f'{prefix}eval_loss'] = loss
    metrics[f'{prefix}seen_images'] = n_iter * all_labels.shape[0]

    experiment.log_metrics(metrics, step=n_iter)


def check_data(args, data_loader):
    print('Run check on data distribution')

    label = []
    origin = []

    for iteration in data_loader:
        label.extend(iteration[1])
        origin.extend(iteration[2])

    label = torch.tensor(label).numpy()
    origin = torch.tensor(origin).numpy()

    y, bins = np.histogram(label, bins=range(5))
    y_origin, bins_origin = np.histogram(origin, bins=range(3))

    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    axs[0].bar(bins[:-1], y, yerr=np.sqrt(y), label='Occurences')
    occ_from_real_data = np.array(data_loader.dataset.class_prob) * np.sum(y)
    axs[0].bar(bins[:-1], occ_from_real_data, yerr=np.sqrt(occ_from_real_data), ecolor='r', capsize=1, edgecolor='r',
               linewidth=2, fill=False, label='Real (scaled)')
    axs[0].set_xticks(range(args['num_classes']))
    axs[0].set_xticklabels(args["classes"])
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Number of images')
    axs[1].bar(bins_origin[:-1], y_origin,
               label=f'Desired ratio G/R = {data_loader.dataset.lambda_gen}\nis ratio: {float(y_origin[1]) / y_origin[0]}')
    axs[1].set_xticks(range(2))
    axs[1].set_xticklabels(['Real', 'Generated'])
    axs[1].set_xlabel('Origin')
    axs[1].set_ylabel('Number of images')
    for ax in axs:
        ax.legend()

    fig.suptitle(
        f'Number of real images: {len(data_loader.dataset.real_data)}, number of combined ds: {np.sum(y_origin)}, random seed: {metadata["random_seed"]}')
    plt.show()
    print('Done.')
    return fig


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Main tool to run classifier training using combined or real-only training data sets. Settings in config_file overwrite manual settings through keywords.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('n_iterations', type=int)
    parser.add_argument('lambda_gen', type=int),
    parser.add_argument('config_file', type=str,
                        default=os.path.join(project_folder, 'configs', 'Generator_xval_0_CNN_Histograms_paths.yaml'),
                        help='Path to json config file containing generator paths')
    parser.add_argument('--balancing_mode', type=str, choices=['realistic', 'balanced'],
                        help='Realistic generates images such that the combined data set has the same class imbalance as real data. Balanced generates them such that the combined data set is class balanced.')
    parser.add_argument('--bsize', type=int)
    parser.add_argument('--exp', type=str, help='used for the name in comet.ml and for the checkpoint save folder')
    parser.add_argument('--no_check_data', action='store_true', default=True)
    parser.add_argument('--evaluate_every', type=int,
                        help='starts the evaluation loop every x classifier training iterations')
    parser.add_argument('--save_every', type=int,
                        help='creates a checkpoint at every xth classifier training iteration')
    parser.add_argument('--classifier_restore', nargs=3,
                        help='{path to classifier checkpoint} {step_to_restore} {epoch_to_restore')
    parser.add_argument('--comet_experiment_key', type=str, help='API needed to communicate with comet.ml')
    parser.add_argument('--classifier', type=str, choices=['simplecnn', 'simplefcn'], default="simplecnn")
    parser.add_argument('--comet_project_name', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--classes', '--list', nargs='+', default=["FRI", "FRII", "Compact", "Bent"])

    parser.add_argument('--input_data_list', type=str, nargs="+", default=["galaxy_data_crossvalid_0_h5.h5"])
    parser.add_argument('--ngpus', type=int, default=1, help='Currently only 0 (cpu) or 1 (one gpu) supported.')
    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())

    args['classifier_path'] = args['classifier_restore'][0] if args['classifier_restore'] is not None else None
    args['classifier_step'] = int(args['classifier_restore'][1]) if args['classifier_restore'] is not None else 0
    args['classifier_epoch'] = int(args['classifier_restore'][2]) if args['classifier_restore'] is not None else 0
    del args['classifier_restore']

    args['num_classes'] = len(args["classes"])

    with open(os.path.join(project_folder, args['config_file']), 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    for k, v in config.items():
        if k not in args or args[k] is None:
            args[k] = v

    print('\n'.join([f'{k}: {v}' for k, v in args.items()]))
    print(f'RANDOM SEED: {metadata["random_seed"]}')
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args['ngpus'] > 0) else "cpu")

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5]),
         transforms.RandomRotation((-360, 360), fill=-1),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.CenterCrop([metadata["image_shape"][1], metadata["image_shape"][2]])
         ]
    )
    real_data = FIRSTGalaxyData(root=os.path.join(project_folder, "data/data_real"),
                                selected_classes=args["classes"], transform=train_transform, selected_split="train",
                                input_data_list=args['input_data_list'])

    if args['input_data_list'] is not None:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5]),
             transforms.RandomRotation((-360, 360), fill=-1),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.CenterCrop([metadata["image_shape"][1], metadata["image_shape"][2]])
             ]
        )
        real_data = FIRSTGalaxyData(root=os.path.join(project_folder, "data/data_real"),
                                    selected_classes=["FRI", "FRII", "Bent", "Compact"], transform=train_transform, selected_split="train",
                                    input_data_list=args['input_data_list'])

        ds = ConcatenatedDataSet(real_data, args['generator_paths'], args['lambda_gen'], real_data.get_class_dict(),
                                 balancing_mode=args['balancing_mode'], batch_size=args['bsize'], ngpus=args['ngpus'])

        train_dataloader = torch.utils.data.DataLoader(ds, batch_size=args['bsize'], shuffle=True,
                                                       num_workers=0)
        checkpoint_folder = os.path.join(project_folder,
                                         f'run/AugmentedClassifier',
                                         f'{args["comet_project_name"]}',
                                         f'{args["classifier"]}',
                                         f'{args["exp"]}',
                                         f'LG_{args["lambda_gen"]}')

    else:
        raise AttributeError('No training data set specified.')

    if not args['no_check_data']:
        fig = check_data(args, train_dataloader)

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    metadata = {**metadata, **args}
    with open(os.path.join(checkpoint_folder, '00_Metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f)

    if args['comet_experiment_key'] is None:
        experiment = comet_ml.Experiment(project_name=args['comet_project_name'],
                                         auto_metric_logging=False)
        experiment.log_parameters(metadata)
        classifier_name_conversion = {
            'simplefcn': 'FCN',
            'simplecnn': 'CNN'
        }
        try:
            if args['input_data_list'] is not None:
                xval = re.search('(?<=crossvalid_)[0-9]*', args['input_data_list'][0]).group(0)
                experiment.set_name(
                    f'{classifier_name_conversion[args["classifier"]]} XVal {xval} LG {args["lambda_gen"]}')

        except TypeError:
            xval = 'NaN'

    else:
        experiment = comet_ml.ExistingExperiment(previous_experiment=args['comet_experiment_key'],
                                                 auto_metric_logging=False)

    if not args['no_check_data']:
        experiment.log_figure('data_check', fig)

    plt.close('all')

    if args['classifier'] == 'simplecnn':
        classifier = SimpleCNN(n_output_nodes=args['num_classes']).to(device)
    elif args['classifier'] == 'simplefcn':
        classifier = SimpleFCN(metadata['image_shape'], n_output_nodes=args['num_classes']).to(device)
    else:
        raise ValueError(f'classifier {args["classifier"]} not found.')

    if args['classifier_path'] is not None:
        classifier.load_state_dict(torch.load(args['classifier_path'], map_location=device))
    if args['lambda_gen'] > 0:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        occurences = ds.occurrences

        inverse = 1. / occurences
        normalised_inverse = inverse / inverse.sum()
        criterion = torch.nn.CrossEntropyLoss(torch.tensor(normalised_inverse, dtype=torch.float, device=device)).to(
            device)
        print(f'Loss class weights: {normalised_inverse}.')

    optimiser = torch.optim.Adam(classifier.parameters(), lr=args['lr'])

    train(args, train_dataloader, classifier, criterion, optimiser, device, experiment, checkpoint_folder)

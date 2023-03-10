################
# This collection of methods is used to loop over a folder of saved wGAN training checkpoints and determine the best
# performing checkpoint according to the fixed set of histograms and their rMAE score.
################
import argparse
import itertools
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

DIRNAME = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(DIRNAME, '../../../'))

import src.GANs.wGAN.wGANModels as Models
from firstgalaxydata import FIRSTGalaxyData
import src.GANs.wGAN.ValidationImages as Validation_Images


def _get_generated_images(generator, bsize, n_gen_images, nz, device):
    tensor_opt = {'device': device, 'dtype': torch.float, 'requires_grad': False}
    onehot = torch.zeros(4, 4, **tensor_opt)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3]).view(4, 1).to(device), 1).view(4, 4, 1, 1)
    generated_images = {k: torch.Tensor() for k in range(4)}

    for label_ind in range(4):

        img_counter = 0
        while img_counter < n_gen_images:
            expected = img_counter + bsize
            if expected > n_gen_images:
                bsize = n_gen_images - img_counter
            with torch.no_grad():
                noise = torch.randn(bsize, nz, 1, 1, device=device, requires_grad=False)
                labels = torch.tensor([label_ind] * bsize, device=device, requires_grad=False)
                res = ((generator(noise, onehot[labels]) / 2 + .5) * 255).int().cpu()
                generated_images[label_ind] = res if generated_images[label_ind].shape[0] == 0 else torch.cat(
                    (generated_images[label_ind], res))
                img_counter += bsize

    return generated_images


def generate_histograms(checkpoint_df, n_generated_images, device, input_data_list, dir_to_save=None):
    metadata = {
        'image_shape': [1, 128, 128],
        'nz':          100,
        'ngf':         64,
        'ngpu':        1,
        'bsize':       250,
        'num_workers': 2
    }

    generator = Models.Generator(metadata['nz'], 1, metadata['ngf'], 4).to(device)

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5]),
         transforms.CenterCrop([metadata["image_shape"][1], metadata["image_shape"][2]])
         ]
    )

    train_data = FIRSTGalaxyData(root=os.path.join(DIRNAME, "../../../data/data_real"),
                                 selected_classes=["FRI", "FRII", "Bent", "Compact"],
                                 transform=train_transform,
                                 selected_split='train', input_data_list=input_data_list)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=9999, shuffle=False,
                                                   num_workers=metadata['num_workers'])

    val_obj = Validation_Images.ValidationImages('', train_dataloader,
                                                 do_test=False)
    classes = ['FRI', 'FRII', 'Compact', 'Bent']
    hist_types = ['intensities', 'sumi', 'activated']
    metrics = ['RMAE', 'KL', 'EMD', 'NEMD']
    keys = list(itertools.product(hist_types, classes, metrics))
    global_metrics = {k: [] for k in keys}
    global_metrics['iteration'] = []
    for row_ind, (iteration, row) in enumerate(checkpoint_df.iterrows()):
        print(f'checking iteration {iteration} [{row_ind} | {checkpoint_df.size}]')
        generator.load_state_dict(torch.load(row['path']))
        generated_images = _get_generated_images(generator, metadata['bsize'], n_generated_images, metadata['nz'],
                                                 device)

        res = val_obj.do_plots(generated_images)

        if dir_to_save is not None:
            curr_folder = os.path.join(dir_to_save, f'{iteration}')
            if not os.path.exists(curr_folder):
                os.makedirs(curr_folder)

            # Save example images..
            fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
            random_indices = [np.random.randint(0, len(generated_images[c]), size=4) for c in range(4)]
            for row, (c, ris) in enumerate(zip(range(4), random_indices)):
                for col, ri in enumerate(ris):
                    axs[row, col].matshow(np.transpose(generated_images[c][ri], (2, 1, 0)))
            for ax in axs.flatten():
                ax.axis('off')
            plt.tight_layout(pad=0, h_pad=0, w_pad=0)

            fig.savefig(os.path.join(curr_folder, 'example_images.svg'))

        for hist_type, dict_ in res.items():
            for class_, dict__ in dict_.items():
                if dir_to_save is not None:
                    dict__['figure'].savefig(os.path.join(curr_folder, f'{hist_type}_{class_}.svg'))

                for metric, value in dict__['metrics'].items():
                    global_metrics[(hist_type, class_, metric)].append(value)
        global_metrics['iteration'].append(iteration)

    res_df = pd.DataFrame(global_metrics).set_index('iteration')
    res_df.columns = pd.MultiIndex.from_tuples(res_df.columns)
    return res_df


def get_argparser():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dir_to_walk', type=str,
                        help='folder to look for checkpoints. Currently takes all files in that folder and tries to infer the corresp. iteration.'),
    parser.add_argument('--input_data_list', type=str, nargs="+",
                        help='Real data set for comparison (needs to be the corresponding X-Validation-fold)'),
    parser.add_argument('-N', '--num_generated_images', type=int, default=1000)
    parser.add_argument('-o', '--dir_to_save', type=str)
    return parser


if __name__ == '__main__':

    device = torch.device('cuda')
    classes = ['FRI', 'FRII', 'Compact', 'Bent']

    parser = get_argparser()
    args = vars(parser.parse_args())
    print(args)

    if not os.path.exists(args['dir_to_walk']):
        sys.exit()

    checkpoints = []
    for dirpath, dirnames, filenames in os.walk(args['dir_to_walk']):
        for filename in filenames:
            if 'generator' not in filename:
                continue
            iteration = int(re.search('[0-9]+(?=.pt)', filename).group(0))
            checkpoints.append((iteration, os.path.join(dirpath, filename)))

    checkpoint_df = pd.DataFrame(checkpoints, columns=['iteration', 'path']).set_index('iteration').sort_index()

    res_df = generate_histograms(checkpoint_df, args['num_generated_images'], device, args['input_data_list'],
                                 args['dir_to_save'])
    if args['dir_to_save'] is not None:
        with open(f'{args["dir_to_save"]}/metrics.pkl', 'wb') as f:
            pickle.dump(res_df, f)

    hist_aggregated_df = res_df.T.reset_index().groupby(['level_1', 'level_2']).mean().T

    fom = 'RMAE'
    minima = {class_:
                  hist_aggregated_df.index[hist_aggregated_df[(class_, fom)].argmin()] for class_ in classes
              }
    print(minima)
    string = """---
"generator_paths": {{
    0: {},
    1: {},
    2: {},
    3: {}
}}

"balancing_mode": "balanced"
"bsize": 250
"evaluate_every": 1000
"save_every": 1000
"lr": 0.0001
""".format(*[checkpoint_df.loc[v, 'path'] for v in minima.values()])
    print(string)
    if args['dir_to_save'] is not None:
        with open(os.path.join(args['dir_to_save'], 'generator_paths.yaml'), 'w') as f:
            f.write(string)

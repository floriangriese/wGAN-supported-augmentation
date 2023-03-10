import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.utils import download_url

file_folder = os.path.dirname(__file__)
project_folder = os.path.normpath(os.path.join(file_folder, "..", ".."))
sys.path.append(os.path.normpath(project_folder))
import src.GANs.wGAN.wGANModels as Models

metadata = {
    'image_shape': [1, 128, 128],
    'random_seed': np.random.randint(1, 9999),
    'nz':          100,
    'ngf':         64,
}


class ConcatenatedDataSet(torch.utils.data.Dataset):
    urls = {
        "generator_epoch_15000_iter_75001.pt": "https://zenodo.org/record/7713168/files/generator_epoch_15000_iter_75001.pt?download=1",
        "generator_epoch_19475_iter_97375.pt": "https://zenodo.org/record/7713168/files/generator_epoch_19475_iter_97375.pt?download=1",
        "generator_epoch_1150_iter_5750.pt":   "https://zenodo.org/record/7713168/files/generator_epoch_1150_iter_5750.pt?download=1",
        "generator_epoch_17575_iter_87875.pt": "https://zenodo.org/record/7713168/files/generator_epoch_17575_iter_87875.pt?download=1",

        "generator_epoch_16450_iter_82250.pt": "https://zenodo.org/record/7713168/files/generator_epoch_16450_iter_82250.pt?download=1",
        "generator_epoch_17175_iter_85875.pt": "https://zenodo.org/record/7713168/files/generator_epoch_17175_iter_85875.pt?download=1",
        "generator_epoch_700_iter_3500.pt":    "https://zenodo.org/record/7713168/files/generator_epoch_700_iter_3500.pt?download=1",
        "generator_epoch_18475_iter_92375.pt": "https://zenodo.org/record/7713168/files/generator_epoch_18475_iter_92375.pt?download=1",

        "generator_epoch_15575_iter_77875.pt": "https://zenodo.org/record/7713168/files/generator_epoch_15575_iter_77875.pt?download=1",
        "generator_epoch_18100_iter_90500.pt": "https://zenodo.org/record/7713168/files/generator_epoch_18100_iter_90500.pt?download=1",
        "generator_epoch_17125_iter_85625.pt": "https://zenodo.org/record/7713168/files/generator_epoch_17125_iter_85625.pt?download=1",
        "generator_epoch_17775_iter_88875.pt": "https://zenodo.org/record/7713168/files/generator_epoch_17775_iter_88875.pt?download=1",

        "generator_epoch_19650_iter_98250.pt": "https://zenodo.org/record/7713168/files/generator_epoch_19650_iter_98250.pt?download=1",
        "generator_epoch_19550_iter_97750.pt": "https://zenodo.org/record/7713168/files/generator_epoch_19550_iter_97750.pt?download=1",
        "generator_epoch_850_iter_4250.pt":    "https://zenodo.org/record/7713168/files/generator_epoch_850_iter_4250.pt?download=1",
        "generator_epoch_17725_iter_88625.pt": "https://zenodo.org/record/7713168/files/generator_epoch_17725_iter_88625.pt?download=1",

        "generator_epoch_7175_iter_35875.pt":  "https://zenodo.org/record/7713168/files/generator_epoch_7175_iter_35875.pt?download=1",
        "generator_epoch_15000_iter_75000.pt": "https://zenodo.org/record/7713168/files/generator_epoch_15000_iter_75000.pt?download=1",
        "generator_epoch_4800_iter_24000.pt":  "https://zenodo.org/record/7713168/files/generator_epoch_4800_iter_24000.pt?download=1",
        "generator_epoch_8525_iter_42625.pt":  "https://zenodo.org/record/7713168/files/generator_epoch_8525_iter_42625.pt?download=1",
    }

    def __init__(self, real_data, generator_paths, lambda_gen, class_dict, balancing_mode='realistic', batch_size=50,
                 is_RGB=False, transform=None, target_transform=None, is_return_real_gen_flag=True, ngpus=0):
        self.generator_paths = generator_paths
        if not self._check_files():
            print("Dataset not found. Trying to download...")
            self.download()
            if not self._check_files():
                raise RuntimeError(
                    "checkpoints not found or checkpoint corrupted or downloading failed. Check data paths or move files manually to checkpoints folder")

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpus > 0) else "cpu")
        self.generators = self.build_generators()
        self.batch_size = batch_size
        self.class_dict = class_dict
        self.num_classes = len(class_dict)
        self.real_counter = 0
        self.gen_counter = 0
        self.lambda_gen = lambda_gen
        self.balancing_mode = balancing_mode
        self.real_data = real_data

        self.occurrences = np.array(
            [self.real_data.get_occurrences()[i] for i in self.real_data.get_occurrences().keys()])
        self.class_prob = self.occurrences / self.occurrences.sum()
        self.is_RGB = is_RGB
        self.transform = transform
        self.target_transform = target_transform
        self.is_return_real_gen_flag = is_return_real_gen_flag

        if ngpus > 0 and not torch.cuda.is_available():
            print("Cuda not available but requested. Aborting.")
            sys.exit(1)

        if self.lambda_gen > 0:
            if self.balancing_mode == 'realistic':
                self.gen_prob = self.class_prob
                print(f'Class probabilities: {self.class_prob}')
            elif self.balancing_mode == 'balanced':
                R = self.occurrences.sum()
                z = np.array([(self.lambda_gen + 1) * R // self.num_classes] * self.num_classes)
                self.gen_prob = (z - (R * self.class_prob)) / (self.lambda_gen * R)
                self.gen_prob = self.gen_prob + (
                        1 - ((z - self.occurrences) / (self.lambda_gen * R)).sum()) / self.num_classes
                self.gen_prob = np.round(self.gen_prob, 5)

                # make sure it sums to 1 - counteract rounding in a dirty way
                if not np.sum(self.gen_prob) == 1:
                    residual = 1 - np.sum(self.gen_prob)
                    self.gen_prob += residual / self.gen_prob.size

            else:
                raise NotImplementedError()

            tensor_opt = {'device': self.device, 'dtype': torch.float}
            onehot = torch.zeros(4, 4, **tensor_opt)
            self.onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3]).view(4, 1).to(self.device), 1).view(4, 4, 1,
                                                                                                                1)
            self.renew_generated()
        else:
            self.gen_prob = None
            self.generators = None

    def build_generators(self):
        generators = {class_label: Models.Generator(metadata['nz'], metadata['image_shape'][0], metadata['ngf'], 4) for
                      class_label in self.generator_paths}
        for class_label, generator_path in self.generator_paths.items():
            generators[class_label].load_state_dict(
                torch.load(os.path.join(project_folder, "checkpoints", generator_path), map_location=self.device))
            generators[class_label].to(self.device)
            generators[class_label].eval()

        return generators

    def renew_generated(self):
        to_choose_from = list(self.class_dict.keys())
        labels = np.random.choice(to_choose_from, p=self.gen_prob, size=self.batch_size)

        generated_images = list()
        ordered_labels = list()
        for class_label in self.generators:
            with torch.no_grad():
                curr_labels = labels[labels == class_label]
                y_label = self.onehot[curr_labels]
                noise = torch.randn(curr_labels.size, metadata['nz'], 1, 1, device=self.device, requires_grad=False)
                generated_images.extend(self.generators[class_label](noise, y_label).tolist())
                ordered_labels.extend(curr_labels.tolist())
        randomised_indices = np.arange(0, labels.size)
        np.random.shuffle(randomised_indices)
        self.labels = np.array(ordered_labels)[randomised_indices].tolist()
        self.gen_image_batch = np.array(generated_images)[randomised_indices].tolist()

    def __getitem__(self, i):
        if self.lambda_gen == 0:
            image, label = self.real_data[i]
            if self.target_transform is not None:
                label = self.target_transform(label)
            if self.is_return_real_gen_flag:
                return image, label, 0
            else:
                return image, label
        else:

            if self.real_counter == 0:
                self.real_counter += 1
                try:
                    image, label = self.real_data[i // (self.lambda_gen + 1)]
                    if self.target_transform is not None:
                        label = self.target_transform(label)
                    if self.is_return_real_gen_flag:
                        return image, label, 0
                    else:
                        return image, label
                except IndexError as e:
                    print(i, len(self.real_data) * (self.lambda_gen + 1), self.lambda_gen)
                    raise e
            else:
                if self.gen_counter == self.lambda_gen:
                    self.real_counter = 0
                    self.gen_counter = 0
                    return self.__getitem__(i)

                else:
                    if len(self.gen_image_batch) == 0 or len(self.labels) == 0:
                        self.renew_generated()

                    if self.is_return_real_gen_flag:
                        image = torch.tensor(self.gen_image_batch.pop(0), dtype=torch.float32)
                    else:
                        image = np.array(self.gen_image_batch.pop(0))[0, :, :]

                    if self.is_RGB:
                        image = Image.fromarray(image, mode="L")
                        image = image.convert("RGB")
                    if self.transform is not None:
                        image = self.transform(image)
                    if not torch.is_tensor(image):
                        image = torch.tensor(image, dtype=torch.float32)
                    label = self.labels.pop(0)
                    if self.target_transform is not None:
                        label = self.target_transform(label)
                    self.gen_counter += 1
                    if self.is_return_real_gen_flag:
                        return image, label, 1
                    else:
                        return image, label

    def __len__(self):
        return len(self.real_data) * (self.lambda_gen + 1)

    def get_probabilities(self):
        if self.lambda_gen > 0:
            if self.balancing_mode == 'realistic':
                return self.class_prob
            elif self.balancing_mode == 'balanced':
                prob = [1.0 / len(self.real_data.get_occurrences()) for k in self.real_data.get_occurrences()]
                return prob
        elif self.lambda_gen == 0:
            return self.class_prob
        else:
            raise Exception("lambda_gen ratio invalid: {0}".format(self.lambda_gen))

    def _check_files(self):
        for data_file in self.generator_paths.values():
            path = os.path.join(project_folder, "checkpoints", data_file)
            if not os.path.exists(path):
                return False
        return True

    def download(self):
        # download file
        for key in self.urls.keys():
            path = os.path.join(project_folder, "checkpoints")
            Path(path).mkdir(parents=True, exist_ok=True)
            download_url(self.urls[key], path, key)

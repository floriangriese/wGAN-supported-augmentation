#######
# Example of how to generate a batch of images
#######
import sys

import torch

import src.GANs.wGAN.wGANModels as Models


def get_generated_images(generator, n_gen_images, nz, device):
    tensor_opt = {'device': device, 'dtype': torch.float, 'requires_grad': False}
    onehot = torch.zeros(4, 4, **tensor_opt)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3]).view(4, 1).to(device), 1).view(4, 4, 1, 1)

    generated_images = {}
    for label_ind in range(4):
        noise = torch.randn(n_gen_images, nz, 1, 1, device=device, requires_grad=False)
        labels = torch.tensor([label_ind] * n_gen_images, device=device, requires_grad=False)
        with torch.no_grad():
            generated_images[label_ind] = ((generator(noise, onehot[labels]) / 2 + .5) * 255).int().cpu()

    return generated_images


if __name__ == '__main__':
    '''
    Class labels are:
    0: FRI
    1: FRII
    2: Compact
    3: Bent
    '''

    metadata = {
        'ngpu':        0,
        # this is fixed by the training setup and thus by the weights
        'num_classes': 4,
        # Size of images final image dimensions including nc as first input
        'image_shape': [1, 128, 128],
        # Size of z latent vector (i.e. size of generator input)
        'nz':          100,
        # Size of feature maps in generator
        'ngf':         64
    }

    device = torch.device('cuda:0' if (torch.cuda.is_available() and metadata['ngpu'] > 0) else 'cpu')
    if metadata['ngpu'] > 0 and not torch.cuda.is_available():
        print('Cuda not available but requested. Aborting.')
        sys.exit(1)

    generator = Models.Generator(metadata['nz'], metadata['image_shape'][0], metadata['ngf'],
                                 metadata['num_classes']).to(device)
    print(generator)

    # generator.load_state_dict(torch.load('path_to_state_dict'))

    n_gen_images_per_label = 10

    images = get_generated_images(generator, n_gen_images_per_label, metadata['nz'], device)

    print(images)

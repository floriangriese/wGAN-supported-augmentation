import torch
from torch import autograd as autograd


def compute_gradient_penalty(critic, real_samples, fake_samples, y_fill_, device):
    n_elements = real_samples.nelement()
    batch_size = real_samples.size(0)
    image_shape = real_samples.shape[1:]
    alpha = torch.rand(batch_size, 1) \
        .expand(batch_size, int(n_elements / batch_size)) \
        .contiguous() \
        .view(batch_size, *image_shape) \
        .to(device)

    fake_data = fake_samples.view(batch_size, *image_shape)
    interpolates = alpha * real_samples.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    critic_interpolates = critic(interpolates, y_fill_)

    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

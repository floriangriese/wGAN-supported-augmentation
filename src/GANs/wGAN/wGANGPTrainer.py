import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

import src.GANs.wGAN.ValidationImages as ValidationImages
from src.GANs.wGAN.wGANUtils import compute_gradient_penalty


def train(experiment, num_iterations, dataloader, generator, critic, optimiser_D, optimiser_G, nz, n_critic,
          lambda_gp, device, img_shape, restored_iteration=0, restored_epoch=0, save_model_every=1,
          validate_model_every=0,
          validate_stats=100, checkpoint_folder='./'):
    tensor_opt = {'device': device, 'dtype': torch.float}
    print("Start of training loop...")

    val_obj = ValidationImages.ValidationImages(generator, dataloader,
                                                do_test=False) if validate_model_every > 0 else None
    with experiment.train():

        onehot = torch.zeros(4, 4, **tensor_opt)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3]).view(4, 1).to(device), 1).view(4, 4, 1, 1)
        fill = torch.zeros([4, 4, img_shape[1], img_shape[2]]).to(device)
        for i in range(4):
            fill[i, i, :, :] = 1

        gen_iter = 0
        step = 0 + restored_iteration
        epoch = 0 + restored_epoch
        while step < num_iterations + restored_iteration:
            for inner_iter, data in enumerate(dataloader, 0):
                # Train critic with all-real batch
                critic.zero_grad()
                generator.eval()
                real_sample = data[0].to(device)
                batch_size = real_sample.size(0)
                real_label = data[1].to(device)
                y_fill_ = fill[real_label]

                # Forward pass real batch through D
                crit_real = critic(real_sample, y_fill_).view(-1)

                # Train critic with all-fake batch
                noise = torch.randn(batch_size, nz, 1, 1, device=device, requires_grad=False)
                y_label_ = onehot[real_label]
                fake_sample = generator(noise, y_label_)

                # Classify all fake batch with D
                crit_fake = critic(fake_sample.detach(), y_fill_).view(-1)
                # Calculate D's loss on the all-fake batch
                gradient_penalty = compute_gradient_penalty(critic, real_sample, fake_sample, y_fill_, device)

                w_dist = torch.mean(crit_fake) - torch.mean(crit_real)
                critic_loss = w_dist + lambda_gp * gradient_penalty
                critic_loss.backward()
                optimiser_D.step()

                # Generator training
                if step % n_critic == 0 and step != 0:
                    generator.zero_grad()
                    generator.train()
                    gen_iter += 1
                    critic.eval()

                    noise = torch.randn(batch_size, nz, 1, 1, device=device, requires_grad=True)
                    fake_sample = generator(noise, y_label_)

                    generator_loss = -1 * critic(fake_sample, y_fill_).mean()
                    generator_loss.backward()
                    optimiser_G.step()
                    experiment.log_metric('critic_loss', critic_loss, step=step)
                    experiment.log_metric('generator_iteration', gen_iter, step=step)
                    print(
                        f'[{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}] Epoch: [{epoch + restored_epoch}] Iteration [{step + restored_iteration} | {num_iterations + restored_iteration}], \
Generator iteration: {gen_iter}, Critic loss: {critic_loss}')

                    critic.train()

                # Checkpoint creation
                if save_model_every > 0 and step % save_model_every == 0 and step != 0 + restored_iteration:
                    generator.eval()
                    noise = torch.randn(16, nz, 1, 1, device=device, requires_grad=False)
                    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4, device=device, requires_grad=False)
                    with torch.no_grad():
                        images = generator(noise, onehot[labels]).cpu()
                        images = (images / 2 + .5) * 255.
                        fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
                        for i, img in enumerate(images):
                            axs.flatten()[i].matshow(np.transpose(img, (2, 1, 0)))
                        for ax in axs.flatten():
                            ax.axis('off')
                        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
                        experiment.log_figure(figure=plt, figure_name="Generated Images", step=step)
                        fig.savefig(os.path.join(checkpoint_folder, f'Images_epoch_{epoch}_iter_{step}.svg'))

                    torch.save(generator.state_dict(),
                               os.path.join(checkpoint_folder, f'generator_epoch_{epoch}_iter_{step}.pt'))
                    torch.save(critic.state_dict(),
                               os.path.join(checkpoint_folder, f'critic_epoch_{epoch}_iter_{step}.pt'))

                if val_obj is not None and step % validate_model_every == 0 and step != 0 + restored_iteration:
                    print(f'Validating performance using {validate_stats} generated images.')
                    _ = val_obj.get_generated_images(validate_stats, nz, device)
                    plots, aggregations = val_obj.do_validations()
                    experiment.log_metrics(aggregations, step=step)
                    for plot_type, class_map in plots.items():
                        for class_name, values in class_map.items():
                            fig = values['figure']
                            experiment.log_figure(figure=fig, figure_name=f'{plot_type}_{class_name}', step=step)
                    plt.close('all')

                step += 1
                experiment.set_step(step)
                if step >= num_iterations + restored_iteration:
                    break

            epoch += 1
            experiment.log_epoch_end(epoch)

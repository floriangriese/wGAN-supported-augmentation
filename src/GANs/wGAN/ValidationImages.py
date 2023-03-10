import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy import stats
from uncertainties import unumpy

from firstgalaxydata import FIRSTGalaxyData


class ValidationImages(object):
    def __init__(self, generator, train_dataloader=None, test_dataloader=None, do_train=True, do_test=False,
                 data_root=None, batch_size=None, workers=None, do_uncerts=False):
        self.generator = generator
        self.do_uncerts = do_uncerts
        if do_train:
            if train_dataloader is not None:
                self.train_dataloader = train_dataloader
            else:
                self.train_dataloader = self.get_data(data_root, batch_size, workers, 'train')

            self.train_real_images = self.get_real_images(self.train_dataloader)

        if do_test:
            if test_dataloader is not None:
                self.test_dataloader = test_dataloader
            else:
                self.test_dataloader = self.get_data(data_root, batch_size, workers, 'test')

            self.test_real_images = self.get_real_images(self.test_dataloader)

        self.class_dict = self.train_dataloader.dataset.class_dict

    def get_data(self, data_root, batch_size, workers, split='train'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5]),
             transforms.CenterCrop([128, 128])
             ]
        )
        data = FIRSTGalaxyData(root=data_root, split=split, selected_classes=["FRI", "FRII", "Bent", "Compact"],
                               transform=transform)
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=workers)
        return dataloader

    def get_real_images(self, data):
        result = {k: [] for k in range(4)}
        for real_images, real_labels in data:
            real_images = np.round((real_images / 2 + 0.5) * 255)
            real_labels = real_labels.numpy()

            for img, label in zip(real_images, real_labels):
                result[label].append(img)

        for key, imgs in result.items():
            result[key] = torch.vstack(imgs).reshape(-1, 1, 128, 128)

        return result

    def get_generated_images(self, n_gen_images, nz, device, inplace=True):
        tensor_opt = {'device': device, 'dtype': torch.float, 'requires_grad': False}
        onehot = torch.zeros(4, 4, **tensor_opt)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3]).view(4, 1).to(device), 1).view(4, 4, 1, 1)

        generated_images = {}
        for label_ind in range(4):
            noise = torch.randn(n_gen_images, nz, 1, 1, device=device, requires_grad=False)
            labels = torch.tensor([label_ind] * n_gen_images, device=device, requires_grad=False)
            with torch.no_grad():
                generated_images[label_ind] = ((self.generator(noise, onehot[labels]) / 2 + .5) * 255).int().cpu()

        if inplace:
            self.generated_images = generated_images
        else:
            return generated_images

    def do_plots(self, generated_images=None, real_images=None, intensities=True, activated=True,
                 sumi=True, train=True):
        if generated_images is None:
            if not hasattr(self, 'generated_images'):
                raise ValueError('either pass set of generated images or call "get_generated_images" first.')
            generated_images = self.generated_images
        if real_images is None and hasattr(self, 'train_real_images'):
            real_images = self.train_real_images if train else self.test_real_images

        images = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print('Warnings wilfully ignored for plotting due to invalid true divisions.')
            if intensities:
                images['intensities'] = self.pixel_intensity(generated_images, real_images)
            if activated:
                images['activated'] = self.activated_pixels(generated_images, real_images)
            if sumi:
                images['sumi'] = self.intensity_sum(generated_images, real_images)

        return images

    def do_validations(self, plots=None):
        if plots is None:
            plots = self.do_plots()

        for plot_type, class_map in plots.items():
            for class_name, values in class_map.items():
                plots[plot_type][class_name]['rMAE'] = RMAE(values['generated'], values['real'])

        # Compute adapted rMAE for each plot type aggregated over all classes.
        aggs = {}
        for plot_type, class_map in plots.items():
            summe = np.sum([x['rMAE'] for x in class_map.values()])
            aggs[plot_type] = summe  # / bincount

        # Overall
        aggs['overall'] = np.sum([x for x in aggs.values()])
        return plots, aggs

    def _count_normalisation(self, gcounts, rcounts, bin_widths):
        gcounts = gcounts / bin_widths
        rcounts = rcounts / bin_widths

        gsum, rsum = gcounts.sum(), rcounts.sum()
        gcounts = gcounts / gsum if gsum != 0 else gcounts
        rcounts = rcounts / rsum if rsum != 0 else rcounts
        return gcounts, rcounts

    def draw_ratio_plot(self, ax, plot_bins, gcounts, rcounts, xlabel, ylabel=None, ylims=[-1, 1], draw_outliers=True):
        ratio = 2 * (rcounts - gcounts) / (rcounts + gcounts)
        ratio[np.invert(np.isfinite(ratio))] = 0

        bin_widths = np.diff(plot_bins)
        bin_centres = plot_bins[:-1] + bin_widths / 2.
        ax.bar(bin_centres, ratio, bin_widths)
        ax.set_ylim(ylims)

        if draw_outliers:
            outlier_indices = np.logical_or(ratio < ylims[0], ratio > ylims[1])
            outlier_centres = bin_centres[outlier_indices]
            outlier_values = ratio[outlier_indices]
            for outlier_centre, outlier_value in zip(outlier_centres, outlier_values):
                if outlier_value < ylims[0]:
                    ylim = ylims[0]
                    y_text = ylim + .5
                else:
                    ylim = ylims[1]
                    y_text = ylim - .5
                if len(outlier_centres) <= 30:
                    ax.annotate(f'{outlier_value:.2f}', (outlier_centre, ylim), xytext=(outlier_centre, y_text),
                                arrowprops=dict(arrowstyle="->",
                                                connectionstyle="arc3"), ha='center',
                                bbox=dict(boxstyle='round', fc='white', alpha=.8))
                else:
                    ax.arrow(outlier_centre, y_text, 0, -.25 if outlier_value < ylims[0] else .25)

        ax.grid()
        ax.set_ylabel(
            r'$2\times\frac{\mathrm{Real} - \mathrm{Gen}}{\mathrm{Real} + \mathrm{Gen}}$' if ylabel is None else ylabel)
        ax.set_xlabel(xlabel, ha='right', x=1)

    def _plot(self, plot_bins, bin_centres, gcounts, rcounts, n_gen_images, n_real_images, ylim, class_name, xlabel,
              ylabel='Frequency', gerr=None, rerr=None):

        metrics = self.get_metrics(bin_centres,
                                   gcounts,
                                   rcounts)

        fig, axs = plt.subplots(2, 1, figsize=(12.8, 8), gridspec_kw={'height_ratios': [.8, .2]})
        hist_w_errorbars(axs[0], gcounts, plot_bins, uncert=gerr, do_uncert=self.do_uncerts,
                         label=f'Generated (n = {n_gen_images})', fill=True, fill_alpha=.4)
        hist_w_errorbars(axs[0], rcounts, plot_bins, uncert=rerr, do_uncert=self.do_uncerts,
                         label=f'Real (n = {n_real_images})')

        axs[0].plot(' ', ' ', label=f'RMAE: {metrics["RMAE"]:.3f}')
        axs[0].set_yscale('log')
        axs[0].set_ylim(ylim)
        axs[0].legend()
        axs[0].set_ylabel(ylabel, y=1, ha='right')
        axs[0].grid()
        # axs[0].set_title(class_name)

        self.draw_ratio_plot(axs[1], plot_bins, gcounts, rcounts, xlabel=xlabel)

        plt.tight_layout()
        return {'figure':    fig,
                'binning':   plot_bins,
                'generated': gcounts,
                'real':      rcounts,
                'metrics':   self.get_metrics(bin_centres[1:-1],
                                              gcounts[1:-1],
                                              rcounts[1:-1])
                }

    def pixel_intensity(self, generated_images, real_images):
        np_bins, plot_bins, bin_widths, bin_centres = generate_under_and_overflow_binning(range(257),
                                                                                          mode='mean')

        images = {}

        for label_ind in range(4):
            class_name = self.class_dict[label_ind]
            selected_gen_images = generated_images[label_ind]
            selected_real_images = real_images[label_ind]

            gcounts, _ = np.histogram(selected_gen_images, bins=np_bins)
            rcounts, _ = np.histogram(selected_real_images, bins=np_bins)
            gerr = np.sqrt(gcounts)
            rerr = np.sqrt(rcounts)
            gcounts = unumpy.uarray(gcounts, gerr)
            rcounts = unumpy.uarray(rcounts, rerr)

            gcounts, rcounts = self._count_normalisation(gcounts, rcounts, bin_widths)
            images[class_name] = self._plot(plot_bins, bin_centres, unumpy.nominal_values(gcounts),
                                            unumpy.nominal_values(rcounts), selected_gen_images.shape[0],
                                            selected_real_images.shape[0], (1e-7, 1.1), class_name, 'Pixel Intensity',
                                            gerr=unumpy.std_devs(gcounts), rerr=unumpy.std_devs(rcounts))

        return images

    def activated_pixels(self, generated_images, real_images, thresh=1):
        binning = np.linspace(0, 500, 10, endpoint=True).tolist() + [600, 700, 800, 900, 1000, 2000]
        np_bins, plot_bins, bin_widths, bin_centres = generate_under_and_overflow_binning(binning, 'mean')

        images = {}

        for label_ind in range(4):
            class_name = self.class_dict[label_ind]
            selected_gen_images = generated_images[label_ind]
            selected_real_images = real_images[label_ind]

            gactivated = (selected_gen_images >= thresh).sum((2, 3))
            gcounts, _ = np.histogram(gactivated, bins=np_bins)

            ractivated = (selected_real_images >= thresh).sum((2, 3))
            rcounts, _ = np.histogram(ractivated, bins=np_bins)

            if self.do_uncerts:
                gerr = np.sqrt(gcounts)
                rerr = np.sqrt(rcounts)
                gcounts = unumpy.uarray(gcounts, gerr)
                rcounts = unumpy.uarray(rcounts, rerr)
            else:
                gcounts = unumpy.uarray(gcounts, 0)
                rcounts = unumpy.uarray(rcounts, 0)

            gcounts, rcounts = self._count_normalisation(gcounts, rcounts, bin_widths)

            images[class_name] = self._plot(plot_bins, bin_centres, unumpy.nominal_values(gcounts),
                                            unumpy.nominal_values(rcounts), selected_gen_images.shape[0],
                                            selected_real_images.shape[0], (1e-5, 1.1), class_name,
                                            'Number of Pixels w/ $\\mathtt{I}>0$', gerr=unumpy.std_devs(gcounts),
                                            rerr=unumpy.std_devs(rcounts))
        return images

    def intensity_sum(self, generated_images, real_images):
        binning = np.linspace(0, 25e3, 15, endpoint=True).tolist() + [50000, 75000]
        np_bins, plot_bins, bin_widths, bin_centres = generate_under_and_overflow_binning(binning, mode='mean')

        images = {}
        for label_ind in range(4):
            class_name = self.class_dict[label_ind]

            selected_gen_images = generated_images[label_ind]
            selected_real_images = real_images[label_ind]

            gactivated = selected_gen_images.sum((2, 3))
            gcounts, _ = np.histogram(gactivated, bins=np_bins)

            ractivated = selected_real_images.sum((2, 3))
            rcounts, _ = np.histogram(ractivated, bins=np_bins)
            gerr = np.sqrt(gcounts)
            rerr = np.sqrt(rcounts)
            gcounts = unumpy.uarray(gcounts, gerr)
            rcounts = unumpy.uarray(rcounts, rerr)

            gcounts, rcounts = self._count_normalisation(gcounts, rcounts, bin_widths)
            images[class_name] = self._plot(plot_bins, bin_centres, unumpy.nominal_values(gcounts),
                                            unumpy.nominal_values(rcounts), selected_gen_images.shape[0],
                                            selected_real_images.shape[0], (1e-5, 1.1), class_name, 'Intensity Sum',
                                            gerr=unumpy.std_devs(gcounts), rerr=unumpy.std_devs(rcounts))

        return images

    def get_metrics(self, bin_centres, gen_hist_count, real_hist_count):
        '''

        Args:
            bin_centres:
            gen_hist_count:
            real_hist_count:

        Returns:

        '''
        emd = EMD(bin_centres, real_hist_count, gen_hist_count)
        nemd = emd / bin_centres[-1]
        return {'RMAE': RMAE(real_hist_count, gen_hist_count),
                'EMD':  emd,
                'NEMD': nemd,
                'KL':   stats.entropy(real_hist_count, gen_hist_count)
                }


def RMAE(a, b):
    # Slightly adapted rMAE, where division is done by a + b so that also a difference is reported where one of the two is zero.
    mask = a + b != 0
    return np.mean(2 * np.abs((a[mask] - b[mask]) / (a[mask] + b[mask])))


def EMD(bin_centres, a, b):
    '''
    Earth movers distance from scipy
    Args:
        a:
        b:

    Returns:

    '''
    try:
        return stats.wasserstein_distance(bin_centres, bin_centres, a, b)
    except ValueError as e:
        print(e)
        return np.nan


def generate_under_and_overflow_binning(binning, mode='unit width'):
    if isinstance(binning, list):
        bin_list = binning
    elif isinstance(binning, range):
        bin_list = list(binning)
    elif isinstance(binning, np.ndarray):
        bin_list = binning.tolist()
    else:
        raise NotImplementedError(f"the type of your binning is {type(binning)}")

    bin_widths = np.diff(binning)
    numpy_binning = np.array([-np.inf] + bin_list + [np.inf])

    if mode == 'unit width':
        plot_binning = np.array([bin_list[0] - 1] + bin_list + [bin_list[-1] + 1])
    elif mode == 'mean':
        mean_binning = np.mean(bin_widths)  # This supposes that binning is monotonously increasing
        plot_binning = np.array([bin_list[0] - mean_binning] + bin_list + [bin_list[-1] + mean_binning])
    elif mode == 'neighbour':
        plot_binning = np.array([bin_list[0] - bin_widths[0]] + bin_list + [bin_list[-1] + bin_widths[-1]])

    bin_widths = np.diff(plot_binning)
    bin_centres = plot_binning[:-1] + bin_widths / 2.

    return numpy_binning, plot_binning, bin_widths, bin_centres


def hist_w_errorbars(ax, hy, hx, uncert=None, label=None, do_uncert=True, line_kw=None, error_kw=None, fill=False,
                     fill_alpha=1.):
    """

    :type ax: matplotlib.axes._subplots.AxesSubplot

    """
    if line_kw is None:
        line_kw = {}
    if error_kw is None:
        error_kw = {
            'capsize':    5,
            'capthick':   1,
            'markersize': 0
        }

    h1_widths = np.diff(hx) * .5
    h1_centres = h1_widths + hx[:-1]

    if not do_uncert:
        if line_kw is None:
            line_kw = {'label': label}
        else:
            line_kw['label'] = label

    line = ax.step(hx, [hy[0]] + hy.tolist(), where="pre", **line_kw)

    if fill:
        fill_obj = ax.fill_between(hx, [hy[0]] + hy.tolist(), step="pre", alpha=fill_alpha, color=line[-1].get_color())

    if not "c" in list(error_kw.keys()):
        error_kw["c"] = line[-1].get_color()
    if not "fmt" in list(error_kw.keys()):
        error_kw["fmt"] = "o"

    if uncert is None and do_uncert:
        uncert = np.sqrt(hy)

    if label is not None:
        error_kw["label"] = label
    else:
        if "label" in list(error_kw.keys()):
            del error_kw["label"]

    if do_uncert:
        err = ax.errorbar(h1_centres, hy, yerr=uncert, **error_kw)

    y_min, y_max = ax.get_ylim()
    upper = hy + uncert if uncert is not None else hy

    if np.max(upper) > y_max:
        y_max = np.max(upper) * 1.1

    ax.vlines(hx[0], y_min, hy[0], color=line[-1].get_color())
    ax.vlines(hx[-1], y_min, hy[-1], color=line[-1].get_color())
    ax.set_ylim([y_min, y_max])

    if do_uncert:
        return line, err
    else:
        return line


if __name__ == '__main__':
    raise NotImplementedError(f'Not to be used as main file.')

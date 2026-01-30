from dz_lib.univariate.data import Sample
from dz_lib.utils import fonts, encode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Distribution:
    def __init__(self, name, x_values, y_values):
        self.name = name
        self.x_values = x_values
        self.y_values = y_values

    def subset(self, x_min: float, x_max: float):
        mask = (self.x_values > x_min) & (self.x_values < x_max)
        new_y_vals = np.where(mask, self.y_values, 0)
        return Distribution(self.name, self.x_values, new_y_vals)


# ----------------------------------------------------------------------
# Peak detection (guaranteed, KDE-safe)
# ----------------------------------------------------------------------
def find_peaks(x: np.ndarray, y: np.ndarray, n_peaks: int):
    """
    Find up to n_peaks peaks in a smooth distribution.
    Always includes the global maximum.
    """
    if n_peaks <= 0:
        return []

    peaks = []

    max_idx = np.argmax(y)
    peaks.append((x[max_idx], y[max_idx]))

    if n_peaks == 1:
        return peaks

    dy = np.diff(y)
    signs = np.sign(dy)

    candidates = np.where(
        (signs[:-1] >= 0) & (signs[1:] <= 0)
    )[0] + 1

    candidates = [i for i in candidates if i != max_idx]
    candidates = sorted(candidates, key=lambda i: y[i], reverse=True)

    for i in candidates:
        peaks.append((x[i], y[i]))
        if len(peaks) >= n_peaks:
            break

    return peaks


def find_modes(x: np.ndarray, y: np.ndarray, n_modes: int):
    """
    Robust mode finder for smooth KDE/PDP curves.
    Always includes the global maximum.
    """
    modes = [(x[np.argmax(y)], np.max(y))]

    if n_modes == 1:
        return modes

    dy = np.diff(y)
    signs = np.sign(dy)

    candidates = np.where(
        (signs[:-1] >= 0) & (signs[1:] <= 0)
    )[0] + 1

    candidates = [i for i in candidates if i != np.argmax(y)]
    candidates = sorted(candidates, key=lambda i: y[i], reverse=True)

    for i in candidates:
        modes.append((x[i], y[i]))
        if len(modes) >= n_modes:
            break

    return modes


# ----------------------------------------------------------------------
# Distribution generators
# ----------------------------------------------------------------------
def kde_function(
    sample: Sample,
    bandwidth: float = 10,
    x_min: float = 0,
    x_max: float = 4500,
):
    n_steps = 10 * int(x_max - x_min + 1)
    x_values = np.linspace(x_min, x_max, n_steps)

    ages = np.array([grain.age for grain in sample.grains])

    ages_2d = ages[:, np.newaxis]
    x_2d = x_values[np.newaxis, :]

    diff_squared = (x_2d - ages_2d) ** 2
    variance_2 = 2 * bandwidth ** 2
    normalization = 1.0 / (np.sqrt(2 * np.pi) * bandwidth)

    kernels = normalization * np.exp(-diff_squared / variance_2)
    y_values = np.sum(kernels, axis=0)

    y_values /= np.sum(y_values)
    return Distribution(sample.name, x_values, y_values)


def pdp_function(sample: Sample, x_min: float = 0, x_max: float = 4500):
    n_steps = 10 * int(x_max - x_min + 1)
    x_values = np.linspace(x_min, x_max, n_steps)

    ages = np.array([grain.age for grain in sample.grains])
    bandwidths = np.array([grain.uncertainty for grain in sample.grains])

    ages_2d = ages[:, np.newaxis]
    bandwidths_2d = bandwidths[:, np.newaxis]
    x_2d = x_values[np.newaxis, :]

    diff_squared = (x_2d - ages_2d) ** 2
    variance_2 = 2 * bandwidths_2d ** 2
    normalization = 1.0 / (np.sqrt(2 * np.pi) * bandwidths_2d)

    kernels = normalization * np.exp(-diff_squared / variance_2)
    y_values = np.sum(kernels, axis=0)

    y_values /= np.sum(y_values)
    return Distribution(sample.name, x_values, y_values)


def cdf_function(distribution: Distribution):
    cdf = np.cumsum(distribution.y_values)
    cdf /= cdf[-1]
    return Distribution(distribution.name, distribution.x_values, cdf)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def distribution_graph(
    distributions: list,
    x_min: float = 0,
    x_max: float = 4500,
    stacked: bool = False,
    legend: bool = True,
    title: str = None,
    font_path: str = None,
    font_size: float = 12,
    fig_width: float = 9,
    fig_height: float = 7,
    color_map: str = 'plasma',
    modes_labeled: int = 0,
    fill: bool = False,
):
    num_samples = len(distributions)
    colors = plt.cm.get_cmap(color_map, num_samples)(
        np.linspace(0, 1, num_samples)
    )

    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()

    if not stacked:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100, squeeze=False)
        ax_list = [ax[0, 0]]

        for i, dist in enumerate(distributions):
            x, y = dist.x_values, dist.y_values

            ax[0, 0].plot(x, y, label=dist.name, color=colors[i])

            if fill:
                ax[0, 0].fill_between(
                    x, y, 0,
                    color=colors[i],
                    alpha=0.25,
                    linewidth=0
                )

            if modes_labeled > 0:
                modes = find_modes(x, y, modes_labeled)
                for px, py in modes:
                    ax[0, 0].annotate(
                        f'{px:.0f}',
                        xy=(px, py),
                        xytext=(25, -12),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=font_size * 0.85,
                        zorder=5,
                        clip_on=False
                    )

        if legend:
            ax[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    else:
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        gs = gridspec.GridSpec(len(distributions), 1, figure=fig)
        ax_list = []

        for i, dist in enumerate(distributions):
            ax = fig.add_subplot(gs[i])
            ax_list.append(ax)

            x, y = dist.x_values, dist.y_values
            ax.plot(x, y, label=dist.name, color=colors[i])

            if fill:
                ax.fill_between(
                    x, y, 0,
                    color=colors[i],
                    alpha=0.25,
                    linewidth=0
                )

            if modes_labeled > 0:
                modes = find_modes(x, y, modes_labeled)
                for px, py in modes:
                    ax.annotate(
                        f'{px:.0f}',
                        xy=(px, py),
                        xytext=(25, -12),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=font_size * 0.85,
                        zorder=5,
                        clip_on=False
                    )

            if legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    for i, ax in enumerate(ax_list):
        ax.set_xlim(x_min, x_max)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

        if stacked and i < len(ax_list) - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            plt.setp(ax.get_xticklabels(), fontproperties=font)

        plt.setp(ax.get_yticklabels(), fontproperties=font)

        ax.set_facecolor('white')
        ax.tick_params(axis='x', colors='black', width=2)
        ax.tick_params(axis='y', colors='black', width=2)

    fig.suptitle(title, fontsize=font_size * 1.75, fontproperties=font)
    fig.text(0.5, 0.02, 'Age (Ma)', ha='center', va='center',
             fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Probability Differential', va='center',
             rotation='vertical', fontsize=font_size, fontproperties=font)

    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    plt.xlim(x_min, x_max)

    plt.close()
    return fig

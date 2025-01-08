from dz_lib.univariate.data import Sample
from dz_lib.utils import fonts, encode
import numpy as np
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self, name, x_values, y_values):
        self.name = name
        self.x_values = x_values
        self.y_values = y_values


def kde_function(sample: Sample, bandwidth: float = 10, x_min: float=0, x_max: float=4500, n_steps: int = 1000):
    kde_sample = sample.replace_grain_uncertainties(bandwidth)
    distro = pdp_function(kde_sample, x_min=x_min, x_max=x_max, n_steps=n_steps)
    x_values = distro.x_values
    y_values = distro.y_values
    return Distribution(distro.name, x_values, y_values)


def pdp_function(sample: Sample, x_min: float=0, x_max: float=4500, n_steps: int = 1000):
    x_values = np.linspace(x_min, x_max, n_steps)
    y_values = np.zeros_like(x_values)
    ages = [grain.age for grain in sample.grains]
    bandwidths = [grain.uncertainty for grain in sample.grains]
    for i in range(len(ages)):
        kernel_sum = np.zeros(n_steps)
        s = bandwidths[i]
        kernel_sum += (1.0 / (np.sqrt(2 * np.pi) * s)) * np.exp(-(x_values - float(ages[i])) ** 2 / (2 * float(s) ** 2))
        y_values += kernel_sum
    y_values /= np.sum(y_values)
    return Distribution(sample.name, x_values, y_values)


def cdf_function(distribution: Distribution):
    x_values = distribution.x_values
    y_values = distribution.y_values
    name = distribution.name
    cdf_values = np.cumsum(y_values)
    cdf_values = cdf_values / cdf_values[-1]
    return Distribution(name, x_values, cdf_values)


def get_x_min(sample: Sample):
    sorted_grains = sorted(sample.grains, key=lambda grain: grain.age)
    return sorted_grains[0].age - sorted_grains[0].uncertainty


def get_x_max(sample: Sample):
    sorted_grains = sorted(sample.grains, key=lambda grain: grain.age)
    return sorted_grains[-1].age + sorted_grains[-1].uncertainty


def distribution_graph(
        distributions: [Distribution],
        stacked: bool = False,
        legend: bool = True,
        title: str = None,
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
        color_map='plasma'
):
    num_samples = len(distributions)
    colors_map = plt.cm.get_cmap(color_map, num_samples)
    colors = colors_map(np.linspace(0, 1, num_samples))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    if not stacked:
        for i, distribution in enumerate(distributions):
            header = distribution.name
            x = distribution.x_values
            y = distribution.y_values
            ax.plot(x, y, label=header, color=colors[i])
            if legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        fig, ax = plt.subplots(nrows=len(distributions), figsize=(fig_width, fig_height), dpi=100, squeeze=False)
        for i, distribution in enumerate(distributions):
            header = distribution.name
            x = distribution.x_values
            y = distribution.y_values
            ax[i, 0].plot(x, y, label=header)
            if legend:
                ax[i, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()
    title_size = font.get_size() * 2  # Adjust title size
    fig.suptitle(title, fontsize=title_size, fontproperties=font)
    fig.text(0.5, 0.01, 'Age (Ma)', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Probability Differential', va='center', rotation='vertical', fontsize=font_size,
             fontproperties=font)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    plt.close()
    return fig

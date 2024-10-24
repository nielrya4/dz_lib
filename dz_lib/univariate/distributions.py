from dz_lib.univariate.data import Sample
from dz_lib.utils import fonts, encode
import numpy as np
import matplotlib.pyplot as plt

class Distribution:
    def __init__(self, name, x_values, y_values):
        self.name = name
        self.x_values = x_values
        self.y_values = y_values

def kernel_density_estimate(sample: Sample, bandwidth: float = 10, n_steps: int = 1000):
    kde_sample = sample.replace_grain_uncertainties(bandwidth)
    x_values, y_values = probability_density_function(kde_sample, n_steps=n_steps)
    return x_values, y_values

def probability_density_function(sample: Sample, n_steps: int = 1000):
    x_min = get_x_min(sample)
    x_max = get_x_max(sample)
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
    return x_values, y_values

def cumulative_distribution_function(x_values: [float], y_values: [float]):
    cdf_values = np.cumsum(y_values)
    cdf_values = cdf_values / cdf_values[-1]
    return x_values, cdf_values

def get_x_min(sample: Sample):
    sorted_grains = sorted(sample.grains, key=lambda grain: grain.age)
    return sorted_grains[0].age - sorted_grains[0].uncertainty

def get_x_max(sample: Sample):
    sorted_grains = sorted(sample.grains, key=lambda grain: grain.age)
    return sorted_grains[-1].age + sorted_grains[-1].uncertainty

def distribution_graph(
    distributions: [Distribution],
    stacked: bool=False,
    legend: bool=True,
    title: str = "Distribution Function",
    output_format: str = 'svg',
    font_path: str = fonts.get_default_font().get_name,
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
        if len(distributions) == 1:
            fig, ax = plt.subplots(nrows=1, figsize=(fig_width, fig_height), dpi=100, squeeze=False)
            for i, distribution in enumerate(distributions):
                header = distribution.name
                x = distribution.x_values
                y = distribution.y_values
                ax[0, 0].plot(x, y, label=header)
                if legend:
                    ax[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            fig, ax = plt.subplots(nrows=len(distributions), figsize=(fig_width, fig_height), dpi=100, squeeze=False)
            for i, distribution in enumerate(distributions):
                header = distribution.name
                x = distribution.x_values
                y = distribution.y_values
                ax[i, 0].plot(x, y, label=header)
                if legend:
                    ax[i, 0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    font = fonts.get_font(font_path)
    title_size = font.size * 2
    fig.suptitle(title, fontsize=title_size, fontproperties=font)
    fig.text(0.5, 0.01, 'Age (Ma)', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Probability Differential', va='center', rotation='vertical', fontsize=font_size, fontproperties=font)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    if output_format == "html":
        return_str = encode.fig_to_html(fig, fig_type="plotly")
    else:
        return_str = encode.buffer_to_utf8(encode.fig_to_img_buffer(fig, fig_type="plotly", img_format=output_format))
    return return_str

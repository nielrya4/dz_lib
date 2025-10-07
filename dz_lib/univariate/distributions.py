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

def kde_function(sample: Sample, bandwidth: float = 10, x_min: float=0, x_max: float=4500, n_steps: int = 1000):
    n_steps = 10*int(x_max - x_min + 1)
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

def pdp_function(sample: Sample, x_min: float=0, x_max: float=4500):
    n_steps = 10*int(x_max - x_min + 1)
    x_values = np.linspace(x_min, x_max, n_steps)
    y_values = np.zeros_like(x_values)
    
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
    x_values = distribution.x_values
    y_values = distribution.y_values
    name = distribution.name
    cdf_values = np.cumsum(y_values)
    cdf_values = cdf_values / cdf_values[-1]
    return Distribution(name, x_values, cdf_values)


def get_x_min(sample: Sample):
    ages = np.array([grain.age for grain in sample.grains])
    uncertainties = np.array([grain.uncertainty for grain in sample.grains])
    min_idx = np.argmin(ages)
    return ages[min_idx] - uncertainties[min_idx]


def get_x_max(sample: Sample):
    ages = np.array([grain.age for grain in sample.grains])
    uncertainties = np.array([grain.uncertainty for grain in sample.grains])
    max_idx = np.argmax(ages)
    return ages[max_idx] + uncertainties[max_idx]


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
        color_map='plasma'
):
    num_samples = len(distributions)
    colors_map = plt.cm.get_cmap(color_map, num_samples)
    colors = colors_map(np.linspace(0, 1, num_samples))

    if not stacked:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100, squeeze=False)
        for i, distribution in enumerate(distributions):
            header = distribution.name
            x = distribution.x_values
            y = distribution.y_values
            ax[0, 0].plot(x, y, label=header, color=colors[i])
        if legend:
            ax[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)
        ax_list = [ax[0, 0]]
    else:
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        gs = gridspec.GridSpec(len(distributions), 1, figure=fig, height_ratios=[1] * len(distributions))
        ax_list = []
        for i, distribution in enumerate(distributions):
            ax = fig.add_subplot(gs[i])
            ax_list.append(ax)
            header = distribution.name
            x = distribution.x_values
            y = distribution.y_values
            ax.plot(x, y, label=header)
            if legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=font_size)

    # Set font
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()

    for i, ax in enumerate(ax_list):
        ax.set_xlim(x_min, x_max)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        
        # Only show x-tick labels on the bottommost subplot when stacked
        if stacked and i < len(ax_list) - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            plt.setp(ax.get_xticklabels(), fontsize=font_size, fontproperties=font)
        
        plt.setp(ax.get_yticklabels(), fontsize=font_size, fontproperties=font)
        
        # Ensure clean background and proper tick visibility
        ax.set_facecolor('white')
        ax.tick_params(axis='x', colors='black', width=2)
        ax.tick_params(axis='y', colors='black', width=2)

    title_size = font_size * 1.75
    fig.suptitle(title, fontsize=title_size, fontproperties=font)
    fig.text(0.5, 0.02, 'Age (Ma)', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Probability Differential', va='center', rotation='vertical',
             fontsize=font_size, fontproperties=font)
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    plt.xlim(x_min, x_max)

    plt.close()
    return fig
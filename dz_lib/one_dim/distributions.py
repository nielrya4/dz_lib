from dz_lib.one_dim.data import Sample
import numpy as np

def kernel_density_estimate(sample: Sample,
                            bandwidth: float = 10,
                            n_steps: int = 1000):
    kde_sample = sample.replace_grain_uncertainties(bandwidth)
    x_values, y_values = probability_density_function(kde_sample, n_steps=n_steps)
    return x_values, y_values

def probability_density_function(sample: Sample,
                                 n_steps: int = 1000):
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

def cumulative_distribution_function(x_values: [float],
                                     y_values: [float]):
    cdf_values = np.cumsum(y_values)
    cdf_values = cdf_values / cdf_values[-1]
    return x_values, cdf_values

def get_x_min(sample: Sample):
    sorted_grains = sorted(sample.grains, key=lambda grain: grain.age)
    return sorted_grains[0].age - sorted_grains[0].uncertainty

def get_x_max(sample: Sample):
    sorted_grains = sorted(sample.grains, key=lambda grain: grain.age)
    return sorted_grains[-1].age + sorted_grains[-1].uncertainty
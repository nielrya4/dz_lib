# Maximum depositional age calculations
# Heavily influenced by Coutts 2019, dzMDA (Sundell), and detritalPy (Sharman)

from dz_lib.univariate.data import Grain, Sample
from dz_lib.univariate import distributions
import numpy as np
import scipy.stats as stats
import peakutils
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
import pandas as pd


# MDA functions:
def youngest_single_grain(grains: [Grain]) -> (Grain, float):
    sorted_grains = sorted(grains, key=lambda grain: grain.age)
    n = 1.0
    return sorted_grains[0], n

def youngest_cluster_1s(
        grains: [Grain],
        min_cluster_size: int = 2,
        contiguous: bool = True
) -> (Grain, int, float):
    sorted_grains = sorted(grains, key=lambda grain: grain.age + grain.uncertainty)
    youngest_cluster = get_youngest_cluster(
        grains=sorted_grains,
        min_cluster_size=min_cluster_size,
        contiguous=contiguous
    )
    if not youngest_cluster:
        return None, float('nan'), 0
    weighted_mean, uncertainty, mswd = get_weighted_mean(
        grains=youngest_cluster,
        confidence_level=0.95
    )
    weighted_grain = Grain(age=weighted_mean, uncertainty=uncertainty)
    return weighted_grain, len(youngest_cluster), mswd

def youngest_cluster_2s(
        grains: [Grain],
        min_cluster_size: int = 3,
        contiguous: bool = True
    ) -> (Grain, int, float):
    for grain in grains:
        grain.uncertainty = grain.uncertainty * 2
    sorted_grains = sorted(grains, key=lambda grain: grain.age + grain.uncertainty)
    youngest_cluster = get_youngest_cluster(
        grains=sorted_grains,
        min_cluster_size=min_cluster_size,
        contiguous=contiguous
    )
    if not youngest_cluster:
        return None, float('nan'), 0
    for grain in youngest_cluster:
        grain.uncertainty = grain.uncertainty / 2
    weighted_mean, uncertainty, mswd = get_weighted_mean(
        grains=youngest_cluster,
        confidence_level=0.95
    )
    weighted_grain = Grain(age=weighted_mean, uncertainty=uncertainty)
    return weighted_grain, len(youngest_cluster), mswd

def youngest_3_zircons(grains: [Grain]) -> (Grain, int, float):
    if len(grains) < 3:
        return None, float('nan')
    sorted_grains = sorted(grains, key=lambda grain: grain.age)
    youngest_three = sorted_grains[:3]
    weighted_mean, uncertainty, mswd = get_weighted_mean(
        grains=youngest_three,
        confidence_level=0.8
    )
    weighted_grain = Grain(age=weighted_mean, uncertainty=uncertainty)
    n = 3
    return weighted_grain, n, mswd


def youngest_3_zircons_overlap(grains: [Grain], sigma: int = 1) -> (Grain, int, float):
    """Find the weighted mean of the youngest 3 overlapping zircon grains."""

    if len(grains) < 3:
        return Grain(age=0, uncertainty=0), 0, 0  # Match MATLAB output

    # Sort grains by adjusted age (like MATLAB's `YGC2s_data`)
    sorted_grains = sorted(grains, key=lambda g: g.age + sigma * g.uncertainty)

    # Select the first 3 grains (like MATLAB's `Y3Zo_data = YGC2s_data(1:3,:)`)
    youngest_cluster = sorted_grains[:3]

    # Compute weighted mean, uncertainty, and MSWD
    weighted_mean, uncertainty_1s, mswd = get_weighted_mean(
        grains=youngest_cluster
    )
    n = len(youngest_cluster)
    # Create a weighted grain with the computed mean age and 2σ uncertainty
    weighted_grain = Grain(age=weighted_mean, uncertainty=uncertainty_1s)

    return weighted_grain, n, mswd


def youngest_graphical_peak(
        grains: [Grain],
        min_cluster_size: int = 2,
        threshold: float = 0.01,
        min_dist: float = 1.0,
        x_min: float = 0,
        x_max: float = 4500,
        n_steps: int = 100000
) -> float:
    if not grains:
        print("No grains provided.")
        return float('nan')
    distro = distributions.pdp_function(Sample("temp", grains), x_min=x_min, x_max=x_max)
    if not distro.x_values.any() or not distro.y_values.any():
        print("Empty distribution data.")
        return float('nan')
    pdp_ages = distro.x_values
    pdp_values = distro.y_values
    step_size = (x_max - x_min) / n_steps
    min_dist_idx = int(min_dist / step_size)
    peak_indexes = list(peakutils.indexes(pdp_values, thres=threshold, min_dist=min_dist_idx))
    if not peak_indexes:
        print("No peaks found.")
        return float('nan')
    peak_ages = [pdp_ages[i] for i in peak_indexes]
    valid_peaks = [
        (age, count_bins_around_peak(age, distro))
        for age in peak_ages
    ]
    valid_peaks = [(age, count) for age, count in valid_peaks if count >= min_cluster_size]
    if not valid_peaks:
        print("No valid peaks found.")
        return float('nan')
    return round(min(valid_peaks, key=lambda p: p[0])[0], 1)

def youngest_statistical_population(
    grains: [Grain],
    min_cluster_size: int = 2,
    mswd_threshold: float = 1.0,
    sigma: float = 1.0,
    add_uncertainty: bool=False
) -> (Grain, int, float):
    if add_uncertainty:
        sorted_grains = sorted(grains, key=lambda g: g.age + sigma * g.uncertainty)
    else:
        sorted_grains = sorted(grains, key=lambda g: g.age)
    best_grain = None
    best_mswd = float('nan')
    best_count = 0
    for j in range(len(sorted_grains) - min_cluster_size + 1):
        subset = sorted_grains[: j + min_cluster_size]
        wm_age, wm_err1s, mswd = get_weighted_mean(subset)
        if j == 0 and mswd > mswd_threshold:
            continue
        if abs(mswd - 1) < abs(best_mswd - 1) if not np.isnan(best_mswd) else True:
            best_grain = Grain(age=wm_age, uncertainty=wm_err1s)
            best_mswd = mswd
            best_count = len(subset)
        if mswd > 1:
            break
    return best_grain, best_count if best_grain else (None, float('nan'), 0), best_mswd

def tau_method(
    grains: [Grain],
    min_cluster_size: int = 3,
    thres: float = 0.01,
    min_dist: int = 1,
) -> (Grain, int, float):
    distro = distributions.pdp_function(Sample("temp", grains))
    x_values = distro.x_values
    y_values = distro.y_values
    trough_indexes = list(peakutils.indexes(-y_values, thres=thres, min_dist=min_dist))
    trough_ages = [0] + list(x_values[trough_indexes]) + [max(x_values)]
    grains_in_troughs = [
        [g for g in grains if trough_ages[j] <= g.age <= trough_ages[j + 1]]
        for j in range(len(trough_ages) - 1)
    ]
    valid_clusters = [i for i, cluster in enumerate(grains_in_troughs) if len(cluster) >= min_cluster_size]
    if not valid_clusters:
        return None, float('nan'), 0
    youngest_index = valid_clusters[0]
    selected_grains = grains_in_troughs[youngest_index]
    tau_WM, tau_WM_err1s, tau_WM_MSWD = get_weighted_mean(selected_grains)
    return Grain(age=tau_WM, uncertainty=tau_WM_err1s), len(selected_grains), tau_WM_MSWD

'''def ygf2(grains: [Grain]):
    def gauss(x, A, B):
        y = A * np.exp(-1 * B * x ** 2)
        return y

    parameters, covariance = curve_fit(gauss, xdata, ydata)

    fit_A = parameters[0]
    fit_B = parameters[1]

    fit_y = gauss(xdata, fit_A, fit_B)
    return '''

def youngest_gaussian_fit(grains: [Grain], x_min=0, x_max=4500) -> (Grain, distributions.Distribution):
    n_steps = 10 * x_max - x_min + 1
    temp_sample = Sample("temp", grains)
    distro = distributions.pdp_function(temp_sample)
    x_values = np.array(distro.x_values)
    y_values = np.array(distro.y_values)
    def gaussian(x, a, mu, sigma):
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    troughs, _ = find_peaks(-y_values)
    if len(troughs) > 0:
        tridx = troughs[0]
    else:
        tridx = len(x_values) - 1
    mask = y_values[:tridx] > 1E-6
    if np.any(mask):
        min_idx = np.where(mask)[0][0]
    else:
        min_idx = 0
    x_young = x_values[min_idx:tridx]
    y_young = y_values[min_idx:tridx]
    initial_guess = [y_young.max(), x_young[np.argmax(y_young)], np.std(x_young)]
    params, _ = curve_fit(gaussian, x_young, y_young, p0=initial_guess)
    a_fit, mu_fit, sigma_fit = params
    YGF_1s = sigma_fit / np.sqrt(2)  # 1 sigma
    x_fit = np.linspace(x_values.min(), x_values.max(), n_steps)
    y_fit = gaussian(x_fit, *params)
    fitted_grain = Grain(mu_fit, YGF_1s)
    fitted_distro = distributions.Distribution(f"Youngest Gaussian Fit\nMean: {mu_fit:.2f} Ma\n1σ: {YGF_1s:.2f}", x_fit, y_fit)
    return fitted_grain, fitted_distro

# MDA utils:
# Not usually used outside of this library
def count_bins_around_peak(peak_age: float, distribution: distributions.Distribution, window: float = 1.0) -> int:
    return sum(1 for x in distribution.x_values if abs(x - peak_age) <= window / 2)

def get_youngest_cluster(
        grains: [Grain],
        min_cluster_size: int,
        add_uncertainty: bool = False,
        contiguous: bool = True
) -> [Grain]:
    if add_uncertainty:
        sorted_grains = sorted(grains, key=lambda grain: grain.age + grain.uncertainty)
    else:
        sorted_grains = sorted(grains, key=lambda grain: grain.age)

    ages_plus_uncertainties = [grain.age + grain.uncertainty for grain in sorted_grains]
    ages_minus_uncertainties = [grain.age - grain.uncertainty for grain in sorted_grains]

    for i, grain in enumerate(sorted_grains):
        overlaps = [
            ages_minus_uncertainties[j] < ages_plus_uncertainties[i]
            for j in range(i, len(sorted_grains))
        ]
        if not contiguous:
            if sum(overlaps) >= min_cluster_size:
                return [sorted_grains[j] for j, overlap in enumerate(overlaps, start=i) if overlap]
        else:
            false_indices = [k for k, overlap in enumerate(overlaps) if not overlap]
            if not false_indices:
                if len(sorted_grains[i:]) >= min_cluster_size:
                    return sorted_grains[i:]
            elif false_indices[0] >= min_cluster_size:
                return sorted_grains[i:i + false_indices[0]]
    return []

def get_weighted_mean(
        grains: [Grain],
        confidence_level: float = 0.95
) -> [float, float, float]:
    ages = np.array([float(abs(grain.age)) for grain in grains])
    errors = np.array([float(abs(grain.uncertainty)) for grain in grains])
    if not grains:
        raise ValueError("Grains list cannot be empty.")
    weight = np.array(errors) ** (-2) / np.sum(np.array(errors) ** (-2))
    weighted_mean_age = np.sum(weight * np.array(ages))
    s = np.sum((np.array(ages) - weighted_mean_age) **2 / np.array(errors) **2)
    n = len(ages)
    mswd = s / (n - 1)
    uncertainty_2s = stats.norm.ppf(confidence_level + (1 - confidence_level) / 2.) * np.sqrt(1. / np.sum(np.array(errors) ** (-2)))
    uncertainty_1s = uncertainty_2s / 2
    return weighted_mean_age, uncertainty_1s, mswd

def ranked_ages_plot(
        grains: [Grain],
        sort_with_uncertainty: bool=True,
        legend: bool=True,
        title: str=None,
        font_path: str=None,
        font_size: float=12,
        fig_width: float=9,
        fig_height: float=7,
        color_1s: str = "black",
        color_2s: str = "cornflowerblue",
):

    if sort_with_uncertainty:
        sorted_grains = sorted(grains, key=lambda grain: grain.age + abs(grain.uncertainty)*2)
    else:
        sorted_grains = sorted(grains, key=lambda grain: grain.age)
    ages = np.array([grain.age for grain in sorted_grains])
    uncertainties = np.array([abs(grain.uncertainty) for grain in sorted_grains])
    ranks = range(len(sorted_grains))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.scatter(ages, ranks, facecolors='white', edgecolors="k", marker='d', s=100, zorder=10)
    ax.hlines(ranks, ages - 2 * uncertainties, ages + 2 * uncertainties, color=color_2s, linewidth=4, label='2σ')
    ax.hlines(ranks, ages - uncertainties, ages + uncertainties, color=color_1s, linewidth=4, label='1σ')
    if font_path:
        font = fm.FontProperties(fname=font_path)
    else:
        font = None
    ax.set_xlabel("Age (Ma)", fontsize=font_size, fontproperties=font)
    ax.set_ylabel("Ranked Grains", fontsize=font_size, fontproperties=font)
    if title:
        ax.set_title(title, fontsize=font_size * 1.5, fontproperties=font)
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.invert_yaxis()
    fig.tight_layout(rect=[0.025, 0.025, 0.975, 1])
    plt.close()
    return fig


def comparison_graph(
        grains: [Grain],
        title: str = None,
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
        color_1s: str = "black",
        color_2s: str = "cornflowerblue",
):
    ysg, ysg_n = youngest_single_grain(grains)
    ypp = Grain(youngest_graphical_peak(grains), float('nan'))
    ygf, _ = youngest_gaussian_fit(grains)
    ygc1s, _, _ = youngest_cluster_1s(grains)
    ygc2s, _, _ = youngest_cluster_2s(grains)
    y3zo, _, _ = youngest_3_zircons_overlap(grains)
    y3za, _, _ = youngest_3_zircons(grains)
    tau, _, _ = tau_method(grains)
    ysp, _, _ = youngest_statistical_population(grains)
    methods = ['YSG', 'YPP', 'YGF', 'YGC1s', 'YGC2s', 'Y3ZO', 'Y3Za', 'TAU', 'YSP']
    ages = [ysg.age, ypp.age, ygf.age, ygc1s.age, ygc2s.age, y3zo.age, y3za.age, tau.age, ysp.age]
    uncertainties = [ysg.uncertainty, ypp.uncertainty, ygf.uncertainty, ygc1s.uncertainty, ygc2s.uncertainty,
                     y3zo.uncertainty, y3za.uncertainty, tau.uncertainty, ysp.uncertainty]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    x = np.arange(len(methods))
    for i in range(len(methods)):
        ax.vlines(x[i], ages[i] - uncertainties[i]*2, ages[i] + uncertainties[i]*2, color=color_2s, linewidth=5)
    for i in range(len(methods)):
        ax.vlines(x[i], ages[i] - uncertainties[i], ages[i] + uncertainties[i], color=color_1s, linewidth=5)
    ax.scatter(x, ages, color='white', edgecolor='black', s=100, zorder=3, marker='s')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_xlabel('Method', fontsize=font_size)
    ax.set_ylabel('Age (Ma)', fontsize=font_size)
    if font_path:
        font_prop = fm.FontProperties(fname=font_path)
    else:
        font_prop = None
    if title:
        ax.set_title(title, fontsize=font_size * 1.5, fontproperties=font_prop)
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', lw=5, label='2s'),
        Line2D([0], [0], color='black', lw=5, label='1s')
    ]
    ax.legend(handles=legend_elements, loc='lower left')
    fig.tight_layout()
    plt.close()
    return fig

def comparison_table(grains: [Grain]):
    ysg, ysg_n = youngest_single_grain(grains)
    ysg_mswd = float('nan')
    ypp = Grain(youngest_graphical_peak(grains), float('nan'))
    ypp_n, ypp_mswd = float('nan'), float('nan')
    ygf, _ = youngest_gaussian_fit(grains)
    ygf_n, ygf_mswd = float('nan'), float('nan')
    ygc1s, ygc1s_n, ygc1s_mswd = youngest_cluster_1s(grains)
    ygc2s, ygc2s_n, ygc2s_mswd = youngest_cluster_2s(grains)
    y3zo, y3zo_n, y3zo_mswd = youngest_3_zircons_overlap(grains)
    y3za, y3za_n, y3za_mswd = youngest_3_zircons(grains)
    tau, tau_n, tau_mswd = tau_method(grains)
    ysp, ysp_n, ysp_mswd = youngest_statistical_population(grains)
    methods = ['YSG', 'YPP', 'YGF', 'YGC1s', 'YGC2s', 'Y3ZO', 'Y3Za', 'TAU', 'YSP']
    ages = [ysg.age, ypp.age, ygf.age, ygc1s.age, ygc2s.age, y3zo.age, y3za.age, tau.age, ysp.age]
    uncertainties = [ysg.uncertainty, ypp.uncertainty, ygf.uncertainty, ygc1s.uncertainty, ygc2s.uncertainty,
                     y3zo.uncertainty, y3za.uncertainty, tau.uncertainty, ysp.uncertainty]
    n_values = [ysg_n, ypp_n, ygf_n, ygc1s_n, ygc2s_n, y3zo_n, y3za_n, tau_n, ysp_n]
    mswd_values = [ysg_mswd, ypp_mswd, ygf_mswd, ygc1s_mswd, ygc2s_mswd, y3zo_mswd, y3za_mswd, tau_mswd, ysp_mswd]
    data = {
        f"% MDA (Ma)": ages,
        "1s (Myr)": uncertainties,
        "2s (Myr)": [uncertainty * 2 for uncertainty in uncertainties],
        "n": n_values,
        "MSWD": mswd_values
    }
    df = pd.DataFrame(data, index=methods)
    df = df.rename_axis(columns="")
    return df
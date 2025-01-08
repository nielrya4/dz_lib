from dz_lib.univariate import distributions, data
from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import find_peaks

def weighted_mean(grains: [data.Grain]):
    weights = 1 / np.array([grain.uncertainty for grain in grains]) ** 2
    ages = np.array([grain.age for grain in grains])
    weighted_mean = np.sum(weights * ages) / np.sum(weights)
    weighted_std = np.sqrt(1 / np.sum(weights))
    mswd = np.sum(((ages - weighted_mean) / np.array([grain.uncertainty for grain in grains])) ** 2) / (len(grains) - 1)
    return weighted_mean, weighted_std, 2 * weighted_std, mswd


def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def youngest_single_grain(sample: data.Sample, cluster_sort):
    if cluster_sort:
        ysg_hi_tmp = [grain.age + 2 * grain.uncertainty for grain in sample.grains]
    else:
        ysg_hi_tmp = [grain.age for grain in sample.grains]
    sort_index = np.argsort(ysg_hi_tmp)
    sorted_grains = [sample.grains[i] for i in sort_index]
    youngest_grain_age = sorted_grains[0].age
    uncertainty_1s = sorted_grains[0].uncertainty
    uncertainty_2s = 2 * sorted_grains[0].uncertainty
    return youngest_grain_age, uncertainty_1s, uncertainty_2s


def youngest_three_zircons_y3za(sample: data.Sample):
    dist_data = sorted(sample.grains, key=lambda grain: grain.age)
    if len(dist_data) > 2:
        y3za_data = dist_data[:3]  # Youngest three zircons
        y3za, y3za_1s, y3za_2s, y3za_mswd = weighted_mean(y3za_data)
    else:
        y3za = y3za_1s = y3za_2s = y3za_mswd = 0
    return y3za, y3za_1s, y3za_2s, y3za_mswd


def youngest_graphical_peak(distribution: distributions.Distribution):
    points = list(zip(distribution.x_values, distribution.y_values))
    points = sorted(points, key=lambda p: p[0])
    peaks = []
    for i in range(1, len(points) - 1):
        if points[i][1] > points[i - 1][1] and points[i][1] > points[i + 1][1]:
            peaks.append(points[i])
    return peaks[0] if peaks else None


def youngest_gaussian_fit(distribution, x_range=(0, 4500), step=0.1, threshold=1e-6):
    # Generate evenly spaced x values for fitting
    x_YGF = np.arange(x_range[0], x_range[1] + step, step)

    # Interpolate and normalize y-values over x_YGF
    pdp_YGF = np.interp(x_YGF, distribution.x_values, distribution.y_values)
    pdp_YGF = pdp_YGF / np.sum(pdp_YGF)  # Normalize to 1

    # Optional: Smooth the data
    from scipy.ndimage import gaussian_filter
    pdp_YGF = gaussian_filter(pdp_YGF, sigma=2)

    # Find local minima in the negative of the distribution
    neg_pdp_YGF = -pdp_YGF
    trs, _ = find_peaks(neg_pdp_YGF)

    # Find the youngest (first) minimum
    if len(trs) > 0:
        tridx = trs[0]
    else:
        tridx = len(x_YGF) - 1  # Default to the last index if no minima are found

    # Extract the range for Gaussian fitting
    above_threshold_indices = np.where(pdp_YGF[:tridx] > threshold)[0]
    if len(above_threshold_indices) == 0:
        min_idx = 0  # Fallback to the start of the distribution
    else:
        min_idx = np.min(above_threshold_indices)

    YGF_x = x_YGF[min_idx:tridx + 1]
    YGF_pdp = pdp_YGF[min_idx:tridx + 1]

    # Perform Gaussian fitting
    try:
        popt, _ = curve_fit(gaussian, YGF_x, YGF_pdp, p0=[1, YGF_x.mean(), 10])
        a, YGF, c = popt
        YGF_1s = c / np.sqrt(2)
        YGF_2s = YGF_1s * 2

        # Generate Gaussian fit values
        x_fit = np.arange(x_range[0], x_range[1] + step, step)
        Yhat = gaussian(x_fit, *popt)

        return {
            "YGF": YGF,
            "YGF_1s": YGF_1s,
            "YGF_2s": YGF_2s,
            "x_fit": x_fit,
            "Yhat": Yhat
        }
    except RuntimeError:
        return {"error": "Gaussian fit failed"}

from dz_lib.univariate import distributions, metrics
from sklearn.manifold import MDS
from dz_lib.univariate.data import Sample
from  dz_lib.utils import fonts, encode
import numpy as np
import matplotlib.pyplot as plt

class MDSPoint:
    def __init__(self, x: float, y: float, label: str, nearest_neighbor: (float, float) = None):
        self.x = x
        self.y = y
        self.label = label
        self.nearest_neighbor = nearest_neighbor

def _compute_dissimilarity_matrix(samples: [Sample], metric: str = "similarity"):
    n_samples = len(samples)
    dissimilarity_matrix = np.zeros((n_samples, n_samples))
    prob_distros = [distributions.pdp_function(sample) for sample in samples]
    c_distros = [distributions.cdf_function(prob_distro) for prob_distro in prob_distros]
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if metric == "similarity":
                dissimilarity_matrix[i, j] = metrics.dis_similarity(prob_distros[i].y_values, prob_distros[j].y_values)
            elif metric == "likeness":
                dissimilarity_matrix[i, j] = metrics.dis_likeness(prob_distros[i].y_values, prob_distros[j].y_values)
            elif metric == "cross_correlation":
                dissimilarity_matrix[i, j] = metrics.dis_r2(prob_distros[i].y_values, prob_distros[j].y_values)
            elif metric == "ks":
                dissimilarity_matrix[i, j] = metrics.ks(c_distros[i].y_values, c_distros[j].y_values)
            elif metric == "kuiper":
                dissimilarity_matrix[i, j] = metrics.kuiper(c_distros[i].y_values, c_distros[j].y_values)
            else:
                raise ValueError(f"Unknown metric '{metric}'")
            dissimilarity_matrix[j, i] = dissimilarity_matrix[i, j]
    
    return dissimilarity_matrix, prob_distros, c_distros

def _compute_mds(dissimilarity_matrix):
    mds_result = MDS(n_components=2, dissimilarity='precomputed')
    scaled_mds_result = mds_result.fit_transform(dissimilarity_matrix)
    return mds_result, scaled_mds_result

def mds_function(samples: [Sample], metric: str = "similarity"):
    n_samples = len(samples)
    dissimilarity_matrix, prob_distros, c_distros = _compute_dissimilarity_matrix(samples, metric)
    mds_result, scaled_mds_result = _compute_mds(dissimilarity_matrix)
    points = []
    for i in range(n_samples):
        distance = float('inf')
        nearest_sample = None
        for j in range(n_samples):
            if i != j:
                if dissimilarity_matrix[i, j] < distance:
                    distance = dissimilarity_matrix[i, j]
                    nearest_sample = samples[j]
        if nearest_sample is not None:
            x1, y1 = scaled_mds_result[i]
            x2, y2 = scaled_mds_result[samples.index(nearest_sample)]
            points.append(MDSPoint(x1, y1, samples[i].name, nearest_neighbor=(x2, y2)))
    stress = mds_result.stress_
    return points, stress, dissimilarity_matrix, scaled_mds_result

def mds_graph(
        points: [MDSPoint],
        title: str = None,
        font_path: str=None,
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
        color_map='plasma'
    ):
    n_samples = len(points)
    colors_map = plt.cm.get_cmap(color_map, n_samples)
    colors = colors_map(np.linspace(0, 1, n_samples))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    for i, point in enumerate(points):
        x1, y1 = point.x, point.y
        x2, y2 = point.nearest_neighbor
        sample_name = point.label
        ax.scatter(x1, y1, color=colors[i])
        ax.text(x1, y1 + 0.005, sample_name, fontsize=font_size*0.75, ha='center', va='center')
        if (x2, y2) is not None:
            ax.plot([x1, x2], [y1, y2], 'k--', linewidth=0.5)
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()
    title_size = font_size * 1.75
    fig.suptitle(title, fontsize=title_size, fontproperties=font)
    fig.text(0.5, 0.01, 'Dimension 1', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Dimension 2', va='center', rotation='vertical', fontsize=font_size, fontproperties=font)
    fig.tight_layout()
    plt.close()
    return fig

def shepard_plot(
        samples: [Sample], 
        metric: str = "similarity",
        title: str = "Shepard Plot",
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 8,
        fig_height: float = 6
    ):
    n_samples = len(samples)
    dissimilarity_matrix, prob_distros, c_distros = _compute_dissimilarity_matrix(samples, metric)
    mds_result, scaled_mds_result = _compute_mds(dissimilarity_matrix)
    
    original_distances = []
    mds_distances = []
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            original_distances.append(dissimilarity_matrix[i, j])
            x1, y1 = scaled_mds_result[i]
            x2, y2 = scaled_mds_result[j]
            euclidean_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            mds_distances.append(euclidean_dist)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.scatter(original_distances, mds_distances, alpha=0.7, s=50)
    
    min_val = min(min(original_distances), min(mds_distances))
    max_val = max(max(original_distances), max(mds_distances))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect fit')
    
    correlation = np.corrcoef(original_distances, mds_distances)[0, 1]
    stress = mds_result.stress_
    
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()
    
    title_size = font_size * 1.75
    fig.suptitle(title, fontsize=title_size, fontproperties=font)
    ax.set_xlabel('Original Dissimilarities', fontsize=font_size, fontproperties=font)
    ax.set_ylabel('MDS Distances', fontsize=font_size, fontproperties=font)
    
    textstr = f'Correlation: {correlation:.3f}\nStress: {stress:.3f}'
    props = dict(boxstyle='round', facecolor='cornflowerblue', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=font_size*0.9,
            verticalalignment='top', bbox=props, fontproperties=font)
    
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.close()
    return fig
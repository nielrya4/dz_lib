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

def mds_function(samples: [Sample], metric: str = "similarity"):
    sample_names = [sample.name for sample in samples]
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

    mds_result = MDS(n_components=2, dissimilarity='precomputed')
    scaled_mds_result = mds_result.fit_transform(dissimilarity_matrix)
    points = []
    for i in range(n_samples):
        distance = float('inf')
        nearest_sample = None
        for j in range(n_samples):
            if i != j:
                if metric == "similarity":
                    dissimilarity = metrics.dis_similarity(prob_distros[i].y_values, prob_distros[j].y_values)
                elif metric == "likeness":
                    dissimilarity = metrics.dis_likeness(prob_distros[i].y_values, prob_distros[j].y_values)
                elif metric == "cross_correlation":
                    dissimilarity = metrics.dis_r2(prob_distros[i].y_values, prob_distros[j].y_values)
                elif metric == "ks":
                    dissimilarity = metrics.ks(c_distros[i].y_values, c_distros[j].y_values)
                elif metric == "kuiper":
                    dissimilarity = metrics.kuiper(c_distros[i].y_values, c_distros[j].y_values)
                else:
                    raise ValueError(f"Unknown metric '{metric}'")
                if dissimilarity < distance:
                    distance = dissimilarity
                    nearest_sample = samples[j]
        if nearest_sample is not None:
            print(nearest_sample.name)
            x1, y1 = scaled_mds_result[i]
            x2, y2 = scaled_mds_result[samples.index(nearest_sample)]
            points.append(MDSPoint(x1, y1, samples[i].name, nearest_neighbor=(x2, y2)))
    stress = mds_result.stress_
    return points, stress

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
        ax.text(x1, y1 + 0.005, sample_name, fontsize=8, ha='center', va='center')
        if (x2, y2) is not None:
            ax.plot([x1, x2], [y1, y2], 'k--', linewidth=0.5)
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()
    title_size = font_size*2
    fig.suptitle(title, fontsize=title_size, fontproperties=font)
    fig.text(0.5, 0.01, 'Dimension 1', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Dimension 2', va='center', rotation='vertical', fontsize=font_size, fontproperties=font)
    fig.tight_layout()
    plt.close()
    return fig

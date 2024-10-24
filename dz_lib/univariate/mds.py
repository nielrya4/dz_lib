from matplotlib.font_manager import FontProperties

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
    probability_distributions = [distributions.probability_density_function(sample) for sample in samples]
    cumulative_distributions = [distributions.cumulative_distribution_function(probability_distributions[0], probability_distributions[1])]
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if metric == "similarity":
                dissimilarity_matrix[i, j] = metrics.dis_similarity(probability_distributions[i][1], probability_distributions[j][1])
            elif metric == "likeness":
                dissimilarity_matrix[i, j] = metrics.dis_likeness(probability_distributions[i][1], probability_distributions[j][1])
            elif metric == "cross_correlation":
                dissimilarity_matrix[i, j] = metrics.dis_r2(probability_distributions[i][1], probability_distributions[j][1])
            elif metric == "ks":
                dissimilarity_matrix[i, j] = metrics.ks(cumulative_distributions[i][1], cumulative_distributions[j][1])
            elif metric == "kuiper":
                dissimilarity_matrix[i, j] = metrics.kuiper(cumulative_distributions[i][1], cumulative_distributions[j][1])
            else:
                raise ValueError(f"Unknown metric '{metric}'")
    mds_result = MDS(n_components=2, dissimilarity='precomputed')
    scaled_mds_result = mds_result.fit_transform(dissimilarity_matrix)
    points = []
    for i in range(n_samples):
        distance = float('inf')
        nearest_sample = None
        for j in range(n_samples):
            if i != j:
                if metric == "similarity":
                    dissimilarity = metrics.dis_similarity(probability_distributions[i][1], probability_distributions[j][1])
                elif metric == "likeness":
                    dissimilarity = metrics.dis_likeness(probability_distributions[i][1], probability_distributions[j][1])
                elif metric == "cross_correlation":
                    dissimilarity = metrics.dis_r2(probability_distributions[i][1], probability_distributions[j][1])
                elif metric == "ks":
                    dissimilarity = metrics.ks(cumulative_distributions[i][1], cumulative_distributions[j][1])
                elif metric == "kuiper":
                    dissimilarity = metrics.kuiper(cumulative_distributions[i][1], cumulative_distributions[j][1])
                else:
                    raise ValueError(f"Unknown metric '{metric}'")
                if dissimilarity < distance:
                    distance = dissimilarity
                    nearest_sample = samples[j]
        if nearest_sample is not None:
            x1, y1 = scaled_mds_result[i]
            x2, y2 = scaled_mds_result[samples.index(nearest_sample)]
            points[i] = MDSPoint(x1, y1, sample_names[i], nearest_neighbor=(x2, y2))
    stress = mds_result.stress_
    return points, stress

def mds_graph(
        points: [MDSPoint],
        title: str = "Multidimensional Scaling Function",
        output_format: str='svg',
        font_path: str=fonts.get_default_font().get_name,
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
        ax.text(x1, y1 + 0.005, sample_name[i], fontsize=8, ha='center', va='center')
        if (x2, y2) is not None:
            ax.plot([x1, x2], [y1, y2], 'k--', linewidth=0.5)
    font = fonts.get_font(font_path)
    title_size = font.size*2
    fig.suptitle(title, fontsize=title_size, fontproperties=font)
    fig.text(0.5, 0.01, 'Dimension 1', ha='center', va='center', fontsize=font_size, fontproperties=font)
    fig.text(0.01, 0.5, 'Dimension 2', va='center', rotation='vertical', fontsize=font_size, fontproperties=font)
    fig.tight_layout()
    if output_format == "html":
        return_str = encode.fig_to_html(fig, fig_type="plotly")
    else:
        return_str = encode.buffer_to_utf8(encode.fig_to_img_buffer(fig, fig_type="plotly", img_format=output_format))
    return return_str

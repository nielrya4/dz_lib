from dz_lib.one_dim import distributions, metrics
from sklearn.manifold import MDS
from dz_lib.one_dim.data import Sample
import numpy as np

class MDSPoint:
    def __init__(self, x: float, y: float, label: str, nearest_neighbor: (float, float) = None):
        self.x = x
        self.y = y
        self.label = label
        self.nearest_neighbor = nearest_neighbor

def multidimensional_scaling_function(samples: [Sample], metric: str = "similarity"):
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
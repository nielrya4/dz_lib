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

def _compute_mds(dissimilarity_matrix, non_metric: bool=True):
    mds_result = MDS(n_components=2, dissimilarity='precomputed', metric=(not non_metric))
    scaled_mds_result = mds_result.fit_transform(dissimilarity_matrix)
    return mds_result, scaled_mds_result

def mds_function(samples: [Sample], metric: str = "similarity", non_metric: bool = True):
    n_samples = len(samples)
    dissimilarity_matrix, prob_distros, c_distros = _compute_dissimilarity_matrix(samples, metric)
    mds_result, scaled_mds_result = _compute_mds(dissimilarity_matrix, non_metric=non_metric)
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
    return points, stress, dissimilarity_matrix, scaled_mds_result, mds_result

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
        dissimilarity_matrix,
        scaled_mds_result,
        mds_result,
        non_metric: bool = True,
        title: str = "Shepard Plot",
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 8,
        fig_height: float = 6
    ):
    n_samples = dissimilarity_matrix.shape[0]
    original_dissimilarities = []
    mds_distances = []
    
    # Get original dissimilarities and MDS distances
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            x1, y1 = scaled_mds_result[i]
            x2, y2 = scaled_mds_result[j]
            euclidean_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            mds_distances.append(euclidean_dist)
            original_dissimilarities.append(dissimilarity_matrix[i, j])
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    
    if non_metric:
        # Plot distances (blue circles) and disparities (red dots) vs dissimilarities
        ax.scatter(original_dissimilarities, mds_distances, alpha=0.7, s=30, 
                  facecolors='none', edgecolors='blue', linewidth=1, label='Distances')
        
        # Create isotonic regression (monotonic fit) to get disparities
        from sklearn.isotonic import IsotonicRegression
        iso_reg = IsotonicRegression()
        disparities = iso_reg.fit_transform(original_dissimilarities, mds_distances)
        
        # Plot disparities as red line (sorted for smooth line)
        sorted_indices = np.argsort(original_dissimilarities)
        sorted_dissim = np.array(original_dissimilarities)[sorted_indices]
        sorted_disparities = disparities[sorted_indices]
        ax.plot(sorted_dissim, sorted_disparities, 'r-', linewidth=2, alpha=0.8, label='Disparities')
        
        # Add 1:1 reference line (dotted diagonal)
        max_val = max(max(original_dissimilarities), max(mds_distances))
        ax.plot([0, max_val], [0, max_val], ':', color='black', alpha=0.6, label='1:1')
        
        ax.set_xlabel('Dissimilarities', fontsize=font_size)
        ax.set_ylabel('Distances/Disparities', fontsize=font_size)
        
        # Calculate R² for the isotonic regression fit (disparities vs MDS distances)
        # But disparities are fitted to match distances, so let's measure the actual fit quality
        r2_disparities = 1.0 - np.sum((mds_distances - disparities)**2) / np.sum((mds_distances - np.mean(mds_distances))**2)
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Add R² and stress text in top right corner
        textstr = f'R² = {r2_disparities:.3f}\nStress = {mds_result.stress_:.5f}'
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, 
                fontsize=font_size*0.9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    else:
        # For metric MDS - match the non-metric styling
        ax.scatter(original_dissimilarities, mds_distances, alpha=0.7, s=30, 
                  facecolors='none', edgecolors='blue', linewidth=1, label='Distances')
        
        # For metric MDS, distances should equal dissimilarities (perfect fit line)
        min_val = min(min(original_dissimilarities), min(mds_distances))
        max_val = max(max(original_dissimilarities), max(mds_distances))
        ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, alpha=0.8, label='Perfect fit')
        
        # Add 1:1 reference line (dotted diagonal) 
        ax.plot([0, max_val], [0, max_val], ':', color='black', alpha=0.6, label='1:1')
        
        ax.set_xlabel('Dissimilarities', fontsize=font_size)
        ax.set_ylabel('Distances', fontsize=font_size)
        
        # Calculate R² for dissimilarities vs distances (how well metric MDS preserves distances)
        r2_metric = np.corrcoef(original_dissimilarities, mds_distances)[0, 1]**2
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Add R² and stress text in top right corner
        textstr = f'R² = {r2_metric:.3f}\nStress = {mds_result.stress_:.5f}'
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, 
                fontsize=font_size*0.9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()
    
    title_size = font_size * 1.75
    fig.suptitle(title, fontsize=title_size, fontproperties=font)
    ax.grid(True, alpha=0.3)
    plt.xlim([0, 1.02])
    fig.tight_layout()
    plt.close()
    return fig

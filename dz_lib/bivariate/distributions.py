import numpy as np
import plotly.graph_objects as go
import scipy.stats as st
from dz_lib.bivariate.data import BivariateSample
from dz_lib.utils.encode import buffer_to_base64, fig_to_img_buffer, buffer_to_utf8
from dz_lib.utils import encode
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

class BivariateDistribution:
    def __init__(
            self,
            mesh_x: [float],
            mesh_y: [float],
            mesh_z: [float],
            kernel: st.gaussian_kde,
            sample_x: [float],
            sample_y: [float]
    ):
        self.mesh_x = mesh_x
        self.mesh_y = mesh_y
        self.mesh_z = mesh_z
        self.kernel = kernel
        self.sample_x = sample_x
        self.sample_y = sample_y

def kde_function_2d(sample: BivariateSample):
    sample_x = [grain.age for grain in sample.grains]
    sample_y = [grain.hafnium for grain in sample.grains]
    deltaX = (max(sample_x) - min(sample_x)) / 10
    deltaY = (max(sample_y) - min(sample_y)) / 10
    xmin = min(sample_x) - deltaX
    xmax = max(sample_x) + deltaX
    ymin = min(sample_y) - deltaY
    ymax = max(sample_y) + deltaY
    mesh_x, mesh_y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([mesh_x.ravel(), mesh_y.ravel()])
    values = np.vstack([sample_x, sample_y])
    kernel = st.gaussian_kde(values)
    bandwidths = np.array([10, 0.25])
    kernel.covariance = np.diag(bandwidths ** 2)
    mesh_z = np.reshape(kernel(positions).T, mesh_x.shape)
    bivariate_distro = BivariateDistribution(mesh_x, mesh_y, mesh_z, kernel, sample_x, sample_y)
    return bivariate_distro


def kde_graph_2d(
        bivariate_distro: BivariateDistribution,
        title: str = "2D Kernel Density Estimate",
        output_format: str = "html",
        show_points: bool = True,
        font: str = 'ubuntu',
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
        x_axis_title: str = "Age (Ma)",
        y_axis_title: str = "εHf(t)",
        z_axis_title: str = "Intensity"
):
    mesh_x = bivariate_distro.mesh_x
    mesh_y = bivariate_distro.mesh_y
    mesh_z = bivariate_distro.mesh_z
    kernel = bivariate_distro.kernel
    sample_x = bivariate_distro.sample_x
    sample_y = bivariate_distro.sample_y
    title_size = font_size * 2
    fig = go.Figure(data=[go.Surface(z=mesh_z, x=mesh_x, y=mesh_y, colorscale='Viridis')])
    if show_points:
        points = np.vstack([sample_x, sample_y])
        scatter_z = kernel(points)
        scatter = go.Scatter3d(
            x=sample_x,
            y=sample_y,
            z=scatter_z,
            mode='markers',
            marker=dict(size=3, color='white', symbol='circle'),
            name='Data Points'
        )
        fig.add_trace(scatter)
    layout_dict = {
        "title": {
            "text": title,
            "font": {
                "family": font,
                "size": title_size,
                "color": "black"
            }
        },
        "scene": {
            "xaxis": {
                "title": {
                    "text": x_axis_title,
                    "font": {
                        "family": font,
                        "size": font_size,
                        "color": "black"
                    }
                }
            },
            "yaxis": {
                "title": {
                    "text": y_axis_title,
                    "font": {
                        "family": font,
                        "size": font_size,
                        "color": "black"
                    }
                }
            },
            "zaxis": {
                "title": {
                    "text": z_axis_title,
                    "font": {
                        "family": font,
                        "size": font_size,
                        "color": "black"
                    }
                }
            }
        },
        "width": fig_width * 100,
        "height": fig_height * 100
    }
    fig.update_layout(layout_dict)
    if output_format == "html":
        return_str = encode.fig_to_html(fig, fig_type="plotly")
    else:
        return_str = buffer_to_utf8(fig_to_img_buffer(fig, fig_type="plotly", img_format=output_format))
    return return_str

def heatmap(
        bivariate_distro: BivariateDistribution,
        show_points=False,
        output_format: str='html',
        title="Heatmap",
        color_map="viridis",
        rescale_factor=1,
        fig_width=9,
        fig_height=7
):
    mesh_x = bivariate_distro.mesh_x
    mesh_y = bivariate_distro.mesh_y
    mesh_z = bivariate_distro.mesh_z
    sample_x = bivariate_distro.sample_x
    sample_y = bivariate_distro.sample_y
    x_rescaled = zoom(mesh_x, rescale_factor)
    y_rescaled = zoom(mesh_y, rescale_factor)
    z_rescaled = zoom(mesh_z, rescale_factor)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    c = ax.pcolormesh(x_rescaled, y_rescaled, z_rescaled, shading='gouraud', cmap=color_map, edgecolors='face')
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel('εHf(t)')
    ax.set_title(title)
    if show_points:
        ax.scatter(sample_x, sample_y, color='white', s=10, edgecolor='black', label='Data Points')
        ax.legend()
    if output_format == "html":
        return_str = encode.fig_to_html(fig, fig_type="matplotlib", vector=False)
    else:
        return_str = buffer_to_base64(fig_to_img_buffer(fig, fig_type="matplotlib", img_format=output_format))
    return return_str
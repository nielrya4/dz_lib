import numpy as np
import plotly.graph_objects as go
import scipy.stats as st
from dz_lib.bivariate.data import Sample
from dz_lib.utils.encode import buffer_to_utf8, fig_to_img_buffer
from dz_lib.utils import encode

def kde_function_2d(sample: Sample):
    x = [grain.age for grain in sample.grains]
    y = [grain.hafnium for grain in sample.grains]
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    bandwidths = np.array([10, 0.25])
    kernel.covariance = np.diag(bandwidths ** 2)
    f = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, f, kernel, x, y


def kde_graph_2d(coordinates: ([float], [float], [float], st.gaussian_kde),
                 title: str = "2D Kernel Density Estimate",
                 output_format: str = "html",
                 show_points: bool = True,
                 font: str = 'ubuntu',
                 font_size: float = 12,
                 fig_width: float = 9,
                 fig_height: float = 7,
                 x_axis_title: str = "Age (Ma)",
                 y_axis_title: str = "εHf(t)",
                 z_axis_title: str = "Intensity"):
    title_size = font_size * 2
    x, y, z, kernel, sample_x, sample_y = coordinates
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
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

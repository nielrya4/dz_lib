from importlib.metadata import distribution

from dz_lib.utils import data
from dz_lib.univariate import distributions, unmix

def test():
    samples_array = data.excel_to_array("/home/ryan/Desktop/dz_data/20_50_30.xlsx")
    samples = data.read_1d_samples(samples_array)
    x_max = data.get_x_max(samples)
    x_min = data.get_x_min(samples)
    kde_distros = []
    for sample in samples:
        kde_distros.append(distributions.kde_function(sample, x_min=x_min, x_max=x_max))
    kde_graph = distributions.distribution_graph(kde_distros, output_format='html')
    with open("/home/ryan/Desktop/kde.html", "w") as file:
        file.write(kde_graph)
    sink_sample = samples[0]
    source_samples = samples[1:]
    sink_y_vals = distributions.kde_function(sink_sample, x_min=x_min, x_max=x_max).y_values
    source_y_vals = []
    for sample in source_samples:
        source_y_val = distributions.kde_function(sample, x_min=x_min, x_max=x_max).y_values
        source_y_vals.append(source_y_val)
    contributions, stds, top_lines = unmix.monte_carlo_model(sink_y_vals, source_y_vals, metric="cross_correlation")
    con_list = []
    for i in range(0, len(source_samples)):
        con_list.append(unmix.Contribution(source_samples[i].name, contributions[i], stds[i]))
    html_graph = unmix.relative_contribution_graph(con_list, output_format='html')
    with open("/home/ryan/Desktop/test.html", "w") as file:
        file.write(html_graph)
if __name__ == '__main__':
    test()

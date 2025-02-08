from dz_lib.univariate.mda import comparison_graph
from dz_lib.utils import data
from dz_lib.univariate.data import Grain, Sample
from dz_lib.univariate import distributions, mda

def test():
    samples_array = data.excel_to_array("/home/ryan/Downloads/DZmda_test_input_1s_uncertainty.xlsx")
    samples = data.read_1d_samples(samples_array)
    sample = samples[0]
    new_sample = sample
    distro = distributions.pdp_function(new_sample)
    fitted_grain, fitted_distro = mda.youngest_gaussian_fit(new_sample.grains)
    graph = distributions.distribution_graph([distro, fitted_distro], fig_height=4, color_map="rainbow", x_min=128, x_max=143)
    graph.show()
    graph = mda.ranked_ages_plot(sample.grains, x_min=128, x_max=143, fig_height=4)
    graph.show()
    table = mda.comparison_table(new_sample.grains)
    print(table)

if __name__ == '__main__':
    test()

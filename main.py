from dz_lib.utils import data
from dz_lib.univariate import distributions, mda

def test():
    samples_array = data.excel_to_array("/home/ryan/Desktop/dz_data/testdata.xlsx")
    samples = data.read_1d_samples(samples_array)
    sample = samples[3]
    distro = distributions.kde_function(sample)
    fitted_grain, fitted_distro = mda.youngest_gaussian_fit(sample.grains)
    graph = distributions.distribution_graph([distro, fitted_distro], color_map="rainbow")
    graph.show()

if __name__ == '__main__':
    test()

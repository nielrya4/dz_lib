from dz_lib.univariate.mda import comparison_graph
from dz_lib.utils import data
from dz_lib.univariate.data import Grain, Sample
from dz_lib.univariate import distributions, mda

def test():
    samples_array = data.excel_to_array("/home/ryan/Desktop/dz_data/testdata.xlsx")
    samples = data.read_1d_samples(samples_array)
    sample = samples[0]
    table = mda.comparison_table(sample.grains)
    graph = comparison_graph(sample.grains)
    print(table)
    graph.show()

if __name__ == '__main__':
    test()

from dz_lib.utils import data
from dz_lib.univariate import mda

def test():
    samples_array = data.excel_to_array("/home/ryan/Desktop/dz_data/Simple_example.xlsx")
    samples = data.read_1d_samples(samples_array)

    grains = samples[3].grains
    new_grains = []
    for grain in grains:
        if 700 < grain.age < 1200:
            new_grains.append(grain)

    graph = mda.ranked_ages_plot(new_grains)
    graph.show()


if __name__ == '__main__':
    test()

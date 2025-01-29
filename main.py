from dz_lib.utils import data
from dz_lib.univariate import mda

def test():
    samples_array = data.excel_to_array("/home/ryan/Desktop/dz_data/Anta_DZmix_Input_1D_only.xlsx")
    samples = data.read_1d_samples(samples_array)

    grains = samples[0].grains

    result = mda.youngest_single_grain(grains)
    print(f"youngest grain age: {result.age} uncertainty: {result.uncertainty}")

    result = mda.youngest_cluster_1s(grains)
    print(f"youngest cluster 1s: {result[0].age} uncertainty: {result[0].uncertainty}")

    result = mda.youngest_cluster_2s(grains)
    print(f"youngest cluster 2s: {result[0].age} uncertainty: {result[0].uncertainty}")

    result = mda.youngest_3_zircons(grains)
    print(f"youngest 3 zircons: {result[0].age} uncertainty: {result[0].uncertainty}")

    result = mda.youngest_3_zircons_overlap(grains)
    print(f"youngest 3 zircons overlap: {result[0].age} uncertainty: {result[0].uncertainty}")

    result = mda.youngest_graphical_peak(grains)
    print(f"youngest graphical peak: {result}")

    result = mda.youngest_statistical_population(grains)
    print(f"youngest statistical population: {result[0].age} uncertainty: {result[0].uncertainty}")

    result = mda.tau_method(grains)
    print(f"Tau Method: {result[0].age} uncertainty: {result[0].uncertainty}")


if __name__ == '__main__':
    test()

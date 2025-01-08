from dz_lib.utils import data
from dz_lib.univariate import distributions, unmix, mda

def test():
    samples_array = data.excel_to_array("/home/ryan/Desktop/dz_data/20_50_30.xlsx")
    samples = data.read_1d_samples(samples_array)
    distro = distributions.pdp_function(samples[0])
    result = mda.youngest_gaussian_fit(distro)
    if "error" in result:
        print(result["error"])
    else:
        print("YGF:", result["YGF"])
        print("YGF 1σ:", result["YGF_1s"])
        print("YGF 2σ:", result["YGF_2s"])
if __name__ == '__main__':
    test()

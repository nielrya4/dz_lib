from dz_lib.one_dim import distributions, data
def test():
    print(distributions.kernel_density_estimate(data.Sample("name", [data.Grain(2, 1)])))

if __name__ == '__main__':
    test()

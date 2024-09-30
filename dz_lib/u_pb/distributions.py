from dz_lib.u_pb.data import Sample
def kernel_density_estimate(sample: Sample, bandwidth: float = 10, steps: int = 1000):
    x_values = None
    y_values = None
    return {"x_values": x_values,
            "y_values": y_values}


def probability_density_function(sample: Sample, steps: int = 1000):
    pass

def cumulative_distribution_function(x_values: [float], y_values: [float]):
    pass
from dz_lib.utils import data, matrices
from dz_lib.univariate import distributions, unmix

def test():
    samples_array = data.excel_to_array("/home/ryan/Desktop/dz_data/Anta_DZmix_Input_1D_only.xlsx")
    samples = data.read_1d_samples(samples_array)
    distros = [distributions.kde_function(sample) for sample in samples]
    
    # Use first sample as sink and rest as sources for unmixing
    sink_distro = distros[0]
    source_distros = distros[1:]
    
    if len(source_distros) == 0:
        print("Need at least 2 samples for unmixing (1 sink, 1+ sources)")
        return


    # Extract y values for unmixing
    sink_y = sink_distro.y_values
    sources_y = [source.y_values for source in source_distros]
    
    # Run Monte Carlo unmixing model
    contributions, std_devs, top_lines = unmix.monte_carlo_model(sink_y, sources_y, n_trials=1000)
    
    # Create contribution objects
    source_names = [f"Source_{i+1}" for i in range(len(source_distros))]
    contribution_objs = [unmix.Contribution(name, contrib, std) 
                        for name, contrib, std in zip(source_names, contributions, std_devs)]
    
    # Display results
    table = unmix.relative_contribution_table(contribution_objs)
    print("Unmixing Results:")
    print(table)
    
    # Show contribution graph
    contrib_graph = unmix.relative_contribution_graph(contribution_objs, title="Sample Contributions")
    contrib_graph.show()
    
    # Show top trials graph
    trials_graph = unmix.top_trials_graph(sink_y, top_lines, title="Best Fitting Models")
    trials_graph.show()

def similarity_matrix_for_testdata():
    # Load testdata.xlsx from the same path
    samples_array = data.excel_to_array("/home/ryan/Desktop/dz_data/testdata.xlsx")
    samples = data.read_1d_samples(samples_array)
    
    # Generate similarity matrix using different metrics
    print("Generating similarity matrices for testdata.xlsx...\n")
    
    # Similarity matrix using similarity metric
    sim_matrix = matrices.generate_data_frame(samples, metric="similarity", function_type="kde")
    print("Similarity Matrix (Gehrels, 2000):")
    print(sim_matrix)
    print()
    
    # Cross-correlation matrix
    corr_matrix = matrices.generate_data_frame(samples, metric="cross_correlation", function_type="kde")
    print("Cross-correlation Matrix (R²):")
    print(corr_matrix)
    print()
    
    # Likeness matrix
    like_matrix = matrices.generate_data_frame(samples, metric="likeness", function_type="kde")
    print("Likeness Matrix (Satkoski et al., 2013):")
    print(like_matrix)
    print()
    
    return sim_matrix, corr_matrix, like_matrix
if __name__ == '__main__':
    test()

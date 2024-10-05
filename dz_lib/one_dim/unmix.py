from concurrent.futures import ProcessPoolExecutor
import numpy as np
from dz_lib.one_dim import metrics
import random

def monte_carlo_model(sink_y_values: [float], sources_y_values: [[float]], n_trials: int=10000, metric: str="cross_correlation"):
    with ProcessPoolExecutor() as executor:
        trials = list(executor.map(create_trial, [(sink_y_values, sources_y_values, metric)] * n_trials))
    if metric == "cross_correlation":
        sorted_trials = sorted(trials, key=lambda x: x.test_val, reverse=True)
    elif metric == "ks" or metric == "kuiper":
        sorted_trials = sorted(trials, key=lambda x: x.test_val, reverse=False)
    else:
        raise ValueError(f"Unknown metric '{metric}'")
    top_trials = sorted_trials[:10]
    top_lines = [trial.model_line for trial in top_trials]
    random_configurations = [trial.random_configuration for trial in top_trials]
    source_contributions = np.average(random_configurations, axis=0) * 100
    source_std = np.std(random_configurations, axis=0) * 100
    return source_contributions, source_std, top_lines

def create_trial(args):
    sink_line, source_lines, test_type = args
    return UnmixingTrial(sink_line, source_lines, metric=test_type)

class UnmixingTrial:
    def __init__(self, sink_line: [float], source_lines: [[float]], metric: str="cross_correlation"):
        self.sink_line = sink_line
        self.source_lines = source_lines
        self.metric = metric
        self.random_configuration, self.model_line, self.test_val = self.__do_trial()

    def __do_trial(self):
        sink_line = self.sink_line
        source_lines = self.source_lines
        n_sources = len(source_lines)
        rands = self.__make_cumulative_random(n_sources)
        model_line = np.zeros_like(sink_line)
        for j, source_line in enumerate(source_lines):
            model_line += source_line * rands[j]
        if self.metric == "r2":
            val = metrics.r2(sink_line, model_line)
        elif self.metric == "ks":
            val = metrics.ks(sink_line, model_line)
        elif self.metric == "kuiper":
            val = metrics.kuiper(sink_line, model_line)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'")
        return rands, model_line, val

    @staticmethod
    def __make_cumulative_random(num_samples):
        rands = [random.random() for _ in range(num_samples)]
        total = sum(rands)
        normalized_rands = [rand / total for rand in rands]
        return normalized_rands

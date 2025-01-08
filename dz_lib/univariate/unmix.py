from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from dz_lib.univariate import metrics
from dz_lib.utils import fonts
import random
from matplotlib import pyplot as plt


class Contribution:
    def __init__(self, name: str, contribution: float, standard_deviation: float):
        self.name = name
        self.contribution = contribution
        self.standard_deviation = standard_deviation

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
        if self.metric == "cross_correlation":
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


def relative_contribution_graph(
        contributions: [Contribution],
        title: str = "Relative Contribution Graph",
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
):
    sample_names = [contribution.name for contribution in contributions]
    x = range(len(contributions))
    y = [contribution.contribution for contribution in contributions]
    e = [contribution.standard_deviation for contribution in contributions]
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100, squeeze=True)
    ax.errorbar(x, y, yerr=e, linestyle="none", marker='.')
    ax.set_title(title, fontsize=font_size * 2, fontproperties=font)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=font_size, fontproperties=font)
    plt.tight_layout()
    plt.close()
    return fig


def relative_contribution_table(
        contributions: [Contribution],
        metric: str = "cross_correlation",
        title=f"Relative Contribution Table"
    ):
    sample_names = [contribution.name for contribution in contributions]
    percent_contributions = [contribution.contribution for contribution in contributions]
    standard_deviations = [contribution.standard_deviation for contribution in contributions]
    data = {
        f"% Contribution (metric={metric})": percent_contributions,
        "Standard Deviation": standard_deviations
    }
    indices = [f"{name}" for name in sample_names]
    df = pd.DataFrame(data, index=indices)
    df.style.set_table_attributes("style='display:inline'").set_caption(title)
    df = df.rename_axis(columns="Sample Name")
    return df


def top_trials_graph(
        sink_line: [float],
        model_lines: [[float]],
        x_range: [float, float] = [0, 4500],
        title: str = "Top Trials Graph",
        font_path: str = None,
        font_size: float = 12,
        fig_width: float = 9,
        fig_height: float = 7,
    ):
    #todo: pass in entire distributions instead of just y values
    x = np.linspace(x_range[0], x_range[1], 1000).reshape(-1, 1)
    if font_path:
        font = fonts.get_font(font_path)
    else:
        font = fonts.get_default_font()
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    for i, model_kde in enumerate(model_lines):
        ax.plot(x, model_kde, 'c-', label="Top Trials" if i == 0 else "_Top Trials")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.plot(x, sink_line, 'b-', label="Sink Sample")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title(title, fontsize=font_size*2, fontproperties=font)
    ax.set_xlabel("Age (Ma)", fontsize=font_size, fontproperties=font)
    ax.set_ylabel("Probability Differential", fontsize=font_size, fontproperties=font)
    plt.tight_layout()
    plt.close()
    return fig
"""Collection of functions to plot, with matplotlib, one or several Results as 
histograms, when the ``job_type`` is ``SAMPLE``."""

import math
import random
from typing import Union

import matplotlib.pyplot as plt
from typeguard import typechecked

from mpqp.execution.job import JobType
from mpqp.execution.result import Result, BatchResult


@typechecked
def plot_results_sample_mode(results: Union[BatchResult, Result]):
    """
    Display the several results in parameter in a grid using subplots.

    Args:
        results: BatchResult to plot in the same window.
    """
    if isinstance(results, Result):
        return _plot_result_sample_mode(results)

    n_cols = math.ceil(len(results.results) // 2)
    n_rows = 2

    for index, result in enumerate(results.results):
        plt.subplot(n_rows, n_cols, index + 1)

        _prep_plot(result)

    plt.tight_layout()


@typechecked
def _plot_result_sample_mode(result: Result):
    """
    Display the result in parameter in a single figure.

    Args:
        result: Result to plot.
    """
    plt.figure()

    _prep_plot(result)


@typechecked
def _prep_plot(result: Result):
    """
    Extract sampling info from the result and construct the bar diagram plot.

    Args:
        result: Result to transform into a bar diagram.
    """

    x_array, y_array = _result_to_array_sample_mode(result)
    x_axis = range(len(x_array))

    plt.bar(x_axis, y_array, color=(*[random.random() for _ in range(3)], 0.9))
    plt.xticks(x_axis, x_array, rotation=-60)
    plt.xlabel("State")
    plt.ylabel("Counts")
    device = result.job.device
    plt.title(type(device).__name__ + ", " + device.name)


@typechecked
def _result_to_array_sample_mode(result: Result):
    """
    Transform a result into an x and y array containing the string of basis state with the associated counts
    Args:
        result: result used to generate the array

    Returns:
        the tuple x_array (strings for each basis state) and y_array (counts for each basis state)
    """
    if result.job.job_type != JobType.SAMPLE:
        raise NotImplementedError(
            f"{result.job.job_type} not handled, only {JobType.SAMPLE} is handled for now."
        )
    if result.job.measure is None:
        raise ValueError(
            f"{result.job=} has no measure, making the counting impossible"
        )
    n = result.job.measure.nb_qubits
    x_array = [f"|{bin(i)[2:].zfill(n)}⟩" for i in range(2**n)]
    y_array = result.counts
    return x_array, y_array

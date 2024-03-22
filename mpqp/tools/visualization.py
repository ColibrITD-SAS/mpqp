"""Collection of functions to plot, with matplotlib, one or several Results as 
histograms, when the ``job_type`` is ``SAMPLE``."""

import math
import random
from typing import Union

import matplotlib.pyplot as plt
from typeguard import typechecked

from mpqp.execution.job import JobType
from mpqp.execution.result import BatchResult, Result


@typechecked
def plot_results_sample_mode(results: Union[BatchResult, Result]):
    """Display the result(s) using ``matplotlib.pyplot``.

    If a ``BatchResult`` is given, the contained results will be displayed in a
    grid using subplots.

    Args:
        results: result(s) to plot.
    """
    if isinstance(results, Result):
        plt.figure()

        return _prep_plot(results)

    n_cols = math.ceil((len(results.results)+1) // 2)
    n_rows = math.ceil(len(results.results) - n_cols)

    for index, result in enumerate(results.results):
        plt.subplot(n_rows, n_cols, index + 1)

        _prep_plot(result)

    plt.tight_layout()


@typechecked
def _prep_plot(result: Result):
    """Extract sampling info from the result and construct the bar diagram plot.

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
def _result_to_array_sample_mode(result: Result) -> tuple[list[str], list[int]]:
    """Transform a result into an x and y array containing the string of basis
    state with the associated counts.

    Args:
        result: Result used to generate the array.

    Returns:
        The list of each basis state and the corresponding counts.
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

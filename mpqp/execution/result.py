"""Once the computation ended, the :class:`Result` contains all the data from
the execution.

The job type affects the data contained in the :class:`Result`. For a given 
``result``, here are how to retrieve the data depending on the job type:

- for a job type ``STATE_VECTOR`` you can retrieve the :class:`StateVector` from 
  ``result.state_vector``. If you want to directly get the amplitudes of your
  state vector, you can reach for ``result.amplitudes``;
- for a job type ``SAMPLE`` you can retrieve the list of :class:`Sample` from 
  ``result.samples``. For a ``SAMPLE`` job type, you might be interested in
  results packed in a different shape than a list of :class:`Sample`, even
  though you could rebuild them from said list, we also provide a few shorthands
  like ``result.probabilities`` and ``result.counts``;
- for a job type ``OBSERVABLE`` you can retrieve the expectation value (a 
  ``float``) from ``result.expectation_value``.

When several devices are given to :func:`~mpqp.execution.runner.run`, the 
results are stored in a :class:`BatchResult`.
"""

from __future__ import annotations

import math
import random
from numbers import Complex
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

from mpqp.core.instruction.measurement.basis_measure import BasisMeasure
from mpqp.core.instruction.measurement.pauli_string import PauliString
from mpqp.execution import Job, JobType
from mpqp.execution.devices import AvailableDevice
from mpqp.tools.display import clean_1D_array, clean_number_repr
from mpqp.tools.errors import ResultAttributeError


@typechecked
class StateVector:
    """Class representing the state vector of a multi-qubit quantum system.

    Args:
        vector: List of amplitudes defining the state vector.
        nb_qubits: Number of qubits of the state.
        probabilities: List of probabilities associated with the state vector.

    Example:
        >>> state_vector = StateVector(np.array([1, 1, 1, -1])/2, 2)
        >>> state_vector.probabilities
        array([0.25, 0.25, 0.25, 0.25])
        >>> print(state_vector)
         State vector: [0.5, 0.5, 0.5, -0.5]
         Probabilities: [0.25, 0.25, 0.25, 0.25]
         Number of qubits: 2

    """

    def __init__(
        self,
        vector: list[Complex] | npt.NDArray[np.complex64],
        nb_qubits: Optional[int] = None,
        probabilities: Optional[list[float] | npt.NDArray[np.float32]] = None,
    ):
        if len(np.asarray(vector)) == 0:
            raise ValueError("vector should not be empty")

        self.vector: npt.NDArray[np.complex64] = np.array(vector, dtype=complex)

        self.nb_qubits = (
            int(math.log(len(vector), 2)) if nb_qubits is None else nb_qubits
        )
        """See parameter description."""
        self.probabilities = (
            abs(self.vector) ** 2 if probabilities is None else np.array(probabilities)
        )
        """See parameter description."""

    @property
    def amplitudes(self):
        """Return the amplitudes of the state vector"""
        return self.vector

    def __str__(self):
        return f""" State vector: {clean_1D_array(self.vector)}
 Probabilities: {clean_1D_array(self.probabilities)}
 Number of qubits: {self.nb_qubits}"""

    def __repr__(self) -> str:
        return f"StateVector({clean_1D_array(self.vector)})"


@typechecked
class Sample:
    """A sample is a partial result of job job with type ``SAMPLE``. It contains
    the count (and potentially the associated probability) for a given basis
    state, *i.e.* the number of times this basis state was measured.

    Args:
        nb_qubits: Number of qubits of the quantum system of the experiment.
        probability: Probability of measuring the basis state associated to this
            sample.
        index: Index in decimal notation representing the basis state.
        count: Number of times this basis state was measured during the
            experiment.
        bin_str: String representing the basis state in binary notation.

    Examples:
        >>> print(Sample(3, index=3, count=250, bin_str="011"))
        State: 011, Index: 3, Count: 250, Probability: None

        >>> print(Sample(4, index=6, probability=0.5))
        State: 0110, Index: 6, Count: None, Probability: 0.5

        >>> print(Sample(5, bin_str="01011", count=1234))
        State: 01011, Index: 11, Count: 1234, Probability: None

    """

    def __init__(
        self,
        nb_qubits: int,
        probability: Optional[float] = None,
        index: Optional[int] = None,
        count: Optional[int] = None,
        bin_str: Optional[str] = None,
    ):
        self.nb_qubits = nb_qubits
        """See parameter description."""
        self.count = count
        """See parameter description."""
        self.probability = probability
        """See parameter description."""
        self.bin_str: str
        """See parameter description."""
        self.index: int
        """See parameter description."""

        if index is None:
            if bin_str is None:
                raise ValueError(
                    "At least one of `bin_str` and `index` arguments is necessary"
                )
            else:
                self.index = int(bin_str, 2)
                self.bin_str = bin_str
        else:
            computed_bin_str = bin(index)[2:].zfill(self.nb_qubits)
            if bin_str is None:
                self.index = index
                self.bin_str = computed_bin_str
            else:
                if computed_bin_str == bin_str:
                    self.index = index
                    self.bin_str = bin_str
                else:
                    raise ResultAttributeError(
                        f"The value of bin_str {bin_str} doesn't match with the"
                        f" index provided {index} and the number of qubits {self.nb_qubits}"
                    )

    def __str__(self):
        return (
            f"State: {self.bin_str}, Index: {self.index}, Count: {self.count}"
            + f", Probability: {np.round(self.probability, 5) if self.probability is not None else None}"
        )

    def __repr__(self):
        return f"Sample({self.nb_qubits}, index={self.index}, count={self.count}, probability={self.probability})"


@typechecked
class Result:
    """Result associated to a submitted job.

    The data type in a result depends on the job type, according to the
    following chart:

    +-------------+--------------+
    | Job Type    | Data Type    |
    +=============+==============+
    | OBSERVABLE  | float        |
    +-------------+--------------+
    | SAMPLE      | list[Sample] |
    +-------------+--------------+
    | STATE_VECTOR| StateVector  |
    +-------------+--------------+

    Args:
        job: Type of the job related to this result.
        data: Data of the result, can be an expectation value (float), a
            StateVector, or a list of sample depending on the job_type.
        errors: Information about the error or the variance in the measurement.
        shots: Number of shots of the experiment (equal to zero if the exact
            value was required).

    Examples:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit(2), ATOSDevice.MYQLM_CLINALG)
        >>> print(Result(job, StateVector(np.array([1, 1, 1, -1], dtype=np.complex64) / 2, 2), 0, 0)) # doctest: +NORMALIZE_WHITESPACE
        Result: ATOSDevice, MYQLM_CLINALG
         State vector: [0.5, 0.5, 0.5, -0.5]
         Probabilities: [0.25, 0.25, 0.25, 0.25]
         Number of qubits: 2
        >>> job = Job(
        ...     JobType.SAMPLE,
        ...     QCircuit([BasisMeasure([0, 1], shots=1000)]),
        ...     ATOSDevice.MYQLM_CLINALG,
        ...     BasisMeasure([0, 1], shots=1000),
        ... )
        >>> print(Result(job, [
        ...     Sample(2, index=0, count=250),
        ...     Sample(2, index=3, count=250)
        ... ], 0.034, 500)) # doctest: +NORMALIZE_WHITESPACE
        Result: ATOSDevice, MYQLM_CLINALG
         Counts: [250, 0, 0, 250]
         Probabilities: [0.5, 0, 0, 0.5]
         Samples:
          State: 00, Index: 0, Count: 250, Probability: 0.5
          State: 11, Index: 3, Count: 250, Probability: 0.5
         Error: 0.034
        >>> job = Job(JobType.OBSERVABLE, QCircuit(2), ATOSDevice.MYQLM_CLINALG)
        >>> print(Result(job, -3.09834, 0.021, 2048)) # doctest: +NORMALIZE_WHITESPACE
        Result: ATOSDevice, MYQLM_CLINALG
         Expectation value: -3.09834
         Error/Variance: 0.021

    """

    # TODO: in this class, there is a lot of manual type checking, this is an
    #  anti-pattern in my opinion, it should probably be fixed using subclasses

    def __init__(
        self,
        job: Job,
        data: float | StateVector | list[Sample],
        errors: Optional[float | dict[PauliString, float] | dict[Any, Any]] = None,
        shots: int = 0,
    ):
        self.job = job
        """See parameter description."""
        self._expectation_value = None
        self._state_vector = None
        self._probabilities = None
        self._counts = None
        self._samples = None
        self.shots = shots
        """See parameter description."""
        self.error = errors
        """See parameter description."""
        self._data = data

        # depending on the type of job, fills the result info from the data in parameter
        if job.job_type == JobType.OBSERVABLE:
            if not isinstance(data, float):
                raise TypeError(
                    "Wrong type of data in the result. "
                    "Expecting float for expectation value of an observable"
                )
            else:
                self._expectation_value = data
        elif job.job_type == JobType.STATE_VECTOR:
            if not isinstance(data, StateVector):
                raise TypeError(
                    "Wrong type of data in the result. Expecting StateVector"
                )
            else:
                self._state_vector = data
                if job.circuit.gphase != 0:
                    # Reverse the global phase introduced when using CustomGate, due to Qiskit decomposition in QASM2
                    self._state_vector.vector *= np.exp(1j * job.circuit.gphase)
                self._probabilities = data.probabilities
        elif job.job_type == JobType.SAMPLE:
            if not isinstance(data, list):
                raise TypeError(
                    "Wrong type of data in the result (not a list). Expecting list of Sample"
                )
            if self.job.measure is None:
                raise ValueError(
                    f"{self.job=} has no measure, making the counting impossible"
                )
            self._samples = data
            is_counts = all([sample.count is not None for sample in data])
            is_probas = all([sample.probability is not None for sample in data])
            if is_probas:
                probas = [0.0] * (2**self.job.measure.nb_qubits)
                for sample in data:
                    probas[sample.index] = sample.probability
                self._probabilities = np.array(probas, dtype=float)

                if not is_counts:
                    counts = [
                        int(count)
                        for count in np.round(
                            self.job.measure.shots * self._probabilities
                        )
                    ]
                    self._counts = counts
                    for sample in self._samples:
                        sample.count = self._counts[sample.index]
            if is_counts:
                counts: list[int] = [0] * (2**self.job.measure.nb_qubits)
                for sample in data:
                    if TYPE_CHECKING:
                        assert sample.count is not None
                    counts[sample.index] = sample.count
                self._counts = counts
                assert shots != 0
                if not is_probas:
                    self._probabilities = np.array(counts, dtype=float) / self.shots
                    for sample in self._samples:
                        sample.probability = self._probabilities[sample.index]
            elif not is_probas:
                raise ValueError(
                    f"For {JobType.SAMPLE.name} jobs, all samples must contain"
                    " either `count` or `probability` (and the non-None "
                    "attribute amongst the two must be the same in all samples)."
                )
            self.samples.sort(key=lambda sample: sample.bin_str)
        else:
            raise ValueError(f"{job.job_type} not handled")

    @property
    def device(self) -> AvailableDevice:
        """Device on which the job of this result was run"""
        return self.job.device

    @property
    def expectation_value(self) -> float:
        """Get the expectation value stored in this result"""
        if self.job.job_type != JobType.OBSERVABLE:
            raise ResultAttributeError(
                f"Job type: {self.job.job_type.name} but cannot get expectation"
                " value if the job type is not OBSERVABLE."
            )
        if TYPE_CHECKING:
            assert self._expectation_value is not None
        return self._expectation_value

    @property
    def amplitudes(self) -> npt.NDArray[np.complex64]:
        """Get the amplitudes of the state of this result"""
        if self.job.job_type != JobType.STATE_VECTOR:
            raise ResultAttributeError(
                "Cannot get amplitudes if the job was not of type STATE_VECTOR"
            )
        if TYPE_CHECKING:
            assert self._state_vector is not None
        return self._state_vector.amplitudes

    @property
    def state_vector(self) -> StateVector:
        """Get the state vector of the state associated with this result"""
        if self.job.job_type != JobType.STATE_VECTOR:
            raise ResultAttributeError(
                "Cannot get state vector if the job was not of type STATE_VECTOR"
            )
        if TYPE_CHECKING:
            assert self._state_vector is not None
        return self._state_vector

    @property
    def samples(self) -> list[Sample]:
        """Get the list of samples of the result"""
        if self.job.job_type != JobType.SAMPLE:
            raise ResultAttributeError(
                "Cannot get samples if the job was not of type SAMPLE"
            )
        if TYPE_CHECKING:
            assert self._samples is not None
        return self._samples

    @property
    def probabilities(self) -> npt.NDArray[np.float32]:
        """Get the list of probabilities associated with this result"""
        if self.job.job_type not in (JobType.SAMPLE, JobType.STATE_VECTOR):
            raise ResultAttributeError(
                "Cannot get probabilities if the job was not of"
                " type SAMPLE or STATE_VECTOR"
            )
        if TYPE_CHECKING:
            assert self._probabilities is not None
        return self._probabilities

    @property
    def counts(self) -> list[int]:
        """Get the list of counts for each sample of the experiment"""
        if self.job.job_type != JobType.SAMPLE:
            raise ResultAttributeError(
                "Cannot get counts if the job was not of type SAMPLE"
            )

        if TYPE_CHECKING:
            assert self._counts is not None
        return self._counts

    def __str__(self):
        label = "" if self.job.circuit.label is None else self.job.circuit.label + ", "
        header = f"Result: {label}{type(self.device).__name__}, {self.device.name}"

        if self.job.job_type == JobType.SAMPLE:
            measures = self.job.circuit.measurements
            if not len(measures) == 1:
                raise ValueError(
                    "Mismatch between the number of measurements and the job type."
                )
            measure = measures[0]
            if not isinstance(measure, BasisMeasure):
                raise ValueError("Mismatch between measurements type and job type.")

            # assert all(sample.probability is not None for sample in self.samples)

            probabilities = [
                sample.probability
                for sample in self.samples
                if sample.probability is not None
            ]

            if len(probabilities) != len(self.samples):
                raise ValueError(
                    f"Some samples ({len(self.samples)-len(probabilities)} of them) have probabilities to None."
                )

            samples_str = "\n".join(
                f"  State: {measure.basis.binary_to_custom(bin(sample.index)[2:].zfill(self.job.circuit.nb_qubits))}, "
                f"Index: {sample.index}, Count: {sample.count}, Probability: {clean_number_repr(probability)}"
                for sample, probability in zip(self.samples, probabilities)
            )
            return f"""{header}
 Counts: {self._counts}
 Probabilities: {clean_1D_array(self.probabilities)}
 Samples:
{samples_str}
 Error: {self.error}"""

        if self.job.job_type == JobType.STATE_VECTOR:
            return header + "\n" + str(self.state_vector)

        if self.job.job_type == JobType.OBSERVABLE:
            return f"""{header}
 Expectation value: {self.expectation_value}
 Error/Variance: {self.error}"""

        raise NotImplementedError(
            f"I don't know how to represent results of {self.job.job_type} jobs"
            " as a string."
        )

    def __repr__(self) -> str:
        return (
            f"Result({repr(self.job)}, {repr(self._data)}, {repr(self.error)}, "
            f"{repr(self.shots)})"
        )

    def plot(self, show: bool = True):
        """Extract sampling info from the result and construct the bar diagram
        plot.

        Args:
            show: ``plt.show()`` is only executed if ``show``, useful to batch
                plots.
        """
        from matplotlib import pyplot as plt

        if show:
            plt.figure()

        x_array, y_array = self._to_display_lists()
        x_axis = range(len(x_array))

        plt.bar(x_axis, y_array, color=(*[random.random() for _ in range(3)], 0.9))
        plt.xticks(x_axis, x_array, rotation=-60)
        plt.xlabel("State")
        plt.ylabel("Counts")
        device = self.job.device
        plt.title(f"{self.job.circuit.label}, {type(device).__name__}\n{device.name}")

        if show:
            plt.show()

    def _to_display_lists(self) -> tuple[list[str], list[int]]:
        """Transform a result into an x and y array containing the string of
        basis state with the associated counts.

        Returns:
            The list of each basis state and the corresponding counts.
        """
        if self.job.job_type != JobType.SAMPLE:
            raise NotImplementedError(
                f"{self.job.job_type} not handled, only {JobType.SAMPLE} is handled for now."
            )
        if self.job.measure is None:
            raise ValueError(
                f"{self.job=} has no measure, making the counting impossible"
            )
        n = self.job.measure.nb_qubits
        x_array = [f"|{bin(i)[2:].zfill(n)}⟩" for i in range(2**n)]
        y_array = self.counts
        return x_array, y_array


@typechecked
class BatchResult:
    """Class used to handle several Result instances.

    Args:
        results: List of results.

    Example:
        >>> result1 = Result(
        ...     Job(JobType.STATE_VECTOR,QCircuit(0, label="StateVector circuit"),
        ...     ATOSDevice.MYQLM_PYLINALG),
        ...     StateVector(np.array([1, 1, 1, -1])/2, 2),
        ...     0,
        ...     0
        ... )
        >>> result2 = Result(
        ...     Job(
        ...         JobType.SAMPLE,
        ...         QCircuit([BasisMeasure([0,1],shots=500)], label="Sample circuit"),
        ...         ATOSDevice.MYQLM_PYLINALG,
        ...         BasisMeasure([0,1],shots=500)
        ...     ),
        ...     [Sample(2, index=0, count=250), Sample(2, index=3, count=250)],
        ...     0.034,
        ...     500)
        >>> result3 = Result(
        ...     Job(JobType.OBSERVABLE,QCircuit(0, label="Observable circuit"),
        ...     ATOSDevice.MYQLM_PYLINALG),
        ...     -3.09834,
        ...     0.021,
        ...     2048
        ... )
        >>> batch_result = BatchResult([result1, result2, result3])
        >>> print(batch_result)
        BatchResult: 3 results
        Result: StateVector circuit, ATOSDevice, MYQLM_PYLINALG
         State vector: [0.5, 0.5, 0.5, -0.5]
         Probabilities: [0.25, 0.25, 0.25, 0.25]
         Number of qubits: 2
        Result: Sample circuit, ATOSDevice, MYQLM_PYLINALG
         Counts: [250, 0, 0, 250]
         Probabilities: [0.5, 0, 0, 0.5]
         Samples:
          State: 00, Index: 0, Count: 250, Probability: 0.5
          State: 11, Index: 3, Count: 250, Probability: 0.5
         Error: 0.034
        Result: Observable circuit, ATOSDevice, MYQLM_PYLINALG
         Expectation value: -3.09834
         Error/Variance: 0.021
        >>> print(batch_result[0])
        Result: StateVector circuit, ATOSDevice, MYQLM_PYLINALG
         State vector: [0.5, 0.5, 0.5, -0.5]
         Probabilities: [0.25, 0.25, 0.25, 0.25]
         Number of qubits: 2

    """

    def __init__(self, results: list[Result]):
        self.results = results
        """See parameter description."""

    def __str__(self):
        header = f"BatchResult: {len(self.results)} results\n"
        body = "\n".join(map(str, self.results))
        return header + body

    def __repr__(self):
        return f"BatchResult({self.results})"

    def __getitem__(self, index: int):
        return self.results[index]

    def plot(self, show: bool = True):
        """Display the result(s) using ``matplotlib.pyplot``.

        The result(s) must be from a job who's ``job_type`` is ``SAMPLE``. They will
        be displayed as histograms.

        If a ``BatchResult`` is given, the contained results will be displayed in a
        grid using subplots.

        Args:
            show: ``plt.show()`` is only executed if ``show``, useful to batch
                plots.
        """
        from matplotlib import pyplot as plt

        n_cols = math.ceil((len(self.results) + 1) // 2)
        n_rows = math.ceil(len(self.results) / n_cols)

        for index, result in enumerate(self.results):
            plt.subplot(n_rows, n_cols, index + 1)

            result.plot(show=False)

        plt.tight_layout()

        if show:
            plt.show()

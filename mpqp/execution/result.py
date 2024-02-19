######################################
# Copyright(C) 2021 - 2023 ColibrITD
#
# Developers :
#  - Hamza JAFFALI < hamza.jaffali@colibritd.com >
#  - Karla BAUMANN < karla.baumann@colibritd.com >
#  - Henri de BOUTRAY < henri.de.boutray@colibritd.com >
#
# Version : 0.1
#
# This file is part of QUICK.
#
# QUICK can not be copied and / or distributed without the express
# permission of ColibrITD
#
######################################
from __future__ import annotations

import math
from numbers import Complex
from textwrap import dedent
from typing import Optional

import numpy as np
import numpy.typing as npt
from typeguard import typechecked

from mpqp.execution.devices import AvailableDevice
from mpqp.tools.errors import ResultAttributeError

from .job import Job, JobType


@typechecked
class StateVector:
    """
    Class representing the state vector of a multi-qubit quantum system.

    Args:
        vector: List of amplitudes defining the state vector.
        nb_qubits: Number of qubits of the state.
        probabilities: List of probabilities associated with the state vector.

    Example:
        >>> state_vector = StateVector(np.array([1, 1, 1, -1])/2, 2)
        >>> state_vector.probabilities
        array([0.25, 0.25, 0.25, 0.25])
        >>> print(state_vector)
        State vector: [ 0.5  0.5  0.5 -0.5]
        Probabilities: [0.25 0.25 0.25 0.25]
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
        return f"""
        State vector: {self.vector}
        Probabilities: {self.probabilities}
        Number of qubits: {self.nb_qubits}"""


@typechecked
class Sample:
    """
    Class representing a sample, which contains the result of the experiment concerning a specific basis state
    of the Hilbert space.

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
                    raise ValueError(
                        f"The value of bin_str {bin_str} doesn't match with the"
                        f" index provided {index}"
                    )

    def __str__(self):
        str1 = "State: " + str(self.bin_str) + ", Index: " + str(self.index)
        str2 = ", Count: " + str(self.count) + ", Probability: " + str(self.probability)
        return str1 + str2

    def __repr__(self):
        str1 = "State: " + str(self.bin_str) + ", Index: " + str(self.index)
        str2 = ", Count: " + str(self.count) + ", Probability: " + str(self.probability)
        return str1 + str2


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
        error: Information about the error or the variance in the measurement.
        shots: Number of shots of the experiment (equal to zero if the exact
            value was required).

    Examples:
        >>> print(Result(Job(), StateVector(np.array([1, 1, 1, -1])/2, 2), 0, 0))
        State vector: [ 0.5  0.5  0.5 -0.5]
        Probabilities: [0.25 0.25 0.25 0.25]
        Number of qubits: 2

        >>> print(Result(Job(), [
        ...     Sample(2, index=0, count=250)
        ...     Sample(2, index=3, count=250)
        ... ], 0.034, 500))
        Counts: [250, 250]
        Probabilities: [0.5 0.5]
        State: 00, Index: 0, Count: 250, Probability: None
        State: 11, Index: 3, Count: 250, Probability: None
        Error: 0.034

        >>> print(Result(Job(), -3.09834, 0.021, 2048))
        Expectation value: -3.09834
        Error: 0.021
    """

    # 3M-TODO: in this class, there is a lot of manual type checking, this is an
    #  anti-pattern in my opinion, it should probably be fixed using subclasses

    def __init__(
        self,
        job: Job,
        data: float | StateVector | list[Sample],
        error: Optional[float] = None,
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
        self.error = error
        """See parameter description."""

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

                counts = [
                    int(count)
                    for count in np.round(self.job.measure.shots * self._probabilities)
                ]
                self._counts = counts
                for sample in self._samples:
                    sample.count = self._counts[sample.index]
            elif is_counts:
                counts: list[int] = [0] * (2**self.job.measure.nb_qubits)
                for sample in data:
                    assert sample.count is not None
                    counts[sample.index] = sample.count
                self._counts = counts
                # if is_counts shots != 0
                assert shots != 0
                self._probabilities = np.array(counts, dtype=float) / self.shots
                for sample in self._samples:
                    sample.probability = self._probabilities[sample.index]
            else:
                raise ValueError(
                    f"For {JobType.SAMPLE.name} jobs, all samples must contain"
                    " either `count` or `probability` (and the non-None "
                    "attribute amongst the two must be the same in all samples)."
                )
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
        assert self._expectation_value is not None
        return self._expectation_value

    @property
    def amplitudes(self) -> npt.NDArray[np.complex64]:
        """Get the amplitudes of the state of this result"""
        if self.job.job_type != JobType.STATE_VECTOR:
            raise ResultAttributeError(
                "Cannot get amplitudes if the job was not of type STATE_VECTOR"
            )
        assert self._state_vector is not None
        return self._state_vector.amplitudes

    @property
    def state_vector(self) -> StateVector:
        """Get the state vector of the state associated with this result"""
        if self.job.job_type != JobType.STATE_VECTOR:
            raise ResultAttributeError(
                "Cannot get state vector if the job was not of type STATE_VECTOR"
            )
        assert self._state_vector is not None
        return self._state_vector

    @property
    def samples(self) -> list[Sample]:
        """Get the list of samples of the result"""
        if self.job.job_type != JobType.SAMPLE:
            raise ResultAttributeError(
                "Cannot get samples if the job was not of type SAMPLE"
            )
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
        assert self._probabilities is not None
        return self._probabilities

    @property
    def counts(self) -> list[int]:
        """Get the list of counts for each sample of the experiment"""
        if self.job.job_type != JobType.SAMPLE:
            raise ResultAttributeError(
                "Cannot get counts if the job was not of type SAMPLE"
            )

        assert self._counts is not None
        return self._counts

    def __str__(self):
        header = f"Result: {type(self.device).__name__}, {self.device.name}"
        if self.job.job_type == JobType.SAMPLE:
            samples_str = ("\n" + " " * 16).join(map(str, self.samples))
            cleaned_probas = str(self._probabilities).replace("\n", " ")
            return header + dedent(
                f"""
                Counts: {self._counts}
                Probabilities: {cleaned_probas}
                {samples_str}
                Error: {self.error}\n\n"""
            )
        if self.job.job_type == JobType.STATE_VECTOR:
            return f"""{header}\n{self._state_vector}\n\n"""
        if self.job.job_type == JobType.OBSERVABLE:
            return header + dedent(
                f"""
                Expectation value: {self.expectation_value}
                Error/Variance: {self.error}\n\n"""
            )
        raise NotImplementedError(
            f"Job type {self.job.job_type} not implemented for __str__ method"
        )


@typechecked
class BatchResult:
    """Class used to handle several Result instances.

    Args:
        results: List of results.

    Example:
        >>> result1 = Result(
        ...     Job(JobType.STATE_VECTOR,QCircuit(0),ATOSDevice.MYQLM_PYLINALG),
        ...     StateVector(np.array([1, 1, 1, -1])/2, 2),
        ...     0,
        ...     0
        ... )
        >>> result2 = Result(
        ...     Job(
        ...         JobType.SAMPLE,
        ...         QCircuit(0),
        ...         ATOSDevice.MYQLM_PYLINALG,
        ...         BasisMeasure([0,1],shots=500)
        ...     ),
        ...     [Sample(2, index=0, count=250), Sample(2, index=3, count=250)],
        ...     0.034,
        ...     500)
        >>> result3 = Result(
        ...     Job(JobType.OBSERVABLE,QCircuit(0),ATOSDevice.MYQLM_PYLINALG),
        ...     -3.09834,
        ...     0.021,
        ...     2048
        ... )
        >>> batch_result = BatchResult([result1, result2, result3])
        >>> print(batch_result)
        BatchResult: 3 results
        Result: ATOSDevice, MYQLM_PYLINALG
        State vector: [ 0.5+0.j  0.5+0.j  0.5+0.j -0.5+0.j]
                Probabilities: [0.25 0.25 0.25 0.25]
                Number of qubits: 2
        Result: ATOSDevice, MYQLM_PYLINALG
        Counts: [250, 0, 0, 250]
        Probabilities: [0.5 0.  0.  0.5]
        State: 00, Index: 0, Count: 250, Probability: 0.5
        State: 11, Index: 3, Count: 250, Probability: 0.5
        Error: 0.034
        Result: ATOSDevice, MYQLM_PYLINALG
        Expectation value: -3.09834
        Error/Variance: 0.021
        >>> print(batch_result[0])
        Result: ATOSDevice, MYQLM_PYLINALG
            State vector: [ 0.5+0.j  0.5+0.j  0.5+0.j -0.5+0.j]
            Probabilities: [0.25 0.25 0.25 0.25]
            Number of qubits: 2
    """

    def __init__(self, results: list[Result]):
        self.results = results
        """See parameter description."""

    def __str__(self):
        header = f"BatchResult: {len(self.results)} results\n"
        body = "".join(map(str, self.results))
        return header + body

    def __repr__(self):
        return str(self)

    def __getitem__(self, index: int):
        return self.results[index]

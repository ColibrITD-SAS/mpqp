"""Exercise the pasted integration without importing the external HDES project."""

import ast
import itertools
import threading
import time
import warnings
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
from scipy.optimize import Bounds
from tqdm import tqdm

from mpqp import QCircuit, IBMDevice, ExpectationMeasure, Observable, pZ
from mpqp.core.instruction import Barrier, BasisMeasure, H, Rz, Ry
from mpqp.execution.job import ExecutionMode, JobType
from mpqp.execution.vqa import Optimizer, OptimizerData, VQAModule


class RunMode(Enum):
    IDEAL = 1
    SHOT = 2
    STATE_VECTOR = 3
    CLASSICAL = 4


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("method", [Optimizer.TRF, Optimizer.BFGS])
@pytest.mark.parametrize("with_boundary", [False, True])
def test_demo_psr_matches_finite_difference(depth, method, with_boundary):
    source = Path(__file__).resolve().parents[2] / "vqa_demo.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ] + [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    ns = dict(globals())
    theta = sp.Symbol("theta")
    ns["vqc"] = lambda **kwargs: (QCircuit([Ry(theta, 0)]), [theta])
    ns["pre_transpile_observables"] = lambda *args: ExpectationMeasure(Observable(pZ))
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    names = ["F_1", "F_0"]  # Deliberately non-sorted parameter order.
    functions = {
        name: SimpleNamespace(nb_qubits=1, depth=depth, obs_list=["Z"])
        for name in names
    }
    scales = {name: sp.Symbol(name.replace("F", "g")) for name in names}
    group = SimpleNamespace(
        functions=functions,
        scaling_dict=scales,
        bc_mode="BC.LOSS" if with_boundary else "BC.SHIFT",
    )
    ansatz = SimpleNamespace(ansatz=SimpleNamespace(is_floquet=lambda: depth > 1))
    ns["generate_circuit"](group, ansatz, IBMDevice.AER_SIMULATOR, RunMode.STATE_VECTOR)
    assert functions[names[0]].angles != functions[names[1]].angles

    matrices = {name: {(0,): np.array([[1.0], [2.0]])} for name in names}
    point = np.array([1.2, 0.23, 0.8, -0.34])

    def residuals_at(point):
        values = []
        for i in range(2):
            values.extend(
                point[2 * i] * np.array([1.0, 2.0]) * np.cos(depth * point[2 * i + 1])
            )
        if with_boundary:
            values.append(point[0] * np.cos(depth * point[1]) ** 2 - 0.4)
        return np.array(values)

    params = {name: point[2 * i : 2 * i + 2] for i, name in enumerate(names)}
    fx = {
        name: {(0,): np.array([1.0, 2.0]) * np.cos(depth * params[name][1])}
        for name in names
    }
    residuals = {name: params[name][0] * fx[name][(0,)] for name in names}
    boundary = {"Bc_1": np.array([residuals_at(point)[-1]])} if with_boundary else {}
    problem = SimpleNamespace(
        result=SimpleNamespace(
            current_params_dict=params,
            fx_dict=fx,
            non_squared_residuals_dict=residuals,
            bc_res_unsquared_dict=boundary,
        ),
        eq_group=group,
        x_list=[[0.0], [1.0]],
        x_symbols=sp.symbols("x_0:1"),
        coeff_matrices_stacked=matrices,
        original_params=params,
        Theta_derivatives_dict={},
        g_derivatives_dict={},
        f_theta_dict={name: {(0,): None} for name in names},
        _vqa_method=method,
        _vqa_shots=0,
        backend=IBMDevice.AER_SIMULATOR,
    )
    ns["compute_scaled_derivatives"] = lambda *args: {
        name: {
            scales[name]: (
                fx[name][(0,)]
                if method == Optimizer.TRF
                else 2 * residuals[name] * fx[name][(0,)]
            )
        }
        for name in names
    }
    ns["compute_theta_derivatives"] = lambda td, subs, points, fx, deriv, xs: {
        name: {"theta": params[name][0] * deriv[name][(0,)]} for name in names
    }
    ns["compute_residuals_precomputed"] = lambda fx, xs, p, sub, eq: (
        {},
        {},
        {"Bc_1": p[names[0]][0] * fx[names[0]][(0,)][:1] ** 2 - 0.4},
    )
    actual = ns["compute_psr_jacobian"](problem)
    step = 1e-6
    expected = np.column_stack(
        [
            (residuals_at(point + step * unit) - residuals_at(point - step * unit))
            / (2 * step)
            for unit in np.eye(len(point))
        ]
    )
    if method != Optimizer.TRF:
        expected = 2 * residuals_at(point) @ expected
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    cached = functions[names[0]]._psr_vqa
    ns["compute_psr_jacobian"](problem)
    assert functions[names[0]]._psr_vqa is cached


@pytest.mark.parametrize("bounds", [None, (1.0, 10.0)])
def test_demo_optimization_stages(bounds):
    source = Path(__file__).resolve().parents[2] / "vqa_demo.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ] + [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    ns = dict(globals())
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
    problem = ns["ProblemLocal"].__new__(ns["ProblemLocal"])
    theta, scale = sp.symbols("theta scale")
    problem.run_mode = RunMode.IDEAL
    problem.backend = IBMDevice.AER_SIMULATOR
    problem.original_params = {"F_0": np.array([1.2, 0.3])}
    problem.eq_group = SimpleNamespace(
        scaling_dict={"F_0": scale},
        functions={
            "F_0": SimpleNamespace(
                circuit=QCircuit([Ry(theta, 0)]),
                angles=[theta],
                obs_mpqp=ExpectationMeasure(Observable(pZ)),
            )
        },
    )

    class MiscOptimizer(Enum):
        TRF = "trf"

    problem.optimizers = [
        SimpleNamespace(
            method=MiscOptimizer.TRF,
            psr=False,
            maxiter=4,
            bounds=bounds,
            nb_shots=32,
        )
    ] * 2
    problem.x0 = np.array([1.2, 0.3])
    problem.result = SimpleNamespace(optimizer_results=[], cost=[], cost_tot=0.0)
    problem.callback = None
    problem.save_run_data = lambda lock: None
    shot_counts = []

    def residual_callback(point, results, problem_local):
        shot_counts.append(results[0].shots)
        residual = np.array(
            [point[0] - 2.0, float(results[0].expectation_values) - 0.7]
        )
        problem_local.result.cost_tot = float(residual @ residual)
        return residual

    ns["vqa_residual_function"] = residual_callback
    result = problem.run_optimization()
    assert len(result.optimizer_results) == 2
    assert len(problem.x0) == 2
    assert shot_counts and set(shot_counts) == {0}
    if bounds is not None:
        assert 1.0 <= problem.x0[0] <= 10.0

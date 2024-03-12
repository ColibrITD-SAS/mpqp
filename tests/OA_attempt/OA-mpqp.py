# %%
import re
from math import floor

# import cma
import numpy as np
import scipy.special as ss
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from mpqp.all import *
from mpqp.execution.vqa.vqa import OptimizerInput


def cheb_mono(x: float, n: int) -> float:
    """This function returns the `n`-th order Chebyshev monomial

    Params:
        x: value the polynomial is evaluated at
        n: order of the Chebyshev polynomial
    """
    cos, acos = (np.cos, np.arccos) if abs(x) < 1 else (np.cosh, np.arccosh)
    factor = 1 if x > -1 else (-1) ** n
    return factor * cos(n * acos(x))


def factorial(n: int) -> float:  # TODO dirty fix
    """This function returns the factorial of any integer number n.

    Params:
        n: number to calculate the factorial of
    """
    res = ss.factorial(n)
    assert isinstance(res, float)
    return res


def cheb_poly(x: float, n: int, order: int) -> float:
    """This function returns the `n`-th order Chebyshev polynomial or its `order`
    derivative

    Params:
        x: value the polynomial is evaluated at
        n: order of the Chebyshev polynomial
        order: order of the derivative
    """
    if order == 0:
        return cheb_mono(x, n)

    def bin_factor(k: int):
        return ss.binom((n + order - k) / 2 - 1, (n - order - k) / 2)

    def factor(k: int):
        return ss.factorial((n + order + k) / 2 - 1)

    def denominator(k: int):
        return ss.factorial((n - order + k) / 2) * (2 if k == 0 else 1)

    return (
        2**order
        * n
        * sum(
            [
                bin_factor(k) * factor(k) / denominator(k) * cheb_mono(x, k)
                for k in range((n - order) % 2, n - order + 1, 2)
            ]
        )
    )


def diagonal_observable(
    x: float,
    n_qubits: int,
    order: int,
) -> Observable:
    """Returns the observable encoding the polynomial expansion.

    Params:
        x: value the polynomial is evaluated at
        n_qubits: number of qubits
        order: order of the derivative
    """
    return Observable(
        np.diag([cheb_poly(x, index, order) for index in range(2 ** (n_qubits - 1))])
    )


def vqc(
    parameters: OptimizerInput,
    depth: int,
    n_qubits: int,
) -> QCircuit:
    """This function returns a Variational Quantum Circuit with different
    versions of the 'hardware efficient' Ansatz.

    Params:
        parameters: parameters of the VQC to optimize.
        n_qubits: number of qubits
        d: depth of the VQC
    """
    return QCircuit(
        sum(
            [
                [Ry(parameters[n_qubits * d + i], i) for i in range(n_qubits)]
                + [CNOT(k, k + 1) for k in range(0, n_qubits - 1, 2)]
                + [CNOT(k, k + 1) for k in range(1, n_qubits - 1, 2)]
                for d in range(depth)
            ],
            [],
        )
    )


def extract_orders(expressions: list[str]) -> dict[str, int]:
    """
    This function returns the maximum order of the PDEs for each function.

    The format of an expression should be a string version of a python expression, except for the
    function and the derivatives, which are represented by a string composed of "d" (for derivative),
    the order of the derivative and the name of the function.

    Params:
        expressions: list of PDEs in string format
    """
    functions = set().union(
        *[
            set(map(lambda m: m.group(1), re.finditer(r"d\d+(\w+)", expr)))
            for expr in expressions
        ]
    )
    orders = {}
    sort_orders = {}
    for function in functions:
        orders_func = []
        for expression in expressions:
            matches = list(re.finditer(r"d(?P<order>\d+)" + function, expression))
            orders_func.append(
                max(map(lambda m: int(m.group("order")), matches), default=0)
            )
        orders[function] = max(orders_func)
        ord_keys = list(orders.keys())
        ord_keys.sort()
        sort_orders = {i: orders[i] for i in ord_keys}
    return sort_orders


nb_qubits = 2
depth = 2
nb_params = nb_qubits * depth

n_samples, x_min, x_max = 20, 0, 0.95
x_samples = np.linspace(x_min, x_max, n_samples)

epsilon_0 = 0.1
sigma_0 = 5
n = 4
b = 10
K = 100
# expressions = ['d1f - d0g/(3*K) - (2/np.sqrt(3))*epsilon_0*(d0g/(np.sqrt(3)*sigma_0))**n', 'd1g + b']
l = 8
k = 0.1
# expressions = ["d1f + l*d0f*(k + np.tan(l*x))"]
expressions = ["d1f - 1"]


functions = ["f", "g", "h", "k"][: len(expressions)]
orders = extract_orders(expressions)
boundaries = {"d0f": [0.0, 1.0]}
# boundaries = {"d0f": [0.0, 1.0], "d0g": [0.95, 2.]}

_nb_variables = len(functions) * nb_params + len(functions)

shots = None


def expectationValues(estimator, _attemptParameters, _shots=None):
    scaling_factor = _attemptParameters[: len(functions)]
    listExpectationValues = []
    for indexFunction in range(len(functions)):
        gamma = scaling_factor[indexFunction]
        angles = _attemptParameters[
            len(functions)
            + indexFunction * nb_params : len(functions)
            + (indexFunction + 1) * nb_params
        ]
        circ = vqc(angles, depth, nb_qubits)

        for order in range(orders[indexFunction] + 1):
            for sample in x_samples:
                observable = gamma * diagonal_observable_cheb(sample, nb_qubits, order)
                psi = circ
                job = estimator.run([psi], [observable], shots=_shots)
                listExpectationValues.append(job.result().values[0])

    return listExpectationValues


def cost_function(_attemptParameters, estimator):
    _expectationValues = expectationValues(estimator, _attemptParameters, shots)
    i = 0
    for indexFunction in range(len(functions)):
        for order in range(orders[indexFunction] + 1):
            locals()["_d" + str(order) + functions[indexFunction]] = np.array(
                _expectationValues[i * n_samples : (i + 1) * n_samples]
            )
            i += 1

    for derivative, coordinate in boundaries.items():
        indexCoordinate = floor(coordinate[0] * n_samples / (x_max - x_min))
        if indexCoordinate == n_samples:
            indexCoordinate -= 1
        shift = coordinate[1] - locals()["_" + derivative][indexCoordinate]
        for indexFunction in range(len(functions)):
            for order in range(orders[indexFunction] + 1):
                if derivative.find("d" + str(order) + functions[indexFunction]) != -1:
                    locals()["_d" + str(order) + functions[indexFunction]] += shift

    loss = 0
    for index in range(n_samples):
        x = x_samples[index]
        for expression in expressions:
            for indexFunction in range(len(functions)):
                for order in range(orders[indexFunction] + 1):
                    if (
                        expression.find("d" + str(order) + functions[indexFunction])
                        != -1
                    ):
                        locals()[
                            "d" + str(order) + functions[indexFunction]
                        ] = locals()["_d" + str(order) + functions[indexFunction]][
                            index
                        ]
        for expression in expressions:
            loss += abs(eval(expression)) ** 2

    return loss  # /n_samples #+ loss_BC


init_theta_list = np.random.uniform(low=0, high=2 * 3.14, size=(_nb_variables,))

# %%
# _options = cma.CMAOptions()
# _options.set('tolfun', 1e-3)
# opt = cma.fmin(cost_function, x0=init_theta_list, sigma0=1.5, options = _options)

method = "BFGS"
options = {"disp": True, "maxiter": 10}
# def callback(param_list): #LOCAL
#     current_loss = cost_function(param_list) #LOCAL
#     # loss_iterations.append(current_loss)    #LOCAL
#     print("current loss: ",current_loss,"\n") #LOCAL

# estimator=Estimator() #LOCAL

service = QiskitRuntimeService()  # IBMQ REMOTE
session = Session(service, backend="ibmq_qasm_simulator")  # IBMQ REMOTE
estimator = Estimator(session=session, options=None)  # IBMQ REMOTE

opt = minimize(
    cost_function, init_theta_list, args=(estimator), method=method, options=options
)
# %%
parameters = opt.x

# %%
scaling_factor = parameters[: len(functions)]
for index in range(len(functions)):
    angles = parameters[
        len(functions) + index * nb_params : len(functions) + (index + 1) * nb_params
    ]
    circ = vqc(angles, depth, nb_qubits)

    solution_attempt = []

    for sample in x_samples:
        observable = scaling_factor[index] * diagonal_observable_cheb(
            sample, nb_qubits, 0
        )
        psi = circ
        job = estimator.run([psi], [observable])
        locals()[functions[index]] = job.result().values[0]
        solution_attempt.append(locals()[functions[index]])

    solution_attempt = np.array(solution_attempt)
    indexCoordinate = int(
        floor(boundaries["d0" + functions[index]][0] * n_samples / (x_max - x_min))
    )
    shift = boundaries["d0" + functions[index]][1] - solution_attempt[indexCoordinate]

    solution_attempt += shift
    locals()[functions[index]] = solution_attempt


# %%

# def target_function_f(_x):
#     return np.exp(-0.1*8*_x) * np.cos(8*_x)


# def target_function_f(_x):
#     return _x


# def target_function_f(_x):
#     # return np.exp(_x)
#     return np.exp(-k * _x * l) * np.cos(l * _x)
def target_function_f(_x):
    return _x + 1


# def target_function_g(_x):
#     return (1/5)*( - 5*_x - 10*(_x**2) - 10*(_x**3) -5*(_x**4) - (_x**5) )

# solutions MD
# def target_function_f(_x):
#      return x_samples*(0.337217 - 0.563122*x_samples + 0.496778*(x_samples**2) - 0.225808*(x_samples**3) + 0.041056*(x_samples**4))
# def target_function_g(_x):
#      return 11 - 10*x_samples

for function in functions:
    plt.plot(
        x_samples,
        locals()["target_function_" + function](x_samples),
        linewidth=4,
        label="Target",
        color="Blue",
    )

    plt.ylabel("y")
    plt.plot(x_samples, locals()[function], label="Attempt OA", color="orange")
    plt.legend()
    plt.savefig(
        "TAN_CMA_" + str(nb_qubits) + "q" + str(depth) + "d" + str(shots) + "tol-2.png",
        dpi=300,
    )

# %%
# session.close() #IBMQ REMOTE

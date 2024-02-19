# %%
import re

# import cma
import numpy as np
import scipy.special as ss
from matplotlib import pyplot as plt
from math import floor
from scipy.optimize import minimize

from mpqp.all import *


# %%
def Cheb(_x, _n):
    return np.cos(_n * np.arccos(_x))


def Chebyshev(_x, _n, _nOrder):
    der = 0
    if _nOrder == 0:
        return Cheb(_x, _n)
    else:
        for k in range((_n - _nOrder) % 2, _n - _nOrder + 1, 2):
            binomialFactor = ss.binom(
                (_n + _nOrder - k) / 2 - 1, (_n - _nOrder - k) / 2
            )
            factor = ss.factorial((_n + _nOrder + k) / 2 - 1)
            denominator = ss.factorial((_n - _nOrder + k) / 2)
            if k == 0:
                der += binomialFactor * factor / (2 * denominator) * Cheb(_x, k)
            else:
                der += binomialFactor * factor / denominator * Cheb(_x, k)
        return 2**_nOrder * _n * der


# %%
def diagonal_observable_cheb(_x, _nQubits, _nOrder):
    diagonal_elements_plus = np.array(
        [Chebyshev(_x, index, _nOrder) for index in range(2 ** (_nQubits - 1))]
    )
    diagonal_elements_minus = -diagonal_elements_plus

    diagonal_elements = np.concatenate(
        (diagonal_elements_plus, diagonal_elements_minus), axis=None
    )
    # print(diagonal_elements)
    return Operator(np.diag(v=diagonal_elements))


# %%
def vqc(list_parameters, _depth, _nQubits):
    """
    Generates a VQC of depth d, for N qubits, with list_parameters as angles.

    :param list_parameters: LIST or np.ARRAY of OPTIMIZABLE parameters. It should have length 3Nd elements in [0,2pi]
    :param N: INTEGER, number of qubits
    :param d: INTEGER, depth of the vqc
    :return: a circuit corresponding to (vqa)|0>
    """
    reg = QuantumRegister(_nQubits)
    _qc = QuantumCircuit(reg)

    for d in range(_depth):
        for index, r in enumerate(reg):
            _qc.ry((list_parameters[_nQubits * d + index]), r)

        [_qc.cx(reg[k], reg[k + 1]) for k in range(_nQubits - 1)]

    return _qc


# %%
def extractOrders(_expressions):  # works up to order 9
    functions = ["f", "g", "h", "k"][: len(_expressions)]

    orders = []
    for expression in _expressions:
        orderFunctions = []
        for function in functions:
            listDerivatives = re.findall(r"d+\w" + function, expression)
            listDerivatives.sort(reverse=True)
            listDerivatives = [
                derivative.replace(function, "") for derivative in listDerivatives
            ]
            listDerivatives = [
                derivative.replace("d", "") for derivative in listDerivatives
            ]
            if len(listDerivatives) == 0:
                orderFunctions.append(0)
            else:
                orderFunctions.append(int(listDerivatives[0]))
        orders.append(orderFunctions)
    orders = np.array(orders)

    maxOrderFunctions = [max(orders[:, i]) for i in range(len(_expressions))]
    return maxOrderFunctions


# %%
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
orders = extractOrders(expressions)
boundaries = {"d0f": [0.0, 1.0]}
# boundaries = {"d0f": [0.0, 1.0], "d0g": [0.95, 2.]}

_nb_variables = len(functions) * nb_params + len(functions)

shots = None


# %%
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


# %%
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


# %%
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

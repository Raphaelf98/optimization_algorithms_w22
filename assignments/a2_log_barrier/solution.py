import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np




def solve(nlp: NLP):
    """
    solver for constrained optimization (cost term and inequalities)


    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---
    See the coding assignment PDF in ISIS.


    Notes:
    ---

    Get the starting point with:

    x = nlp.getInitializationSample()

    To know which type (cost term or inequalities) are the entries in
    the feature vector, use:
    types = nlp.getFeatureTypes()

    Index of cost term
    id_f = [ i for i,t in enumerate(types) if i == OT.f ]
    There is only one term of type OT.f ( len(id_f) == 1 )

    Index of inequality constraints:
    id_ineq = [ i for i,t in enumerate(types) if i == OT.ineq ]

    Get all features (cost and constraints) with:

    y,J = nlp.evaluate(x)
    H = npl.getFHessian(x)

    The value, gradient and Hessian of the cost are:

    y[0], J[0], getFHessian

    The value and Jacobian of inequalities are:
    y[id_ineq], J[id_ineq]


    """

    types = nlp.getFeatureTypes()
    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]

    x = nlp.getInitializationSample()

    #
    # Write your code Here
    #


    return x

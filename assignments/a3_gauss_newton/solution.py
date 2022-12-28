import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np






def solve(nlp: NLP):
    """
    solver for unconstrained optimization, including least squares terms

    Arguments:
    ---
        nlp: object of class NLP that contains one feature of type OT.f,
            and m features of type OT.r

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---
    See the coding assignment PDF in ISIS.


    Notes:
    ---

    Get the starting point with:

    x = nlp.getInitializationSample()

    You can query the problem with:

    y,J = nlp.evaluate(x)
    H = npl.getFHessian(x)

    To know which type (normal cost or least squares) are the entries in
    the feature vector, use:

    types = nlp.getFeatureTypes()

    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]

    The total cost is:

    y[self.id_f[0]] + np.dot(y[self.id_r], y[self.id_r])

    Note that getFHessian(x) only returns the Hessian of the term of type OT.f.

    For example, you can get the value and Jacobian of the least squares terms with y[id_r] (1-D np.array), and J[id_r] (2-D np.array).

    The input NLP contains one feature of type OT.f (len(id_f) is 1) (but sometimes f = 0 for all x).
    If there are no least squares terms, the lists of indexes id_r will be empty (e.g. id_r = []).

    """
    x = nlp.getInitializationSample()

    types = nlp.getFeatureTypes()

    #
    # Write your code Here
    #


    return x

import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np


def solve(nlp: NLP):
    """
    Gradient descent with backtracking Line search


    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---

    Implement a solver that does iterations of gradient descent
    with a backtracking line search

    x = x - k * Df(x),

    where Df(x) is the gradient of f(x)
    and the step size k is computed adaptively with backtracking line search

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Use the following to query the problem:
    phi, J = nlp.evaluate(x)

    phi is a vector (1D np.ndarray); use phi[0] to access the cost value 
    (a float number). 

    J is a Jacobian matrix (2D np.ndarray). Use J[0] to access the gradient 
    (1D np.array) of phi[0].

    """
    x = nlp.getInitializationSample()
    phi, J = nlp.evaluate(x)
    
    """
    init params
    """
    rho_alpha_plus = 1.2
    rho_alpha_minus = 0.5
    delta_max = np.inf
    rho_ls = 0.01
    theta = 1e-3
    alpha= 1.0
    delta = 1
    iter = 0
    
    while np.linalg.norm(alpha*delta) >= theta:
        phi,J = nlp.evaluate(x)
        delta = -J/np.linalg.norm(J)
        
        while nlp.evaluate(x+alpha*delta[0])[0] > phi +rho_ls*np.dot(J[0],alpha*delta[0]):
            alpha = alpha*rho_alpha_minus
            
        x += alpha*delta[0]
        alpha = min(rho_alpha_plus*alpha,delta_max)
        

    # sanity check on the input nlp
    assert len(nlp.getFeatureTypes()) == 1
    assert nlp.getFeatureTypes()[0] == OT.f

    # get start point
    #x = nlp.getInitializationSample()

    #
    # Write your code here
    #
    print(f'finished after #{iter} iterations at x:{x}')
    return x

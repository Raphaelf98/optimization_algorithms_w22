import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np


def solve(nlp: NLP):
    """
    Solver for unconstrained optimization


    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---

    See instructions and requirements in the coding assignment PDF in ISIS.

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Use the following to query the function and gradient of the problem:
    phi, J = nlp.evaluate(x)

    phi is a vector (1D np.ndarray); use phi[0] to access the cost value 
    (a float number). 

    J is a Jacobian matrix (2D np.ndarray). Use J[0] to access the gradient 
    (1D np.array) of phi[0].

    Use getFHessian to query the Hessian.

    H = nlp.getFHessian(x)

    H is a matrix (2D np.ndarray) of shape n x n.


    """

    # sanity check on the input nlp
    assert len(nlp.getFeatureTypes()) == 1
    assert nlp.getFeatureTypes()[0] == OT.f

    # get start point
    x = nlp.getInitializationSample()

    #
    # Write your code here
    #
    phi, J = nlp.evaluate(x)
    
    """
    init params
    """
    rho_alpha_plus = 1.2
    rho_alpha_minus = 0.5
    rho_ls = 0.01
    theta = 1e-4
    alpha= 1.0
    delta = np.eye(2)
    iter = 0
    lam = 10e-3
    while np.linalg.norm(alpha*delta,np.inf) >= theta:
        phi,J = nlp.evaluate(x)
        
        A = nlp.getFHessian(x)
        min_eig_val = np.linalg.eigvalsh(A)[0]
        while True:
            try:
                np.linalg.cholesky(A)
                
            except:
                #print("non-pos-def fallback")
                A -= np.identity(A.shape[0])*(min_eig_val - lam)
                #print(f'A NEW: {A} updated with {np.linalg.eigvalsh(A)[0]}')

            else: 
                break

        delta = -np.linalg.inv(A)@J[0]
            

        
        while nlp.evaluate(x+alpha*delta)[0] > phi +rho_ls*np.dot(J[0],alpha*delta):
            alpha = alpha*rho_alpha_minus
            #print(alpha)
        x += alpha*delta
        alpha = min(rho_alpha_plus*alpha,1)
        iter += 1


    # return found solution
    #print(f'finished after #{iter} iterations at x:{x}')
    return x

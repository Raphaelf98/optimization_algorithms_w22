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
    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]


    phi, J = nlp.evaluate(x)

    def total_cost(x):
        phi, J = nlp.evaluate(x)
        return phi[id_f[0]] + np.dot(phi[id_r], phi[id_r])

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
        
        phi, J = nlp.evaluate(x)
        phi_total = phi[id_f[0]] + np.dot(phi[id_r], phi[id_r])
        H_sos = 2*J[id_r].T@J[id_r]
        J_total = J[id_f[0]] + 2*J[id_r].T@phi[id_r]
        H_f = nlp.getFHessian(x)

        min_eig_val = np.linalg.eigvalsh(H_f)[0]

        while True:
            try:
                np.linalg.cholesky(H_f)
                
            except:
                #print("non-pos-def fallback")
                H_f -= np.identity(H_f.shape[0])*(min_eig_val - lam)
                #print(f'A NEW: {A} updated with {np.linalg.eigvalsh(A)[0]}')

            else: 
                break


        A = H_f + H_sos

        delta = -np.linalg.inv(A)@J_total
            

        
        while total_cost(x+alpha*delta) > phi_total +rho_ls*np.dot(J_total,alpha*delta):

            alpha = alpha*rho_alpha_minus
            #print(alpha)
        x += alpha*delta
        alpha = min(rho_alpha_plus*alpha,1)
        iter += 1


    # return found solution
    #print(f'finished after #{iter} iterations at x:{x}')
    return x

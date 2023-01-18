import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np

class AugLag(NLP):
    def __init__(self, nlp : NLP, id_f,id_r, id_ineq ,id_eq,mu,nu,lambda_,kappa, epsilon,theta_log):
        self.nlp = nlp

        self.id_r = id_r
        self.id_f = id_f
        self.id_ineq = id_ineq
        self.id_eq = id_eq

        self.epsilon = epsilon
        self.theta_log  = theta_log

        if len(id_ineq) == 0:
            self.ineq_active = False
            
        else:
            self.ineq_active = True

        if len(id_eq) == 0:
            self.eq_active = False

        else:
            self.eq_active = True

        if len(self.id_r) == 0:
            self.sos_active = False

        else:
            self.sos_active =True
        
        self.x =0
        self.eq = np.inf
        self.ineq = np.inf
        """
        init params
        """
        self.mu = mu
        self.nu = nu
        #init with dims of ineq un eq
        self.lambda_ = lambda_[:,np.newaxis]
        self.kappa = kappa[:,np.newaxis]

   
    def updateNu(self, rho_nu_plus):

        self.nu = self.nu*rho_nu_plus

    def updateMu(self, rho_mu_plus):

        self.mu = self.mu*rho_mu_plus

    def ineqCondition(self):

        if not self.ineq_active:
            return False 
        else:
            max = np.max(self.ineq)
            return  max >= self.epsilon

    def terminate(self, delta_x):
        
        one = np.linalg.norm(delta_x) >= self.theta_log  
        two = self.ineqCondition() 
        three = self.eqCondition()
        if not one and not two and not three:
            return False
        else: 
            return True
    def eqCondition(self):

        if not self.eq_active:
            return False 
        else: 
            abs = np.linalg.norm(self.eq)
            return  abs >= self.epsilon

    def updateKappa(self, x):

        if self.eq_active:
            y,J = self.nlp.evaluate(x)
            self.eq = y[self.id_eq]
            self.eq = self.eq[:,np.newaxis]
            self.kappa = self.kappa + 2*self.nu*self.eq
    
    def updateLambda(self,x):
        
        if self.ineq_active:
            y,J = self.nlp.evaluate(x)
            self.ineq  = y[self.id_ineq]
            self.ineq  = self.ineq[:,np.newaxis]
            shape = (len(self.id_ineq),2)
            zeros = np.zeros(shape)
            tmp = self.lambda_ + 2*self.mu*self.ineq
            zeros[:,:-1] = tmp
            self.lambda_ = np.max(zeros, axis= 1)
            self.lambda_ = self.lambda_[:,np.newaxis]
            

    def evaluate(self,x):

        y,J = self.nlp.evaluate(x)
        # f(x)
        f = y[self.id_f]
        J_f = J[self.id_f]
        phi = np.array([f])
        Jacobian = J_f
        # sos² 
        if self.sos_active:
            sos = np.dot(y[self.id_r], y[self.id_r])
            J_sos = 2*J[self.id_r].T@y[self.id_r]
            phi += sos 
            Jacobian += J_sos

        # inequalities
        if self.ineq_active:
            ineq = y[self.id_ineq]
            J_ineq_sq = np.zeros(J_f.shape)
            J_ineq = np.zeros(J_f.shape)
            grad = J[self.id_ineq]
            grad = grad[:,np.newaxis]
            for i, j in enumerate(ineq):
                if j >= 0:
                    phi += self.mu*j*j + self.lambda_[i]*j
                    J_ineq_sq += grad[i]*j
                    J_ineq += self.lambda_[i]*grad[i]
            Jacobian += self.mu*2*J_ineq_sq + J_ineq

        #equalities
        if self.eq_active:
            eq = y[self.id_eq]
            eq = eq[:,np.newaxis]
            J_eq_sq = np.zeros(J_f.shape)
            J_eq = np.zeros(J_f.shape)
            for i, j in enumerate(J[self.id_eq]):
                j = j[np.newaxis,:]
                J_eq_sq += j*eq[i]    
                J_eq += self.kappa[i]*j
            three = self.nu*eq.T@eq
            four = self.kappa.T@eq
            phi += three + four
            Jacobian += 2*self.nu*J_eq_sq + J_eq
    
        return  phi[0] , Jacobian

    
        
    def getInitializationSample(self):

        return self.x

    def setInitializationSample(self,x):

        self.x =x

    def getHessian(self, x):

        y,J = self.nlp.evaluate(x)
        grad = J[self.id_ineq]
        grad = grad[:,np.newaxis]
        H = self.nlp.getFHessian(x)
        H_= np.zeros(np.shape(H))

        if self.sos_active:
            H_+= 2*J[self.id_r].T@J[self.id_r]

        # H inequalities
        if self.ineq_active:
            for i, j in enumerate(y[self.id_ineq]):
                if j >= 0:
                    H_ += 2*self.mu*grad[i].T@grad[i]
                    
        # H equalities
        if self.eq_active:
            for j in J[self.id_eq]:
                j = j[np.newaxis,:]
                H_ += 2*self.nu*j.T@j
        
        
        return H ,H_


    def getFeatureTypes(self):

        return self.nlp.getFeatureTypes()





def solve(nlp: NLP):
    """
    solver for constrained optimization


    Arguments:
    ---
        nlp: object of class NLP that contains features of type OT.f, OT.r, OT.eq, OT.ineq

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

    To know which type (normal cost, least squares, equalities or inequalities) are the entries in
    the feature vector, use:

    types = nlp.getFeatureTypes()

    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    id_eq = [i for i, t in enumerate(types) if t == OT.eq]

    Note that getFHessian(x) only returns the Hessian of the term of type OT.f.

    For example, you can get the value and Jacobian of the equality constraints with y[id_eq] (1-D np.array), and J[id_eq] (2-D np.array).

    All input NLPs contain one feature of type OT.f (len(id_f) is 1). In some problems,
    there no equality constraints, inequality constraints or residual terms.
    In those cases, some of the lists of indexes will be empty (e.g. id_eq = [] if there are not equality constraints).

    """
    
    

    x = nlp.getInitializationSample()

    types = nlp.getFeatureTypes()
    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    id_eq = [i for i, t in enumerate(types) if t == OT.eq]


    y,J = nlp.evaluate(x) 
    
    delta_x = 1.0
    theta_log = 10e-5
    epsilon = 10e-3

    rho_mu_plus = 1.2
    rho_nu_plus  = 1.2
    mu_0 = 1
    nu_0= 1
    lambda_ = np.zeros(np.shape(y[id_ineq]))
    kappa = np.zeros(np.shape(y[id_eq]))

    aug_lag = AugLag(nlp,id_f,id_r, id_ineq, id_eq, mu_0, nu_0, lambda_, kappa, epsilon,theta_log)
    first_iter = True
    x_prev = 0.0
    while aug_lag.terminate(delta_x):
        #print("outer")
        if first_iter:
            x = nlp.getInitializationSample()
            first_iter = False
            x_prev = nlp.getInitializationSample()

        aug_lag.setInitializationSample(x)
        x = solve_unconstrained(aug_lag)
        delta_x = x - x_prev
        #print("DELTA X ",delta_x)
        
        x_prev = x.copy()
        aug_lag.updateKappa(x)
        aug_lag.updateLambda(x)
        
        aug_lag.updateMu(rho_mu_plus)
        aug_lag.updateNu(rho_nu_plus)

    return x

def solve_unconstrained(nlp: NLP):
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

    while np.linalg.norm(alpha*delta[0],np.inf) >= theta:
        phi,J = nlp.evaluate(x)
    
        A, A_ = nlp.getHessian(x)
        min_eig_val = np.linalg.eigvalsh(A)[0]
        while True:
            try:
                np.linalg.cholesky(A)
                
            except:
                #print("non-pos-def fallback")
                A -= np.identity(A.shape[0])*(min_eig_val - lam)
                

            else: 
                break
        A = A + A_     
        delta = -np.linalg.inv(A)@J.T
        delta = delta.T
        while nlp.evaluate(x+alpha*delta[0])[0][0] > phi[0] +rho_ls*np.dot(J[0],alpha*delta[0]):
            alpha = alpha*rho_alpha_minus
            
        x += alpha*delta[0]
        alpha = min(rho_alpha_plus*alpha,1)
        iter += 1


    # return found solution
    #print(f'finished after #{iter} iterations at x:{x}')
    return x

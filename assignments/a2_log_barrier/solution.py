import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np


class LogBarrier(NLP):
    def __init__(self, nlp : NLP, id_f, id_ineq):
        self.nlp = nlp
        self.id_f = id_f
        self.id_ineq = id_ineq
        self.x =0
        self.mu = 1
        
    def setMu(self):
        self.mu = self.mu /2
    def evaluate(self,x):
        y,J = self.nlp.evaluate(x)
        barrier = self.getLogBarrierVals(x) 
        barrier_grad = self.getLogBarrierJacobian(x)
        
        return y[self.id_f[0]] + barrier, np.array([J[self.id_f[0]] + barrier_grad])

    def getInitializationSample(self):

        return self.x

    def setInitializationSample(self,x):
        self.x =x

    def violateConstraints(self,x):
        y,J = self.nlp.evaluate(x)
        exceed_bounds = False
        #print(-y[self.id_ineq])
        with np.errstate(invalid='raise'):
            try:
                #print(-y[self.id_ineq])
                barrier = -self.mu *np.sum(np.log(-y[self.id_ineq]))
            except: 
                #print("used np inf")
                barrier = np.inf
                exceed_bounds = True
            else: 
                exceed_bounds = False
        return exceed_bounds

    def getLogBarrierJacobian(self,x):
        #barrier_grad = 0
        #y,J = self.nlp.evaluate(x)
        #div =1./y[self.id_ineq]
        #print(div)
        #bla = J[self.id_ineq].T
        #div =np.tile(div,(bla.shape[0],1))
        #print("MULTIPLy")
        #print(div.T)
        #print(J[self.id_ineq])
        
        #li = np.multiply(J[self.id_ineq],div.T)

        #if not self.exceed_bounds:
        #    barrier_grad = -self.mu*np.sum(np.multiply(J[self.id_ineq],div.T), axis=0)
        
        y,J = self.nlp.evaluate(x)
        div =np.reciprocal(y[self.id_ineq])
        J_log = []
        for i,j in enumerate(J[self.id_ineq]):
            #J = J[self.id_ineq]
        
            #j = j[np.newaxis,:]

            J_log.append(-self.mu*j*div[i])
            
        J_log = np.sum(J_log, axis=0)

        if  self.violateConstraints(x):
            barrier_grad = -10e6*np.ones(np.shape(J_log))
            #print(y[self.id_ineq])
            #("exceeded constraints")
        else:
            barrier_grad = J_log

        return barrier_grad

    def getLogBarrierVals(self,x):
        y,J = self.nlp.evaluate(x)
        barrier = 0 
        #print(-y[self.id_ineq])
        with np.errstate(invalid='raise'):
            try:
                #print(-y[self.id_ineq])

                barrier = -self.mu *np.sum(np.log(-y[self.id_ineq]))

            except: 
                #print("used np inf")
                barrier = 10e6
                self.exceed_bounds = True
            else: 
                self.exceed_bounds = False
        return barrier
            

    def getFHessian(self, x):
        y,J = self.nlp.evaluate(x)
        ##numerical issue
        div = np.reciprocal(y[self.id_ineq])**2
    
        H = self.nlp.getFHessian(x)
        '''
        evaluate Hessian
        sum gradg@gradg.T/g²
    
        '''
        H_log =[]
        
        for i,j in enumerate(J[self.id_ineq]):
            #J = J[self.id_ineq]
        
            j = j[np.newaxis,:]
            
            H_log.append(self.mu*j.T@j*div[i])
            
        H_log = np.sum(np.array(H_log), axis=0)
        
        if not self.violateConstraints(x):
            H = H + H_log
        #else:
            #H = -10e6*np.ones(np.shape(H_log))
        #for i,grad in enumerate(jac):
        #    denominator = 1./vals**2
        #    H += grad@grad.T*denominator[i]
        
        return H


    def getFeatureTypes(self):
        return self.nlp.getFeatureTypes()

def solve(nlp: NLP):
    """
    solver for constrained optimization (cost term and inequalities)


    Arguments:
    ---
        nlp: object of class NLP that contains one feature of type OT.f,
            and m features of type OT.ineq.

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
    id_f = [ i for i,t in enumerate(types) if t == OT.f ]
    There is only one term of type OT.f ( len(id_f) == 1 )

    Index of inequality constraints:
    id_ineq = [ i for i,t in enumerate(types) if t == OT.ineq ]

    Get all features (cost and constraints) with:

    y,J = nlp.evaluate(x)
    H = npl.getFHessian(x)

    The value, gradient and Hessian of the cost are:

    y[id_f[0]] (scalar), J[id_f[0]], H

    The value and Jacobian of inequalities are:
    y[id_ineq] (1-D np.array), J[id_ineq]


    """

    types = nlp.getFeatureTypes()
    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]

    x = nlp.getInitializationSample()

    #
    # Write your code Here
    #

    y,J = nlp.evaluate(x)
    H = nlp.getFHessian(x)

    #The value, gradient and Hessian of the cost are:

    #y[id_f[0]] (scalar), J[id_f[0]], H

    #The value and Jacobian of inequalities are:
    #y[id_ineq] (1-D np.array), J[id_ineq]
    delta_x = 1.0
    theta_log = 10e-5
    

    log_bar = LogBarrier(nlp,id_f,id_ineq)
    first_iter = True
    x_prev = 0.0
    while np.linalg.norm(delta_x) >= theta_log:
        

        if first_iter:
            x = nlp.getInitializationSample()
            first_iter = False
            x_prev = nlp.getInitializationSample()
         

        log_bar.setInitializationSample(x)
        x = solve_unconstrained(log_bar)
        delta_x = x-x_prev
        x_prev = x.copy()
        log_bar.setMu()
        
        
    print(f'found solution at {x}')
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
    
    return x

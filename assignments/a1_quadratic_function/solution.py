import sys
sys.path.append("../..")


from optalg.interface.nlp import NLP
import numpy as np


class NLP_xCCx(NLP):
    """
    Nonlinear program with quadratic cost  x^T C^T C x
    x in R^n
    C in R^(m x n)
    ^T means transpose
    feature types: [ OT.f ]

    """
    delta = 1e-3
    def __init__(self, C):
        """
        """
        self.C = C
        try:
            self.dim = np.shape(C)[1]
        except: 
            self.dim = np.shape(C)[0]

    def f(self,x):
        return x.T@self.C.T@self.C@x

    def evaluate(self, x):
        """
        Returns the features and the Jacobians
        of a nonlinear program.
        In this case, we have a single feature (the cost function)
        because there are no constraints or residual terms.
        Therefore, the output should be:
            y: the feature (1-D np.ndarray of shape (1,)) 
            J: the jacobian (2-D np.ndarray of shape (1,n))

        See also:
        ----
        NLP.evaluate
        """

        y = self.f(x)
        J = self.finite_diff_grad(self.f, x, self.delta)

        return  np.array([y]) , J.reshape(1,-1)

    def getDimension(self):
        """
        Return the dimensionality of the variable x

        See Also
        ------
        NLP.getDimension
        """

        # n =
        return self.dim

    def getFHessian(self, x):
        """
        Returns the hessian of the cost term.
        The output should be: 
            H: the hessian (2-D np.ndarray of shape (n,n))

        See Also
        ------
        NLP.getFHessian
        """
        H = self.finite_diff_hess(self.f, x, self.delta)

        return H

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.getDimension())

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        return "Quadratic function x^T C^T C x "

    def finite_diff_grad(self, f, x, eps):
        n = x.size
        grad = np.zeros(n)
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = eps
            grad[i] = (f(x + ei) - f(x - ei)) / (2 * eps)
        return grad


    def finite_diff_hess(self,f, x, eps):
        """
        Arguments:
        ----
        f: function
        x: np.array 1-D
        eps: float

        Returns:
        ----
        hess: np.array 2-D

        """
        n = x.size
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                ei = np.zeros(n)
                ej = np.zeros(n)
                ei[i] = eps
                ej[j] = eps
                hess[i, j] = (f(x + ei + ej) + f(x - ei - ej)
                            - f(x + ei - ej) - f(x - ei + ej)) / (4 * eps * eps)
                if i != j:
                    hess[j, i] = hess[i, j]
        return hess

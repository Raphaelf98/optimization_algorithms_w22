import numpy as np
import sys
sys.path.append("../..")
from optalg.interface.nlp import NLP
import math


class RobotTool(NLP):
    """
    """

    def __init__(self, q0: np.ndarray, pr: np.ndarray, l: float):
        """
        Arguments
        ----
        q0: 1-D np.array
        pr: 1-D np.array
        l: float
        """
        self.q0 = q0
        self.pr = pr
        self.l = l
    def function(self,x):
        return np.cos(x[0]) + 0.5*np.cos(x[0]+x[1]) + (1/3+x[3])*np.cos(x[0]+x[1]+x[2]),np.sin(x[0]) + 0.5*np.sin(x[0]+x[1]) + (1/3+x[3])*np.sin(x[0]+x[1]+x[2])
    def phi(self,x):
        return np.array([np.sqrt(np.linalg.norm(self.function(x) - self.pr)**2 + self.l*np.linalg.norm(x-self.q0)**2)])
    def grad(self,x):
        dq1cos = -np.sin(x[0]) - 0.5*np.sin(x[0]+x[1]) - (1/3+x[3])*np.sin(x[0]+x[1]+x[2])
        dq1sin = np.cos(x[0]) + 0.5*np.cos(x[0]+x[1]) + (1/3+x[3])*np.cos(x[0]+x[1]+x[2])

        dq2cos = - 0.5*np.sin(x[0]+x[1]) - (1/3+x[3])*np.sin(x[0]+x[1]+x[2])
        dq2sin =  0.5*np.cos(x[0]+x[1]) + (1/3+x[3])*np.cos(x[0]+x[1]+x[2])

        dq3cos = - (1/3+x[3])*np.sin(x[0]+x[1]+x[2])
        dq3sin =  (1/3+x[3])*np.cos(x[0]+x[1]+x[2])

        dq4cos =  np.cos(x[0]+x[1]+x[2])
        dq4sin =  np.sin(x[0]+x[1]+x[2])

        grad_cos = np.array([dq1cos,dq2cos,dq3cos,dq4cos])
        grad_sin = np.array([dq1sin,dq2sin,dq3sin,dq4sin])

        return np.array([grad_cos,grad_sin])
    def evaluate(self, x):
        """
        """

        # y = ...
        a = 2*(self.function(x) - self.pr)
        
        a = a[np.newaxis,:]
        
        J = a@self.grad(x) + self.l*2*(x-self.q0)
        print(J)
        return  self.phi(x), J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return 4

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return self.q0

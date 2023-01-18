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
    
    def evaluate(self, x):
        """
        """
        y = np.empty(6)
        
        y[0] = self.function(x)[0] - self.pr[0]
        y[1] = self.function(x)[1] - self.pr[1]
        y[2] = np.sqrt(self.l)*(x[0] - self.q0[0])
        y[3] = np.sqrt(self.l)*(x[1] - self.q0[1])
        y[4] = np.sqrt(self.l)*(x[2] - self.q0[2])
        y[5] = np.sqrt(self.l)*(x[3] - self.q0[3])

        J = np.zeros((6,4))
        J[0, 0] = -np.sin(x[0]) - 0.5*np.sin(x[0]+x[1]) - (1/3+x[3])*np.sin(x[0]+x[1]+x[2])
        J[1, 0] = np.cos(x[0]) + 0.5*np.cos(x[0]+x[1]) + (1/3+x[3])*np.cos(x[0]+x[1]+x[2])
        J[0, 1] = - 0.5*np.sin(x[0]+x[1]) - (1/3+x[3])*np.sin(x[0]+x[1]+x[2])
        J[1, 1] = 0.5*np.cos(x[0]+x[1]) + (1/3+x[3])*np.cos(x[0]+x[1]+x[2])
        J[0, 2] = -(1/3+x[3])*np.sin(x[0]+x[1]+x[2])
        J[1, 2] = (1/3+x[3])*np.cos(x[0]+x[1]+x[2])
        J[0, 3] = np.cos(x[0]+x[1]+x[2])
        J[1, 3] = np.sin(x[0]+x[1]+x[2])
#
        J[2, 0] = np.sqrt(self.l)
        J[3, 1] = np.sqrt(self.l)
        J[4, 2] = np.sqrt(self.l)
        J[5, 3] = np.sqrt(self.l)
        return  y, J

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

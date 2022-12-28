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

    def evaluate(self, x):
        """
        """

        # y = ...
        # J = ...


        return y, J

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

import sys
sys.path.append("../..")
import numpy as np
from solution import solve
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.quarter_circle import QuaterCircle
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.interface.nlp_traced import NLPTraced

H = np.array([[1., -1.], [-1., 2.]])
g = np.array([-2., -6.])
Aineq = np.array([[1., 1.], [-1., 2.], [2., 1.]])
bineq = np.array([2., 2., 3.])
problem = QuadraticProgram(H=H, g=g, Aineq=Aineq, bineq=bineq)

problem.getInitializationSample = lambda: np.zeros(2)
x = solve(problem)
solution = np.array([0.66667, 1.3333])
print(x)

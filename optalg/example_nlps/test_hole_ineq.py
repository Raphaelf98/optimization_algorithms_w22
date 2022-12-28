import sys
sys.path.append("../..")

from optalg.example_nlps.hole_ineq import HoleIneq as Problem
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import *
import numpy as np
import unittest


class testHoleIneq(unittest.TestCase):

    problem = Problem

    def testJacobian(self):
        C = np.array([[2, 0], [0, 3]])
        A = np.array([[1, 2], [0, 4]])
        b = np.array([1.5, 2.5])
        a = .1
        problem = self.problem(C, a, A, b)
        x = np.array([0.1, 0.3])
        flag, _, _ = check_nlp(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)

    def testJacobian2(self):
        C = np.array([[2, 0], [0, 3]])
        A = np.array([[1, 2], [0, 4]])
        b = np.array([1.5, 2.5])
        a = 1.
        problem = self.problem(C, a, A, b)
        x = np.array([1.3, 2.8])
        print("error here")
        flag, _, _ = check_nlp(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)

    def testJacobian3(self):
        C = np.array([[2, 0], [0, 3]])
        A = np.array(([[1, 2], [0, 4]]))
        b = np.array([1.5, 2.5])
        a = 1.
        problem = self.problem(C, a, A, b)
        x = np.array([-0.5, 0.8])
        flag, _, _ = check_nlp(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)

    def testHessian(self):

        C = np.array([[2, 0], [0, 3]])
        A = np.array([[1, 2], [0, 4]])
        b = np.array([1.5, 2.5])
        a = 1.

        problem = self.problem(C, a, A, b)
        x = np.array([.5, .8])
        H = problem.getFHessian(x)

        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        self.assertTrue(np.allclose(H, Hdiff, 10 * tol, 10 * tol))


# usage:
# print results in terminal
# python3 test.py
# store results in file
# python3 test.py out.log
if __name__ == "__main__":
    if len(sys.argv) == 2:
        log_file = sys.argv.pop()
        with open(log_file, "w") as f:
            runner = unittest.TextTestRunner(f, verbosity=2)
            unittest.main(testRunner=runner)
    else:
        unittest.main()

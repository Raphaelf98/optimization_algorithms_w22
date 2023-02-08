import sys
sys.path.append("../..")
from optalg.interface.nlp_stochastic import NLP_stochastic
import numpy as np


def solve(nlp: NLP_stochastic):
    """
    stochastic gradient descent


    Arguments:
    ---
        nlp: object of class NLP_stochastic that contains one feature of type OT.f.

    Returns:
    ---
        x: local optimal solution (1-D np.ndarray)

    Task:
    ---
    See the coding assignment PDF in ISIS.

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Get the number of samples with:
    N = nlp.getNumSamples()

    You can query the problem with any index i=0,...,N (N not included)

    y, J = nlp.evaluate_i(x, i)

    As usual, get the cost function (scalar) and gradient (1-D np.array) with y[0] and J[0]

    The output (y,J) is different for different values of i and x.

    The expected value (over i) of y,J at a given x is SUM_i [ nlp.evaluate_i(x, i) ]  / N

    """
    
    x = nlp.getInitializationSample()
    N = nlp.getNumSamples()

    #
    # Write your code Here
    #
    rng = np.random.default_rng()

    N = nlp.getNumSamples()
    tolerance = np.array(10e-6)
    lambda_ = 0.3
    alpha_0 = 1

    k = 0
    max_query= 10000
    iter_ = int(max_query / N - 1)
    print(iter_)
    for _ in range(iter_): 
        
        for i in rng.integers(low = 0, high = N, size = N):
            
            phi , grad = nlp.evaluate_i(x,i)
            #print(phi)
            learn_rate = alpha_0 / (1 + alpha_0*lambda_*k)
            diff = - learn_rate * grad[0]
            k = k + 1
            if np.all(np.abs(diff) <= tolerance):
                break
            x += diff
    return x

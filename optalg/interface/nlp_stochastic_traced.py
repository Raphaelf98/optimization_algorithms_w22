from .nlp import NLP


class NLPStochasticTraced(NLP):

    def __init__(self, nlp, max_evaluate=0, max_evaluate_i=1e4):
        self.nlp = nlp

        self.counter_evaluate_i = 0
        self.counter_evaluate = 0
        self.max_evaluate = max_evaluate
        self.max_evaluate_i = max_evaluate_i

        self.trace_x = []
        self.trace_phi = []
        self.trace_J = []

        self.trace_x_hessian = []

        super().__init__()

    def reset_counters(self):
        """
        """
        self.counter_evaluate = 0
        self.counter_hessian = 0
        self.counter_evaluate_i = 0

        self.trace_x = []
        self.trace_phi = []
        self.trace_J = []

        self.trace_x_hessian = []

    def evaluate(self, x):
        """
        """
        if self.counter_evaluate > self.max_evaluate:
            raise RuntimeError("too many iterations, returning")
        self.counter_evaluate += 1
        phi, J = self.nlp.evaluate(x)
        self.appendToTrace(x, phi, J)
        return phi, J

    def evaluate_i(self, x, i):
        """
        """
        if self.counter_evaluate_i > self.max_evaluate_i:
            raise RuntimeError("too many iterations, returning")
        self.counter_evaluate_i += 1
        phi, J = self.nlp.evaluate_i(x, i)
        self.appendToTrace(x, phi, J)
        return phi, J

    def appendToTrace(self, x, phi, J):
        """
        This should be called at the end of an evaluate implementation. It adds
        the evaluated x, features and objectives to the traces
        """
        self.trace_x.append(x.copy())
        self.trace_J.append(J.copy())
        self.trace_phi.append(phi.copy())

    def getBounds(self):
        """
        """
        return self.nlp.getBounds()

    def getDimension(self):
        """
        """
        return self.nlp.getDimension()

    def getFHessian(self, x):
        """
        """
        self.counter_hessian += 1
        self.trace_x_hessian.append(x)
        return self.nlp.getFHessian(x)

    def getFeatureTypes(self):
        """
        """
        return self.nlp.getFeatureTypes()

    def getInitializationSample(self):
        """
        """
        return self.nlp.getInitializationSample()

    def getNumSamples(self):
        """
        """
        return self.nlp.getNumSamples()

    def report(self, verbose):
        """
        """
        header = "Traced NLP Stochastic \n"
        out = self.nlp.report(verbose)
        return header + out

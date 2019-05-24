import itertools
import numpy as np
import numpy.linalg as la
from itertools import chain, combinations


class CellularAutomation:
    """Class created to solve Task 2.4."""

    def __init__(self):
        self.setup()

    def setup(self):
        """Sets up the X transpose matrix by getting the permutation triplet of -1 and +1,
        translating rule number 110 and 126 into first binary, then bipolar vectors
        """

        self.X_T = np.asarray(list(itertools.product([-1, 1], repeat=3)))
        self.rule_110_y = 2 * np.array(list('{0:08b}'.format(110))).astype('int') - 1
        self.rule_126_y = 2 * np.array(list('{0:08b}'.format(126))).astype('int') - 1

    def subtask_1(self):
        """ Direct implementation of least squares solution to rule 110 and 126
        """
        w_110 = la.lstsq(self.X_T, self.rule_110_y)[0]
        yhat_110 = np.dot(self.X_T, w_110)
        print "Rule 110: \n y     values are: {} \n y_hat values are: {}\n".format(self.rule_110_y.astype('float'), yhat_110)

        w_126 = la.lstsq(self.X_T, self.rule_126_y)[0]
        yhat_126 = np.dot(self.X_T, w_126)
        print "Rule 126: \n y     values are: {} \n y_hat values are: {}\n".format(self.rule_126_y.astype('float'), yhat_126)

    def subtask_2(self, var_list):
        """ Gets the powerset for a given vector, applies a map function to obtain
        basis function results for the given vector
        """
        power_set = self.get_powerset(var_list)
        result = list(map(self.apply_basis_func, power_set))
        return result

    def apply_basis_func(self, vec):
        """ Returns the corresponding result for a given powerset
        """
        vec_len = len(vec)
        if vec_len == 0:
            return 1
        elif vec_len == 1:
           return vec[0]
        else:
            return np.prod(vec)

    def get_powerset(self, set):
        """ Returns the powerset of a given set
        """
        return np.asarray(list(chain.from_iterable(combinations(set, n) for n in range(len(set) + 1))))

    def subtask_3(self):
        """ Transforms design matrix into feature matrix by applying the subtask 2
        into X transpose, then runs subtask 1 on the feature matrix
        """
        result_list = list()
        for input_row in self.X_T:
            result_list.append(ca.subtask_2(input_row.tolist()))
        self.X_T = np.array(result_list)
        self.subtask_1()

if __name__ == "__main__":
    ca = CellularAutomation()
    print "#############################################"
    print "###  Subtask 1   ############################\n"
    ca.subtask_1()
    print "#############################################"
    print "###  Subtask 3   ############################\n"
    ca.subtask_3()
    print "#############################################"

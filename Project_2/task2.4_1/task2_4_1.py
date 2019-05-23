import itertools
import numpy as np
import numpy.linalg as la
from itertools import chain, combinations


class CellularAutomation:
    def __init__(self):
        self.setup()

    def setup(self):
        self.input = np.array(list(list(tup) for tup in itertools.product([-1, 1], repeat=3)))
        rule_110_y = np.array(list('{0:08b}'.format(110))).astype('int')
        self.rule_110_y = self.correct_y_values(rule_110_y)
        rule_126_y = np.array(list('{0:08b}'.format(126))).astype('int')
        self.rule_126_y = self.correct_y_values(rule_126_y)

    def correct_y_values(self, y):
        return 2*y-1

    def subtask_1(self):
        X_T = self.input

        y110 = self.rule_110_y
        w110 = la.lstsq(X_T, y110)[0]
        yhat_110 = np.dot(X_T, w110)
        print "Rule 110: \n y values are: {} \n y^values are: {}".format(y110.astype('float'), yhat_110)

        y126 = self.rule_126_y
        w126 = la.lstsq(X_T, y126)[0]
        yhat_126 = np.dot(X_T, w126)
        print "Rule 126: \n y values are: {} \n y^values are: {}".format(y126.astype('float'), yhat_126)
        #TODO refactor

    def subtask_2(self, var_list):
        power_set = self.get_powerset(var_list)
        for index in range(len(power_set)):
            sublist_len = len(power_set[index])
            if sublist_len == 0:
                power_set[index] = 1
            elif sublist_len == 1:
                power_set[index] = power_set[index][0]
            else:
                power_set[index] = np.prod(power_set[index])
        return power_set

    def get_powerset(self, set):
        return list(list(tup) for tup in chain.from_iterable(combinations(set, n) for n in range(len(set) + 1)))

    def subtask_3(self):
        result_list = list()
        for input_row in self.input:
            result_list.append(ca.subtask_2(input_row.tolist()))
        self.input = np.array(result_list)
        self.subtask_1()


if __name__ == "__main__":
    ca = CellularAutomation()
    print "---------------------------------------------"
    print "Subtask 1"
    ca.subtask_1()
    print "---------------------------------------------"
    print "Subtask 3"
    ca.subtask_3()
    print "---------------------------------------------"

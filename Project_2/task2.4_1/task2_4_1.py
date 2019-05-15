import itertools
import numpy as np


class CellularAutomation:
    def __init__(self):
        self.setup()

    def setup(self):
        input_perm = list(itertools.product([1, -1], repeat=3))
        data_type = np.dtype('int,int,int')
        design_matrix_X = np.array(input_perm, dtype=data_type)
        rule_110_y = list('{0:08b}'.format(110)) #TODO change these to -1,+1
        rule_126_y = list('{0:08b}'.format(126))

    def subtask_1(self):
        pass

    def subtask_2(self):
        pass

    def subtask_3(self):
        pass


if __name__ == "__main__":
    ca = CellularAutomation()
    ca.subtask_1()
    ca.subtask_2()
    ca.subtask_3()

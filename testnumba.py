import numpy as np
import numba
from numba import jit
from numba import jitclass


spec = [
    ('x', numba.float32[:]),
]
@jitclass(spec)
class Test(object):
    def __init__(self, ptr):
        self.x = ptr

    def setX(self, val):
        for i in range(self.x.shape[0]):
            self.x[i] = val

    def printX(self):
        for i in range(self.x.shape[0]):
            print self.x[i]
l = np.ones(2, dtype=np.float32)
a = Test(l)
print l
a.setX(2.11)
print l
# Python program to illustrate the concept
# of threading
from threading import Thread
import os
from multiprocessing.pool import ThreadPool
import multithreading

import numpy as np


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


def task1(arr):
    arr1 = list(i for i in range(1, 10, 2))
    arr = np.copy(arr1)
    return arr
    pass


def task2(arr):
    arr1 = list(i for i in range(100, 200, 5))
    arr = np.copy(arr1)
    return arr


if __name__ == "__main__":

    arr1 = []
    arr2 = []
    # # creating threads
    # t1 = ThreadWithReturnValue(target=task1, args=(arr1,))
    # t2 = ThreadWithReturnValue(target=task2, args=(arr2,))

    # # starting threads
    # t1.start()
    # t2.start()

    # # wait until all threads finish
    # arr1 = t1.join()
    # arr2 = t2.join()
    # print(arr1)
    # print(arr2)
    pool = ThreadPool(processes=2)
    res = pool.apply(func=task1, args=(arr1,))
    res2 = pool.apply(func=task2, args=(arr2,))
    print(res, res2)
    print(arr1)

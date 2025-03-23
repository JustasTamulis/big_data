import timeit_wrapper as tw
import time

@tw.timeit
def test_timeit():
    time.sleep(1)
    return "Done"

if __name__ == '__main__':
    test_timeit()
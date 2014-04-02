from utils import *


def test_max_info_feature():
    X = np.eye(2)
    X[1,0] = 1
    y = np.array([1,0])
    print X
    print y
    print H(y)
    print max_info_feature(X,y,H(y))

<<<<<<< HEAD
    print max_info_feature(X,y)
=======

if __name__ == '__main__':
    test_max_info_feature()
>>>>>>> 18fe244a70e6bfc33db18e440ce77a142b5e975c

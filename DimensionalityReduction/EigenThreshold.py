import pickle as pi


# import numpy as np

def main():
    """
    Will compute the threshold.
    :return:
    """
    Sigma = pi.load(open("Matrix.dump"))
    print(Sigma.shape)


if __name__ == '__main__':
    main()
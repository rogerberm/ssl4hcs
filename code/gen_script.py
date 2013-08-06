#!/usr/bin/env python

import numpy as np
import sys


def main(cmdline):
    alpha_labeled = np.linspace(0, 0.5, 6)
    alpha_unlabeled = np.linspace(0.5, 1, 6)
    alpha_soft_uninf = np.linspace(0, 0.5, 6)
    alpha_soft_inf = np.linspace(0.3, 0.7, 5)
    knn = np.arange(4, 9)

    print len(alpha_labeled) * len(alpha_unlabeled) * len(alpha_soft_uninf) * len(alpha_soft_inf) * len(knn)

    grid = np.meshgrid(alpha_labeled, alpha_unlabeled, alpha_soft_uninf, alpha_soft_inf, knn)
    alpha_labeled_grid = grid[0].flatten()
    alpha_unlabeled_grid = grid[1].flatten()
    alpha_soft_uninf_grid = grid[2].flatten()
    alpha_soft_inf_grid = grid[3].flatten()
    knn_grid = grid[4].flatten()
    params = zip(alpha_labeled_grid, alpha_unlabeled_grid, alpha_soft_uninf_grid, alpha_soft_inf_grid, knn_grid)
    np.set_printoptions(precision=6)

    for p in params:
        print "bsub -n 4 './hcs.py -l all/labeled/gw*.arff -s all/soft/ -c -f 4 3 2 1 -L 1000 -v -al %f\
        -au %f -asu %f -asi %f -nf knn%i -q'" % (p[0], p[1], p[2], p[3], p[4])


if __name__ == "__main__":
    main(sys.argv[1:])

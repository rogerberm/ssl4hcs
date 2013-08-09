#!/usr/bin/env python

import numpy as np
import sys


def r(arr):
    a = arr
    reminder = np.array([], dtype=int)
    if len(arr) % 2 == 1:
        reminder = a[-1]
        a = a[:-1]
    a = np.vstack([np.flipud(a[:len(a) / 2]), a[len(a) / 2:]]).T.flatten()
    return np.hstack([a, reminder])


def main(cmdline):
    alpha_labeled = r(np.linspace(0.05, 0.25, 5))
    alpha_soft_uninf = r(np.linspace(0, 0.3, 4))
    alpha_soft_inf = r(np.linspace(0.3, 0.7, 5))
    alpha_unlabeled = r(np.linspace(0.5, 0.9, 5))
    knn = r(np.arange(4, 10))

    grid = np.meshgrid(alpha_labeled, alpha_soft_uninf, alpha_soft_inf, alpha_unlabeled, knn)
    alpha_labeled_grid = grid[0].flatten()
    alpha_soft_uninf_grid = grid[1].flatten()
    alpha_soft_inf_grid = grid[2].flatten()
    alpha_unlabeled_grid = grid[3].flatten()
    knn_grid = grid[4].flatten()
    params = zip(alpha_labeled_grid, alpha_soft_uninf_grid, alpha_soft_inf_grid, alpha_unlabeled_grid, knn_grid)
    print params
    np.set_printoptions(precision=5)

    for p in params:
        if p[0] <= p[1] and p[1] < p[2] and p[2] <= p[3]:
            for i in range(4):
                print "bsub -n 2 -q pub.36h -R \"rusage[mem=2048]\" './hcs.py -l all/labeled/gw*.arff -s all/soft/ -c -f 4 3 2 1 92 53 54 -L 200 -n 800 -v -al %f -au %f -asu %f -asi %f -nf knn%i -q'" % \
                    (p[0], p[1], p[2], p[3], p[4])


if __name__ == "__main__":
    main(sys.argv[1:])

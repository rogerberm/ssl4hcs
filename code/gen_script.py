#!/usr/bin/env python

import numpy as np
import sys

RECALL, A_LABELED, A_UNLABELED, A_S_UNINFECTED, A_S_INFECTED = 0, 2, 4, 6, 8

def r(arr):
    a = arr
    reminder = np.array([], dtype=int)
    if len(arr) % 2 == 1:
        reminder = a[-1]
        a = a[:-1]
    a = np.vstack([np.flipud(a[:len(a) / 2]), a[len(a) / 2:]]).T.flatten()
    return np.hstack([a, reminder])


def get_configurations(*list_ranges):
    grid = np.meshgrid(*list_ranges)
    grid_dimensions = [grid_dim.flatten() for grid_dim in grid]
    params = zip(*grid_dimensions)
    return params


def main(cmdline):
    alpha_labeled = r(np.linspace(0.01, 0.16, 6))
    alpha_soft_uninf = r(np.linspace(0.3, 0.99, 6))
    alpha_soft_inf = r(np.linspace(0.3, 0.99, 6))
    alpha_unlabeled = r(np.linspace(0.01, 0.66, 6))
    knn = r(np.arange(7, 8))

    grid = np.meshgrid(alpha_labeled, alpha_soft_uninf, alpha_soft_inf, alpha_unlabeled, knn)
    alpha_labeled_grid = grid[0].flatten()
    alpha_soft_uninf_grid = grid[1].flatten()
    alpha_soft_inf_grid = grid[2].flatten()
    alpha_unlabeled_grid = grid[3].flatten()
    knn_grid = grid[4].flatten()
    ref = {A_LABELED: 0.17, A_UNLABELED: 0.8350, A_S_UNINFECTED: 0.66, A_S_INFECTED: 0.82}
    params = []
    params += get_configurations(*[alpha_labeled, [ref[A_S_UNINFECTED]], [ref[A_S_INFECTED]], alpha_unlabeled, knn])
    params += get_configurations(*[alpha_labeled, [ref[A_S_UNINFECTED]], alpha_soft_inf, [ref[A_UNLABELED]], knn])
    params += get_configurations(*[alpha_labeled, alpha_soft_uninf, [ref[A_S_INFECTED]], [ref[A_UNLABELED]], knn])
    params += get_configurations(*[[ref[A_LABELED]], [ref[A_S_UNINFECTED]], alpha_soft_inf, alpha_unlabeled, knn])
    params += get_configurations(*[[ref[A_LABELED]], alpha_soft_uninf, ref[A_S_INFECTED], alpha_unlabeled, knn])
    params += get_configurations(*[[ref[A_LABELED]], alpha_soft_uninf, alpha_soft_inf, [ref[A_UNLABELED]], knn])
    np.set_printoptions(precision=3)

    num_labeled = 45

    for p in params:
        #if p[0] <= p[1] and p[1] < p[2] and p[2] <= p[3]:
        if True:
            for i in range(20):
                print "bsub -n 1 -q pub.8h -R \"rusage[mem=1024]\" './hcs.py -l all/labeled/gw*.arff -s all/soft/ -c \
-f 4 3 2 1 92 53 54 -L %i -n %i -v -al %f -asu %f -asi %f -au %f -nf knn%i -q'" % \
                    (num_labeled, 4 * num_labeled, p[0], p[1], p[2], p[3], p[4])


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python

import sys
import numpy as np
import argparse

RECALL, A_LABELED, A_UNLABELED, A_S_UNINFECTED, A_S_INFECTED = 0, 2, 4, 6, 8

def r(arr):
    a = arr
    reminder = np.array([], dtype=int)
    if len(arr) % 2 == 1:
        reminder = a[-1]
        a = a[:-1]
    a = np.vstack([np.flipud(a[:len(a) / 2]), a[len(a) / 2:]]).T.flatten()
    return np.hstack([a, reminder])


def process_cmdline(cmdline):
    parser = argparse.ArgumentParser(description="generate batch of scripts for brutus cluster")
    parser.add_argument("-n", "--num-labeled", help="Number of labeled points", required=True, type=int)
    parser.add_argument("-x", "--num-repetitions", help="Number of repetitions per configuration", type=int, default=20)
    parser.add_argument("-r", "--reference", help="Reference configuration (array)", type=float, default=None, nargs='+')
    params = vars(parser.parse_args())
    return params


def get_configurations(*list_ranges):
    grid = np.meshgrid(*list_ranges)
    grid_dimensions = [grid_dim.flatten() for grid_dim in grid]
    params = zip(*grid_dimensions)
    return params


def main(cmdline):
    params = process_cmdline(cmdline)
    ref_list = params['reference']
    num_repetitions = params['num_repetitions']
    num_labeled = params['num_labeled']
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
    ref = {}
    params = []
    if ref_list:
        ref = {A_LABELED: [ref_list[0]], A_UNLABELED: [ref_list[3]],
               A_S_UNINFECTED: [ref_list[1]], A_S_INFECTED: [ref_list[2]]}
        params += get_configurations(*[alpha_labeled, ref[A_S_UNINFECTED], ref[A_S_INFECTED], alpha_unlabeled, knn])
        params += get_configurations(*[alpha_labeled, ref[A_S_UNINFECTED], alpha_soft_inf, ref[A_UNLABELED], knn])
        params += get_configurations(*[alpha_labeled, alpha_soft_uninf, ref[A_S_INFECTED], ref[A_UNLABELED], knn])
        params += get_configurations(*[ref[A_LABELED], ref[A_S_UNINFECTED], alpha_soft_inf, alpha_unlabeled, knn])
        params += get_configurations(*[ref[A_LABELED], alpha_soft_uninf, ref[A_S_INFECTED], alpha_unlabeled, knn])
        params += get_configurations(*[ref[A_LABELED], alpha_soft_uninf, alpha_soft_inf, ref[A_UNLABELED], knn])
        np.set_printoptions(precision=3)
    else:
        ref = {A_LABELED: alpha_labeled, A_UNLABELED: alpha_unlabeled,
               A_S_UNINFECTED: alpha_soft_uninf, A_S_INFECTED: alpha_soft_inf}
        params += get_configurations(*[alpha_labeled, alpha_soft_uninf, alpha_soft_inf, alpha_unlabeled, knn])

    for p in params:
        #if p[0] <= p[1] and p[1] < p[2] and p[2] <= p[3]:
        if True:
            for i in range(num_repetitions):
                print "bsub -n 1 -q pub.8h -R \"rusage[mem=1024]\" './hcs.py -l all/labeled/gw*.arff -s all/soft/ -c \
-f 4 3 2 1 92 53 54 -L %i -n %i -v -al %f -asu %f -asi %f -au %f -nf knn%i -q'" % \
                    (num_labeled, 4 * num_labeled, p[0], p[1], p[2], p[3], p[4])


if __name__ == "__main__":
    main(sys.argv[1:])

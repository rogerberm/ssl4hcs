#!/usr/bin/env python

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import argparse
import csv
from itertools import groupby
from matplotlib import cm

RECALL, A_LABELED, A_UNLABELED, A_S_UNINFECTED, A_S_INFECTED = 0, 2, 4, 6, 8
columns = {A_LABELED: 'Alpha (labeled)',
           A_UNLABELED: 'Alpha (unlabeled)',
           A_S_UNINFECTED: 'Alpha (Soft-labeled, uninfected)',
           A_S_INFECTED: 'Alpha (Soft-labeled, infected)',
           RECALL: 'recall'}
aggregating_functions = {'median': np.median,
                         'mean': np.mean,
                         'max': np.max,
                         'min': np.min,
                         'sum': np.sum}


def process_cmdline(cmdline):
    parser = argparse.ArgumentParser(description="generate surface plot")
    parser.add_argument("-i", "--input-file", help="Input file (csv)", required=True)
    parser.add_argument("-a", "--aggregating-function", help="Aggregating function. Default: median",
                        choices=aggregating_functions.keys(), default='median')
    params = vars(parser.parse_args())
    return params


def get_csv_data(csv_file_path, use_headers=False):
    try:
        with open(csv_file_path, 'rb') as csv_file_pointer:
            csv_data = csv.reader(csv_file_pointer)
            if use_headers:
                csv_data.next()

            datalist = np.array([row for row in csv_data])
            return datalist
        return None
    except:
        return None


def plot_surface(figure, position, grid, x_label='X', y_label='Y', z_label='Z'):
    try:
        ax = figure.add_subplot(230 + position, projection='3d')
        #ax = figure.add_subplot(110 + position, projection='3d')
        X, Y = np.meshgrid(np.unique(grid[:, 0]), np.unique(grid[:, 1]))
        Z = grid[:, 2].reshape(X.shape[0], Y.shape[1], order='F')
        #ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0.5, rstride=1, cstride=1, vmin=0, vmax=1)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0.5, rstride=1, cstride=1)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
    except Exception as e:
        print "ERROR! %s " % e
        pass


def get_plot_data(csv_data, column_list, reference_point=None, aggregating_function=np.median):
    if reference_point:
        missing_keys = [key for key in reference_point.keys() if key not in column_list]
        row_filter = np.all(csv_data[:, missing_keys].astype('float') ==
                            [reference_point[key] for key in missing_keys], axis=1)
        #plot_data = csv_data[row_filter, column_list].astype('float').copy()
        plot_data = csv_data[row_filter][:, column_list].astype('float').copy()
    else:
        plot_data = csv_data[:, column_list].astype('float').copy()
    plot_data = \
        plot_data.view('float64, float64, float64').copy()
    plot_data = np.sort(plot_data, order=['f0', 'f1'], axis=0)

    #all_lists = []
    #for key, group in groupby(plot_data, lambda row:row[['f0', 'f1']]):
    #    all_lists += [list(list(group)[0][0])]
    #result = np.vstack(all_lists)
    #print result

    result = np.vstack([np.hstack([key.view('float'), aggregating_function([item['f2'] for item in group])]) for key, group in
                        groupby(plot_data, lambda row:row[['f0', 'f1']])])

    #result = np.vstack([np.array(list(group)).flatten() for _,group in groupby(plot_data, lambda row:row[['f0', 'f1']])])
    return result


def main(cmdline):
    params = process_cmdline(cmdline)
    input_file = params['input_file']
    aggregating_function = aggregating_functions[params['aggregating_function']]
    csv_data = get_csv_data(input_file)
    np.set_printoptions(edgeitems=55000, linewidth=180)

    fig = plt.figure()

    label_z = columns[RECALL]
    label_x = columns[A_LABELED]
    label_y = columns[A_UNLABELED]

    # best
    #reference_point = {A_LABELED: 0.0900,
    #                   A_UNLABELED: 0.9800,
    #                   A_S_UNINFECTED: 0.9000,
    #                   A_S_INFECTED: 0.5000}
    reference_point = {A_LABELED: 0.17, A_UNLABELED: 0.8350, A_S_UNINFECTED: 0.66, A_S_INFECTED: 0.82}
    reference_point = None
    #0.826000 a-labeled, 0.0900, a-unlabeled, 0.9800, a-soft-uninf, 0.9000, a-soft-inf, 0.5000, nf, knn7,
    #0.824000 a-labeled, 0.0900, a-unlabeled, 0.4725, a-soft-uninf, 0.9800, a-soft-inf, 0.9800, nf, knn7,
    #0.822000 a-labeled, 0.1700, a-unlabeled, 0.8350, a-soft-uninf, 0.9000, a-soft-inf, 0.8200, nf, knn7,
    #0.820000 a-labeled, 0.1300, a-unlabeled, 0.6900, a-soft-uninf, 0.3400, a-soft-inf, 0.8200, nf, knn7,
    #0.816000 a-labeled, 0.1700, a-unlabeled, 0.7625, a-soft-uninf, 0.9000, a-soft-inf, 0.9800, nf, knn7,
    #0.816000 a-labeled, 0.0900, a-unlabeled, 0.6175, a-soft-uninf, 0.6600, a-soft-inf, 0.5800, nf, knn7,

    a_lab_unlab = get_plot_data(csv_data, [A_LABELED, A_UNLABELED, RECALL], reference_point, aggregating_function)
    plot_surface(fig, 1, a_lab_unlab, label_x, label_y, label_z)

    a_lab_s_inf = get_plot_data(csv_data, [A_LABELED, A_S_INFECTED, RECALL], reference_point, aggregating_function)
    label_y = columns[A_S_INFECTED]
    plot_surface(fig, 2, a_lab_s_inf, label_x, label_y, label_z)

    a_lab_s_uninf = get_plot_data(csv_data, [A_LABELED, A_S_UNINFECTED, RECALL], reference_point, aggregating_function)
    label_y = columns[A_S_UNINFECTED]
    plot_surface(fig, 3, a_lab_s_uninf, label_x, label_y, label_z)

    label_x = columns[A_UNLABELED]
    a_unlab_s_inf = get_plot_data(csv_data, [A_UNLABELED, A_S_INFECTED, RECALL], reference_point, aggregating_function)
    label_y = columns[A_S_INFECTED]
    plot_surface(fig, 4, a_unlab_s_inf, label_x, label_y, label_z)

    label_x = columns[A_UNLABELED]
    a_unlab_s_uninf = get_plot_data(csv_data, [A_UNLABELED, A_S_UNINFECTED, RECALL], reference_point, aggregating_function)
    label_y = columns[A_S_UNINFECTED]
    plot_surface(fig, 5, a_unlab_s_uninf, label_x, label_y, label_z)

    label_x = columns[A_S_INFECTED]
    a_s_inf_s_uninf = get_plot_data(csv_data, [A_S_INFECTED, A_S_UNINFECTED, RECALL], reference_point, aggregating_function)
    label_y = columns[A_S_UNINFECTED]
    plot_surface(fig, 6, a_s_inf_s_uninf, label_x, label_y, label_z)

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python
'''
Semi-supervised learning for phenotypic profiling of high-content screens.
Label propagation algorithm on similarity graphs

Author: Roger Bermudez-Chacon
Supervisor: Peter Horvath

ETH Zurich, 2013.
'''
import sys
from math import exp
from random import seed
import numpy as np
from numpy import set_printoptions
from numpy.random import RandomState
import argparse
from scipy.spatial.distance import pdist, squareform, euclidean
from utils import parentdir, get_sample, generate_test_data, get_files, \
    hcs_labels, hcs_soft_labels, hcs_soft_label_alphas, dummy_soft_labels, dummy_soft_label_alphas
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

quiet = False


def ___(text, quiet=False):
    if not globals()['quiet']:
        print text


def read_hcs_file(file_path, feature_list, delimiter=None, column_offset=2, soft_label=-1):
    '''
    Reads a file with the tif.txt format (delimiter='  ', ignore first '  ' at the beginning of each line)

    Parameters
    ----------
    file_path     -- Path to a single file to be read
    feature_list  -- Array with the indices to select
    delimiter     -- Column separator character
    column_offset -- Number of missing heading columns in labeled data with respect to unlabeled txt files

    Returns
    -------
    numpy [len(features) + 1]-dimensional array containing the feature values (as float) and a default (-1) label
    for each data point in the file.
    '''
    feature_values = None
    with open(file_path, 'r') as txt_file_pointer:
        feature_values = [[float(text_fields[feature_index + column_offset - 1])
                           for feature_index in feature_list + [len(text_fields) - column_offset]]
                          for text_fields in [txt_line.strip().split(delimiter) + [soft_label]
                                              for txt_line in txt_file_pointer]]
    return np.array(feature_values)


def read_arff_file(file_path, feature_list, label_line_prefix='@attribute class ', data_header='@data',
                   delimiter=',', ignore_labels=['6'], label_column=94):
    '''
    Reads a file with the arff format (feature description lines, labels, data)

    Parameters
    ----------
    file_path         -- Path to a single file to be read
    feature_list      -- Array with the indices to select
    label_line_prefix -- Prefix that the line containing labels starts with
    data_header       -- Header line for the actual feature data
    delimiter         -- Column data separator character
    ignore_labels     -- Label list for which corresponding data should be ignored
    label_column      -- Number of column to read labels from

    Returns
    -------
    numpy [len(features) + 1]-dimensional array containing the feature values (as float) and the label
    for each data point in the file.
    '''
    feature_values = None
    feature_list_with_label = feature_list + [label_column]
    with open(file_path, 'r') as arff_file_pointer:
        arff_line = ''
        labels = None
        while not arff_line.startswith(label_line_prefix):
            arff_line = arff_file_pointer.readline().strip()
        if arff_line.startswith(label_line_prefix):
            label_text = arff_line[len(label_line_prefix) + 1:-1]
            labels = [label for label in label_text.split() if label not in ignore_labels]
        if labels:
            while arff_line != data_header:
                arff_line = arff_file_pointer.readline().strip()
            # for each line in the @data section that corresponds to data
            # labeled with valid (i.e. not ignored) labels, return all the data as a float matrix
            feature_values = [[float(text_fields[feature_index - 1]) for feature_index in feature_list_with_label]
                              for text_fields in [arff_data.strip().split(delimiter)
                              for arff_data in arff_file_pointer] if text_fields[-1] in labels]
        else:
            raise ValueError('label specification not found')
    return np.array(feature_values)


def process_cmdline(argv):
    '''
    Reads program parameters from the command line and sets default values for missing parameters
    '''
    # TODO: move this to utils
    parser = argparse.ArgumentParser(description="Label propagation")
    parser.add_argument("-t", "--test", help="Performs a test run.", action='store_true')
    parser.add_argument("-l", "--labeled", help="Labeled files.", dest="labeled_file_references", nargs='+',
                        metavar='LABELED_FILE')
    parser.add_argument("-u", "--unlabeled", help="Unlabeled files.", dest='unlabeled_file_references', nargs='+',
                        metavar='UNLABELED_FILE')
    parser.add_argument("-s", "--soft-labeled", help="Path to soft labeled files. One directory per label expected.",
                        dest='soft_labeled_path')
    parser.add_argument("-L", "--num-labeled", help="Number of labeled data points to use. Default: use all available",
                        type=int, dest='labeled_sample_size', default=-1, metavar='NUM_LABELED_POINTS')
    parser.add_argument("-n", "--num-samples", help="Number of samples. Default: 3000", type=int, dest='sample_size',
                        default=3000, metavar='NUM_SAMPLES')
    parser.add_argument("-c", "--class-sampling", help='Distributes the number of samples given by [NUM_SAMPLES]\
                        uniformly over all soft classes', dest='class_sampling', action='store_true')
    parser.add_argument("--max-iterations", help="Maximum number of iterations. Default: 1000", type=int,
                        dest='max_iterations', default=1000)
    parser.add_argument("-d", "--display-columns", help="Max width used for matrix display on console", type=int,
                        dest='width', default=310)
    parser.add_argument("-nf", "--neighborhood-function", help="Neighborhood function to use. Default: exp",
                        choices=['exp', 'knn3', 'knn4', 'knn5', 'knn6'], dest='neighborhood_fn', default='exp')
    parser.add_argument("-dm", "--distance-metric", help="Metric for calculating pairwise distances. Default: euclidean",
                        choices=['euclidean', 'cityblock', 'cosine', 'sqeuclidean', 'hamming', 'chebyshev'],
                        dest='distance_metric', default='euclidean')
    parser.add_argument("-f", "--features", help='Selected feature indices (as given by the labeled data).', nargs='+',
                        dest='feature_list', type=int, default=[1, 2], metavar='FEATURE_INDEX')
    parser.add_argument("-q", "--quiet", help="Displays progress and messages.", action='store_true')
    args = vars(parser.parse_args())
    args['soft_labeled_sample_size'] = args['sample_size'] / 2
    args['unlabeled_sample_size'] = args['sample_size'] / 2
    if args['test']:
        args['unlabeled_file_references'] = ["../dummydata/unlabeled/bDummy/bDummy1.txt",
                                             "../dummydata/unlabeled/bDummy/bDummy2.txt",
                                             "../dummydata/unlabeled/bDummy/bDummy3.txt"]
        args['soft_labeled_path'] = "../dummydata/soft/"
        args['labeled_file_references'] = ["../dummydata/labeled/labeled.arff"]
    #print args

    #TODO: pass args namespace object
    unlabeled_file_references = args['unlabeled_file_references']
    soft_labeled_path = args['soft_labeled_path']
    labeled_file_references = args['labeled_file_references']
    feature_list = args['feature_list']
    soft_labeled_sample_size = args['soft_labeled_sample_size']
    unlabeled_sample_size = args['unlabeled_sample_size']
    labeled_sample_size = args['labeled_sample_size']
    class_sampling = args['class_sampling']
    distance_metric = args['distance_metric']
    neighborhood_fn = args['neighborhood_fn']
    alpha = 0.94
    max_iterations = args['max_iterations']
    set_printoptions(linewidth=args['width'] - 1, nanstr='0', precision=3)

    test = args['test']
    return (unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list,
            soft_labeled_sample_size, unlabeled_sample_size, labeled_sample_size, class_sampling, distance_metric,
            neighborhood_fn, alpha, max_iterations, test)


def generate_color_map(label_array, num_labeled, num_soft_labeled, unlabeled_initial_color=None):
    color_map = np.concatenate([
        np.where(label_array[0, 0:num_labeled] > 0, 1.0,  0.0),
        np.where(label_array[0, num_labeled:num_labeled + num_soft_labeled] > 0, 0.9,  0.1),
        np.where(label_array[0, num_labeled + num_soft_labeled:] > 0, unlabeled_initial_color or 0.75,  unlabeled_initial_color or 0.25)])
    return color_map


def propagate_labels_SSL(feature_matrix, initial_labels, distance_metric, neighborhood_fn, alpha_vector,
                         max_iterations, num_labeled_points, num_soft_labeled_points, labels=hcs_labels):
    '''
    Implementation of the label spreading algorithm (Zhou et al., 2004) on a graph, represented by
    its similarity matrix.
    Parameters:
    [TODO: add parameter information]
    '''
    ___("calculating pairwise distances for %i datapoints, %i dimensions each..." %
        (len(feature_matrix), len(feature_matrix[0])))
    pairwise = pdist(feature_matrix)
    ___("  pairwise distances: %s... (%i distances)" % (str(pairwise[1:5]).replace(']', ''), len(pairwise)))
    ___("getting the square form...",)
    pairwise_matrix = squareform(pairwise)
    ___("%ix%i pairwise distances matrix created" % (len(pairwise_matrix), len(pairwise_matrix)))

    # Create weight matrix W:
    if neighborhood_fn == 'exp':
        exp_matrix = np.vectorize(lambda x: exp(-x))
        W = squareform(exp_matrix(pairwise))
    elif neighborhood_fn.startswith('knn'):
        k = int(neighborhood_fn[len('knn'):])
        print "Using %i nearest neighbors" % k
        knns = np.argsort(pairwise_matrix, axis=1)[:, :k + 1]
        W = np.array([[1 if j in knns[i] else 0 for j in range(pairwise_matrix.shape[0])] for i in range(pairwise_matrix.shape[0])])
    np.fill_diagonal(W, 0)

    D_sqrt_inv = np.matrix(np.diag(1.0 / np.sqrt(np.sum(W, axis=1))))  # D^{-1/2}
    Alpha = np.matrix(np.diag(alpha_vector))
    OneMinusAlpha = np.matrix(np.diag(1 - np.array(alpha_vector)))
    Y_0 = np.matrix(initial_labels).T
    Y_t = np.matrix(initial_labels).T
    ___("Calculating Laplacian..."),
    Laplacian = D_sqrt_inv * W * D_sqrt_inv
    ___("Iterating up to %i times..." % max_iterations)

    # setup plot
    fig = plt.figure()
    splot_orig = fig.add_subplot(121)
    splot = fig.add_subplot(122)
    labeled_array = generate_color_map(np.array(Y_t.T), num_labeled_points, num_soft_labeled_points, 0.5)
    scat_orig = splot_orig.scatter(feature_matrix[:, 0], feature_matrix[:, 1], s=70, c=labeled_array, cmap=cm.RdBu)
    splot_orig.set_xlabel("feature 1")
    splot_orig.set_ylabel("feature 2")
    splot_orig.set_title("Original label assignment")
    splot.set_xlabel("feature 1")
    splot.set_ylabel("feature 2")
    splot.set_title("Current label assignment")
    scat = splot.scatter(feature_matrix[:, 0], feature_matrix[:, 1], s=70, c=labeled_array, cmap=cm.RdBu)
    header = plt.figtext(0.5, 0.96, "Label propagation", weight="bold", size="large", ha="center")
    ani = None

    def init_scatter():
        scat.set_array(labeled_array)

    # Animation frame
    def get_propagated_labels(t):
        if t == 0:
            raw_input()
        try:
            Y_new = Laplacian * Alpha * Y_t + OneMinusAlpha * Y_0
            if (euclidean(Y_t, Y_new) < 1e-6 or t >= max_iterations) and ani:
                ani._stop()
            np.copyto(Y_t, Y_new)
            color_map = generate_color_map(np.array(Y_t.T), num_labeled_points, num_soft_labeled_points)
            if t == 0:  # display first label assignment
                scat_orig.set_array(color_map)
            scat.set_array(color_map)
            header.set_text("Label propagation. Iteration %i" % t)
        except KeyboardInterrupt:
            return

    ani = animation.FuncAnimation(fig, get_propagated_labels, init_func=init_scatter, interval=600)
    plt.show()
    return None


def normalize(M):
    return (M - np.mean(M, 0)) / np.std(M, 0)


def setup_feature_matrix(unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list,
                         soft_labeled_sample_size, unlabeled_sample_size, labeled_sample_size, class_sampling,
                         alpha_labeled=0.95, alpha_unlabeled=0.95, alpha_soft_labeled=hcs_soft_label_alphas,
                         class_labels=hcs_soft_labels, ignore_labels=[6], normalize_data=True):
    '''
    Reads files with unlabeled and labeled information, and creates the feature matrix with the features
    specified in the feature list

    Parameters
    ----------
    unlabeled_file_references -- paths to txt files with raw unlabeled data
    labeled_file_references   -- paths to arff files with labeled data
    soft_labeled_path         -- paths txt soft-labeled files. This path should contain a directory per label
    feature_list              -- list with the indices of the selected features (1 based)
    sample_size               -- size of sampling over unlabeled data (-1 means use all data)
    class_sampling            -- if sampling is required, sample the same number of points per class

    Returns
    -------
    Matrix with features as columns, and individuals (cells) as rows.
    The matrix contains an additional column for the labeling, if present, or -1 otherwise.
    '''

    # [TODO] parametrize this:
    #alpha_labeled, alpha_unlabeled = 0.95, 0.95
    #alpha_soft_labeled = {1: 0.95, 5: 0.95}
    alpha_vector = []

    labeled_data = [read_arff_file(input_file, feature_list, ignore_labels=ignore_labels)
                    for input_file in labeled_file_references]
    labeled_points = np.concatenate(labeled_data)
    labeled_points = get_sample(labeled_points, labeled_sample_size)  # ???
    ___("%i labeled points:\n\t%s" % (len(labeled_points), labeled_points[:, -1]))

    alpha_vector += [alpha_labeled] * len(labeled_points)

    soft_labeled_points = None
    if soft_labeled_path:
        soft_labeled_data = {label_key: [] for label_key in class_labels.keys()}
        soft_labeled_file_references = get_files(soft_labeled_path)
        for input_file in soft_labeled_file_references:
            label_key = parentdir(input_file)
            soft_labeled_data[label_key] += [read_hcs_file(input_file, feature_list, soft_label=class_labels[label_key])]
        # Sample unlabeled data uniformly over classes
        if class_sampling:
            class_sample_size = {class_label: soft_labeled_sample_size / len(class_labels) if class_sampling else -1
                                 for class_label in class_labels.keys()}
            soft_labeled_data = [get_sample(np.concatenate(soft_labeled_set), class_sample_size[label_key])
                                 for soft_labeled_set_name, soft_labeled_set in soft_labeled_data.iteritems()]
        else:
            soft_labeled_data = soft_labeled_data.values()
        soft_labeled_points = np.concatenate(soft_labeled_data)
        soft_labeled_alphas = [alpha_soft_labeled[label] for label in soft_labeled_points[:, -1]]
        ___("%i soft labeled points:\n\t%s" % (len(soft_labeled_points), soft_labeled_points[:, -1]))
        alpha_vector += soft_labeled_alphas

    unlabeled_data = [read_hcs_file(input_file, feature_list) for input_file in unlabeled_file_references]
    unlabeled_points = np.concatenate(unlabeled_data)

    unlabeled_points = get_sample(unlabeled_points, unlabeled_sample_size)  # ???
    ___("%i unlabeled points:\n\t%s" % (len(unlabeled_points), unlabeled_points[:, -1]))

    alpha_vector += [alpha_unlabeled] * len(unlabeled_points)

    if soft_labeled_points is None:
        M = np.concatenate([labeled_points, unlabeled_points])
    else:
        M = np.concatenate([labeled_points, soft_labeled_points, unlabeled_points])

    if normalize_data:
        M = np.column_stack([normalize(M[:, :-1]), M[:, -1]])
    return (M, alpha_vector, len(labeled_points), len(soft_labeled_points))


def create_dummy_data1():
    dummy_matrix = np.array(
        [[0.80, 1.30, -1],
         [0.78, 1.40, -1],
         [4.60, 5.00, +1],
         [5.10, 4.90, +1],
         [0.00, 1.00, -1],
         [1.00, 0.00, -1],
         [0.50, 0.50, -1],
         [4.00, 4.00, +1],
         [3.00, 3.00, +1],
         [2.60, 2.80, +1],
         [4.80, 5.10,  0],  # 1
         [0.50, 1.20,  0],
         [1.10, 1.10,  0],
         [4.30, 6.20,  0],  # 1
         [0.00, 1.20,  0],
         [4.90, 4.60,  0],  # 1
         [2.48, 2.80,  0],  # ?
         [2.55, 2.80,  0]]  # ?
    )
    alpha_vector = np.array([0.95] * 4 + [0.95] * 6 + [0.95] * 8)
    return dummy_matrix, alpha_vector


def get_uniform_sample(center, width, num_points):
    randvals = np.random.rand(num_points, 2)
    points = center + width * np.column_stack([randvals[:, 0] * np.sin(2 * np.pi * randvals[:, 1]), randvals[:, 0] * np.cos(2 * np.pi * randvals[:, 1])])
    return points


def create_dummy_data(labeled_points, soft_labeled_points, unlabeled_points, normalize_data=True):
    center_labeled_inf, width_labeled_inf = [1.5, 1.2], 0.75
    #center_labeled_inf, width_labeled_inf = [2.5, 2], 0.15
    center_labeled_noninf, width_labeled_noninf = 4.5, 0.9
    center_soft_inf, width_soft_inf = [1.4, 1.6], 1.4
    center_soft_noninf, width_soft_noninf = 4.6, 1.2
    #center_unlabeled1, width_unlabeled1 = 3.0, 1.0
    #center_unlabeled2, width_unlabeled2 = 2.9, 0.1
    center_unlabeled1, width_unlabeled1 = 3.0, 3.0  # 75%
    center_unlabeled2, width_unlabeled2 = 3.0, 3.0  # 25%

    M = np.column_stack((get_uniform_sample(center_labeled_inf, width_labeled_inf, labeled_points / 2), [1.0] * (labeled_points / 2)))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_labeled_noninf, width_labeled_noninf, labeled_points / 2), [-1.0] * (labeled_points / 2)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft_inf, width_soft_inf, soft_labeled_points / 2), [1.0] * (soft_labeled_points / 2)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft_noninf, width_soft_inf, 3), [1.0] * 3))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft_noninf, width_soft_noninf, soft_labeled_points / 2 - 3), [-1.0] * (soft_labeled_points / 2 - 3)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_unlabeled1, width_unlabeled1, int(0.75 * unlabeled_points)), [0.0] * int(0.75 * unlabeled_points)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_unlabeled2, width_unlabeled2, unlabeled_points - int(0.75 * unlabeled_points)), [0.0] * (unlabeled_points - int(0.75 * unlabeled_points))))))

    alpha_vector = np.array([0.5] * labeled_points + [0.85] * soft_labeled_points + [0.95] * unlabeled_points)
    if normalize_data:
        M = np.column_stack([normalize(M[:, :-1]), M[:, -1]])
    return M, alpha_vector, labeled_points, soft_labeled_points


def main(argv):
    '''
    Reads the parameters from the command line, loads the referred labeled and unlabeled files, and
    applies the label propagation algorithm iteratively
    '''
    unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list, soft_labeled_sample_size, \
        unlabeled_sample_size, labeled_sample_size, class_sampling, distance_metric, neighborhood_fn, alpha, \
        max_iterations, test = process_cmdline(argv)

    RandomState(7283)
    seed(7283)
    if test:
        M, alpha_vector, labeled_points, soft_labeled_points = create_dummy_data(40, 180, 180, normalize_data=False)
    else:
        M, alpha_vector, labeled_points, soft_labeled_points = \
            setup_feature_matrix(unlabeled_file_references, soft_labeled_path, labeled_file_references,
                                 feature_list, soft_labeled_sample_size, unlabeled_sample_size,
                                 labeled_sample_size, class_sampling, ignore_labels=['6'])

    Y = propagate_labels_SSL(M[:, :-1], M[:, -1], distance_metric, neighborhood_fn, alpha_vector, max_iterations, labeled_points, soft_labeled_points)
    return Y


def load_test():
    '''
    Creates and uses dummy data to test the label propagation algorithm
    '''
    unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list, soft_labeled_sample_size, \
        unlabeled_sample_size, labeled_sample_size, class_sampling, distance_metric, neighborhood_fn, alpha, \
        max_iterations, test = generate_test_data()

    M, alpha_vector, labeled_points, soft_labeled_points = \
        setup_feature_matrix(unlabeled_file_references, soft_labeled_path, labeled_file_references,
                             feature_list, soft_labeled_sample_size, unlabeled_sample_size,
                             labeled_sample_size, class_sampling, alpha_labeled=0.95,
                             alpha_unlabeled=0.95, alpha_soft_labeled=dummy_soft_label_alphas,
                             class_labels=dummy_soft_labels, ignore_labels=['2', '3', '4', '6'])

    label_value_translation = {-1: 0, 1: -1, 5: 1}
    initial_labels = [label_value_translation[label] for label in M[:, -1]]
    Y = propagate_labels_SSL(M[:, :-1], initial_labels, distance_metric, neighborhood_fn, alpha_vector,
                             max_iterations, labeled_points, soft_labeled_points)
    return Y


if __name__ == "__main__":
    main(sys.argv[1:])
else:
    load_test()

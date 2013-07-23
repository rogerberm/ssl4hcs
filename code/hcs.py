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
from glob import glob
import numpy as np
#from numpy import array as np.array, concatenate, vectorize, where, diag
#from numpy.ma import masked_array
from getopt import getopt, GetoptError
from scipy.spatial.distance import pdist, squareform
from utils import print_help, assert_argument, parentdir, get_sample, generate_test_data, get_files, \
    hcs_soft_labels, dummy_soft_labels, MESSAGES

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
    try:
        opts, args = getopt(argv, "hu:l:f:n:s:cqt", ["unlabeled=", "labeled=", "features="])
    except GetoptError as err:
        print(str(err))
        sys.exit(2)
    unlabeled_file_references, soft_labeled_path, labeled_file_references = None, None, None
    # Defaults:
    feature_list = [1, 2]
    sample_size = 3000
    soft_labeled_sample_size = sample_size / 2
    unlabeled_sample_size = sample_size / 2
    class_sampling = True
    distance_metric = 'euclidean'
    neighborhood_fn = 'exp'
    alpha = 0.94
    max_iterations = 1000

    for opt, arg in opts:
        if opt in ('-u', '--unlabeled'):
            unlabeled_file_references = glob(arg)
        if opt in ('-t', '--test'):
            load_test()
            sys.exit()
        elif opt in ('-l', '--labeled'):
            labeled_file_references = glob(arg)
        elif opt in ('-s', '--soft-labeled-path'):
            soft_labeled_arg = glob(arg)
            if len(soft_labeled_arg) > 0:
                soft_labeled_path = glob(arg)[0]
            else:
                print MESSAGES['INVALID_SOFT_LABELED_FILE_PATH']
                sys.exit(2)
        elif opt in ('-h', '--help'):
            print_help()
            sys.exit()
        elif opt in ('-q', '--quiet'):
            globals()['quiet'] = True
        elif opt in ('-n', '--num-samples'):
            sample_size = int(arg)
            soft_labeled_sample_size = sample_size / 2
            unlabeled_sample_size = sample_size / 2
        elif opt == '-c':
            class_sampling = True
        elif opt in ('-f', '--features'):
            try:
                feature_list = eval(arg)
                if not isinstance(feature_list, list):
                    raise NameError('Argument is not a list')
            except NameError:
                print MESSAGES['INVALID_FEATURE_LIST_MESSAGE']
                sys.exit(2)
            except SyntaxError:
                print MESSAGES['INVALID_FEATURE_LIST_MESSAGE']
                sys.exit(2)
            pass
        else:
            assert False, "unhandled option"
    assert_argument(unlabeled_file_references, MESSAGES['NO_UNLABELED_FILE_MESSAGE'])
    ___("loading %i unlabeled datasets..." % len(unlabeled_file_references))
    assert_argument(labeled_file_references, MESSAGES['NO_LABELED_FILE_MESSAGE'])
    ___("loading %i labeled datasets..." % len(labeled_file_references))
    return (unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list,
            soft_labeled_sample_size, unlabeled_sample_size, class_sampling, distance_metric,
            neighborhood_fn, alpha, max_iterations)


def propagate_labels_SSL(feature_matrix, initial_labels, distance_metric, neighborhood_fn, alpha_vector, max_iterations):
    '''
    Implementation of the label spreading algorithm (Zhou et al., 2004) on a graph, represented by
    its similarity matrix.
    Parameters:
    [TODO: add parameter information]
    '''
    ___("calculating pairwise distances for %i datapoints, %i dimensions each..." %
        (len(feature_matrix), len(feature_matrix[0])))
    pairwise = pdist(feature_matrix[:, :-1])
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
        #W = masked_array(pairwise_matrix, mask=pairwise_matrix.argsort(axis=1) >= k)
        W = np.where(pairwise_matrix.argsort() >= k, -1, pairwise_matrix)
        print W

        from numpy import set_printoptions, array2string
        from os import environ
        set_printoptions(linewidth=environ.get("COLUMNS") or 230, nanstr='0', precision=4)
        print(array2string(pairwise_matrix.argsort(axis=1)))
        print(array2string(W))
        print(array2string(pairwise_matrix))

    np.fill_diagonal(W, 0)
    D_sqrt_inv = np.diag(1.0 / np.sum(W, axis=1))  # D^{-1/2}
    Alpha = np.diag(alpha_vector)
    OneMinusAlpha = np.diag(1 - np.array(alpha_vector))
    Y_0, Y = np.array(initial_labels), np.array(initial_labels)
    # print Alpha, OneMinusAlpha
    Laplacian = D_sqrt_inv.dot(W).dot(D_sqrt_inv)
    ___("Iterating %i times..." % max_iterations)
    from numpy import set_printoptions, array2string
    from os import environ
    set_printoptions(linewidth=environ.get("COLUMNS") or 230, nanstr='0', precision=4)
    print Y, Laplacian
    alpha = 0.95
    for i in range(max_iterations):
        print "\r>>iteration: %i" % i,
        try:
            #Y_new = Laplacian.dot(Alpha).dot(Y) + OneMinusAlpha.dot(Y_0)
            Y_new = alpha * Laplacian.dot(Y) + (1 - alpha) * Y_0
            if all(np.equal(Y, Y_new)):
                break
            Y = Y_new
        except KeyboardInterrupt:
            break
        print Y[-499: -1] ,
        sys.stdout.flush()
        raw_input()
        # useful functions: numpy.put, numpy.putmask
    return None


def setup_feature_matrix(unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list,
                         soft_labeled_sample_size, unlabeled_sample_size, class_sampling, class_labels=hcs_soft_labels, ignore_labels=[6]):
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
    #alpha_labeled, alpha_unlabeled = 0.5, 0.95
    #alpha_soft_labeled = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}
    alpha_labeled, alpha_unlabeled = 0.95, 0.95
    alpha_soft_labeled = {1: 0.95, 5: 0.95}
    alpha_vector = []

    labeled_data = [read_arff_file(input_file, feature_list, ignore_labels=ignore_labels) for input_file in labeled_file_references]
    labeled_points = np.concatenate(labeled_data)
    labeled_points = get_sample(labeled_points, 8)  # ???
    ___("%i labeled points" % len(labeled_points))

    alpha_vector += [alpha_labeled] * len(labeled_points)

    # [TODO]: remove next line when not testing

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
        ___("%i soft labeled points" % len(soft_labeled_points))
        alpha_vector += soft_labeled_alphas

    unlabeled_data = [read_hcs_file(input_file, feature_list) for input_file in unlabeled_file_references]
    unlabeled_points = np.concatenate(unlabeled_data)

    unlabeled_points = get_sample(unlabeled_points, unlabeled_sample_size)  # ???
    ___("%i unlabeled points" % len(unlabeled_points))

    alpha_vector += [alpha_unlabeled] * len(unlabeled_points)

    if soft_labeled_points is None:
        M = np.concatenate([labeled_points, unlabeled_points])
    else:
        M = np.concatenate([labeled_points, soft_labeled_points, unlabeled_points])
    return (M, alpha_vector)


def main(argv):
    '''
    Reads the parameters from the command line, loads the referred labeled and unlabeled files, and
    applies the label propagation algorithm iteratively
    '''
    unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list, soft_labeled_sample_size, \
        unlabeled_sample_size, class_sampling, distance_metric, neighborhood_fn, alpha, max_iterations \
        = process_cmdline(argv)

    M, alpha_vector = setup_feature_matrix(unlabeled_file_references, soft_labeled_path, labeled_file_references,
                                           feature_list, soft_labeled_sample_size, unlabeled_sample_size, class_sampling)

    Y = propagate_labels_SSL(M[:, :-1], M[:, -1], distance_metric, neighborhood_fn, alpha_vector, max_iterations)
    # ___("Alpha vector: %s" % alpha_vector)
    return Y


def load_test():
    '''
    Creates and uses dummy data to test the label propagation algorithm
    '''
    unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list, soft_labeled_sample_size, \
        unlabeled_sample_size, class_sampling, distance_metric, neighborhood_fn, alpha, max_iterations \
        = generate_test_data()

    M, alpha_vector = setup_feature_matrix(unlabeled_file_references, soft_labeled_path, labeled_file_references,
                                           feature_list, soft_labeled_sample_size, unlabeled_sample_size,
                                           class_sampling, dummy_soft_labels, ignore_labels=['2', '3', '4', '6'])

    label_value_translation = {-1: 0, 1: -1, 5: 1}
    initial_labels = [label_value_translation[label] for label in M[:, -1]]
    print initial_labels
    Y = propagate_labels_SSL(M[:, :-1], initial_labels, distance_metric, neighborhood_fn, alpha_vector, max_iterations)
    #return Y


if __name__ == "__main__":
    main(sys.argv[1:])
else:
    load_test()

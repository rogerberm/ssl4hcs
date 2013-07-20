#!/usr/bin/env python
'''
Semi-supervised learning for phenotypic profiling of high-content screens.
Label propagation algorithm on similarity graphs

Author: Roger Bermudez-Chacon
Supervisor: Peter Horvath

ETH Zurich, 2013.
'''
import sys
from glob import glob
from numpy import array, concatenate
from getopt import getopt, GetoptError
from scipy.spatial.distance import pdist, squareform
from utils import print_help, assert_argument, parentdir, \
    get_sample, generate_test_data, hcs_class_labels, \
    dummy_class_labels, MESSAGES

quiet = False


def ___(text, quiet=False):
    if not globals()['quiet']:
        print text


def read_hcs_file(file_path, feature_list, delimiter=None, column_offset=2):
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
    2yynumpy [len(features) + 1]-dimensional array containing the feature values (as float) and a default (-1) label
    for each data point in the file.
    '''
    feature_values = None
    with open(file_path, 'r') as txt_file_pointer:
        feature_values = [[float(text_fields[feature_index + column_offset - 1])
                           for feature_index in feature_list + [len(text_fields) - column_offset]]
                          for text_fields in [txt_line.strip().split(delimiter) + ['-1']
                                              for txt_line in txt_file_pointer]]
    return array(feature_values)


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
    return array(feature_values)


def process_cmdline(argv):
    '''
    Reads program parameters from the command line and sets default values for missing parameters
    '''
    try:
        opts, args = getopt(argv, "hu:l:f:s:cq", ["unlabeled=", "labeled=", "features="])
    except GetoptError as err:
        print(str(err))
        sys.exit(2)
    unlabeled_file_references, labeled_file_references = None, None
    # Defaults:
    feature_list = [1, 2]
    sample_size = 3000
    class_sampling = True
    distance_metric = 'euclidean'
    neighborhood_fn = 'exp'
    alpha = 0.94
    max_iterations = 1000

    for opt, arg in opts:
        if opt in ('-u', '--unlabeled'):
            unlabeled_file_references = glob(arg)
        elif opt in ('-l', '--labeled'):
            labeled_file_references = glob(arg)
        elif opt in ('-h', '--help'):
            print_help()
            sys.exit()
        elif opt in ('-q', '--quiet'):
            globals()['quiet'] = True
        elif opt in ('-s', '--sample-size'):
            sample_size = int(arg)
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
    return (unlabeled_file_references, labeled_file_references, feature_list, sample_size,
            class_sampling, distance_metric, neighborhood_fn, alpha, max_iterations)


def propagate_labels_SSL(feature_matrix, initial_labels, distance_metric, neighborhood_fn, alpha, max_iterations):
    '''
    Implementation of the label spreading algorithm (Zhou XXXX?) on a graph, represented by its similarity matrix
    '''
    ___("calculating pairwise distances for %i datapoints, %i dimensions each..." %
        (len(feature_matrix), len(feature_matrix[0])))
    pairwise = pdist(feature_matrix[:, :-1])
    ___("  pairwise distances: %s... (%i distances)" % (str(pairwise[1:5]).replace(']', ''), len(pairwise)))
    ___("getting the square form...",)
    pairwise_matrix = squareform(pairwise)
    ___("%ix%i pairwise distances matrix created" % (len(pairwise_matrix), len(pairwise_matrix)))
    return None


def setup_feature_matrix(unlabeled_file_references, labeled_file_references, feature_list,
                         sample_size, class_sampling, class_labels=hcs_class_labels):
    '''
    Reads files with unlabeled and labeled information, and creates the feature matrix with the features
    specified in the feature list

    Parameters
    ----------
    unlabeled_file_references -- paths to txt files with raw unlabeled data
    labeled_file_references   -- paths to arff files with labeled data
    feature_list              -- list with the indices of the selected features (1 based)
    sample_size               -- size of sampling over unlabeled data (-1 means use all data)
    class_sampling            -- if sampling is required, sample the same number of points per class

    Returns
    -------
    Matrix with features as columns, and individuals (cells) as rows.
    The matrix contains an additional column for the labeling, if present, or -1 otherwise.
    '''

    labeled_data = [read_arff_file(input_file, feature_list) for input_file in labeled_file_references]
    labeled_points = concatenate(labeled_data)

    unlabeled_points = None
    if class_sampling:
        # Sample unlabeled data uniformly over classes
        # (classes hardcoded, txt files into directories with these names assumed)
        class_sample_size = {class_label: sample_size / len(class_labels) for class_label in class_labels.keys()[:-1]}
        class_sample_size[class_labels.keys()[-1]] = sample_size - sum(class_sample_size.values())
        semilabeled_data = {prefix: [] for prefix in class_labels.keys()}
        for input_file in unlabeled_file_references:
            semilabeled_data[parentdir(input_file)] += [read_hcs_file(input_file, feature_list)]
        unlabeled_data = [get_sample(concatenate(unlabeled_set), class_sample_size[unlabeled_set_name])
                          for unlabeled_set_name, unlabeled_set in semilabeled_data.iteritems()]
        unlabeled_points = concatenate(unlabeled_data)
    else:
        unlabeled_data = [read_hcs_file(input_file, feature_list) for input_file in unlabeled_file_references]
        unlabeled_points = concatenate(unlabeled_data)
        unlabeled_points = get_sample(unlabeled_points, sample_size)

    M = concatenate([unlabeled_points, labeled_points])
    return M


def main(argv):
    '''
    Reads the parameters from the command line, loads the referred labeled and unlabeled files, and
    applies the label propagation algorithm iteratively
    '''
    unlabeled_file_references, labeled_file_references, feature_list, sample_size, class_sampling, \
        distance_metric, neighborhood_fn, alpha, max_iterations = process_cmdline(argv)

    M = setup_feature_matrix(unlabeled_file_references, labeled_file_references, feature_list,
                             sample_size, class_sampling)

    Y = propagate_labels_SSL(M[:, :-1], M[:, -1], distance_metric, neighborhood_fn, alpha, max_iterations)
    return Y


def load_test():
    '''
    Creates and uses dummy data to test the label propagation algorithm
    '''
    unlabeled_file_references, labeled_file_references, feature_list, sample_size, class_sampling, \
        distance_metric, neighborhood_fn, alpha, max_iterations = generate_test_data()

    M = setup_feature_matrix(unlabeled_file_references, labeled_file_references, feature_list,
                             sample_size, class_sampling, dummy_class_labels)

    Y = propagate_labels_SSL(M[:, :-1], M[:, -1], distance_metric, neighborhood_fn, alpha, max_iterations)
    return Y


if __name__ == "__main__":
    main(sys.argv[1:])
else:
    load_test()

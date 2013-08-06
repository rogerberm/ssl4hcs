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
import numpy as np
from numpy import set_printoptions
# timeit
import argparse
from random import seed
from scipy.spatial.distance import pdist, squareform, euclidean
from utils import parentdir, get_sample, generate_test_data, get_files, hcs_labels, hcs_soft_labels, \
    hcs_soft_label_alphas, dummy_labels2, dummy_soft_labels, dummy_soft_label_alphas
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib import widgets

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
    global quiet
    '''
    Reads program parameters from the command line and sets default values for missing parameters
    '''
    # TODO: move this to utils
    parser = argparse.ArgumentParser(description="Label propagation")
    parser.add_argument("-t", "--test", help="Performs a test run.", action='store_true')
    parser.add_argument("-v", "--validation", help="Performs a validation run.", action='store_true')
    parser.add_argument("-l", "--labeled", help="Labeled files.", dest="labeled_file_references", nargs='+',
                        metavar='LABELED_FILE')
    parser.add_argument("-u", "--unlabeled", help="Unlabeled files.", dest='unlabeled_file_references', nargs='+',
                        metavar='UNLABELED_FILE')
    parser.add_argument("-s", "--soft-labeled", help="Path to soft labeled files. One directory per label expected.",
                        dest='soft_labeled_path')
    parser.add_argument("-L", "--num-labeled", help="Number of labeled data points to use. Default: all available",
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
                        choices=['exp', 'knn3', 'knn4', 'knn5', 'knn6', 'knn7', 'knn8'], dest='neighborhood_fn', default='exp')
    parser.add_argument("-dm", "--distance-metric", help="Metric for pairwise distances. Default: euclidean",
                        choices=['euclidean', 'cityblock', 'cosine', 'sqeuclidean', 'hamming', 'chebyshev'],
                        dest='distance_metric', default='euclidean')
    parser.add_argument("-f", "--features", help='Selected feature indices (as given by the labeled data)', nargs='+',
                        dest='feature_list', type=int, default=[1, 2], metavar='FEATURE_INDEX')
    parser.add_argument("-q", "--quiet", help="Displays progress and messages.", action='store_true')
    parser.add_argument("-g", "--use-gui", help="Displays the graphical user interface.", action='store_true')
    parser.add_argument("-al", "--alpha-labeled", help="Learning rate for labeled data",
                        type=float, dest='alpha_labeled', default=0.1, metavar='ALPHA')
    parser.add_argument("-au", "--alpha-unlabeled", help="Learning rate for unlabeled data",
                        type=float, dest='alpha_unlabeled', default=0.8, metavar='ALPHA')
    parser.add_argument("-asu", "--alpha-soft-uninfected", help="Learning rate for soft data (uninfected)",
                        type=float, dest='alpha_soft_uninfected', default=0.3, metavar='ALPHA')
    parser.add_argument("-asi", "--alpha-soft-infected", help="Learning rate for soft data (infected)",
                        type=float, dest='alpha_soft_infected', default=0.5, metavar='ALPHA')
    args = vars(parser.parse_args())
    args['soft_labeled_sample_size'] = args['sample_size'] / 2
    args['unlabeled_sample_size'] = args['sample_size'] / 2
    if args['test']:
        args['unlabeled_file_references'] = ["../dummydata/unlabeled/bDummy/bDummy1.txt",
                                             "../dummydata/unlabeled/bDummy/bDummy2.txt",
                                             "../dummydata/unlabeled/bDummy/bDummy3.txt"]
        args['soft_labeled_path'] = "../dummydata/soft/"
        args['labeled_file_references'] = ["../dummydata/labeled/labeled.arff"]

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
    quiet = args['quiet']
    use_gui = args['use_gui']
    alpha_labeled = args['alpha_labeled']
    alpha_unlabeled = args['alpha_unlabeled']
    alpha_soft_uninfected = args['alpha_soft_uninfected']
    alpha_soft_infected = args['alpha_soft_infected']
    max_iterations = args['max_iterations']
    set_printoptions(linewidth=args['width'] - 1, nanstr='0', precision=3)

    test = args['test']
    validation = args['validation']
    return (unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list,
            soft_labeled_sample_size, unlabeled_sample_size, labeled_sample_size, class_sampling, distance_metric,
            neighborhood_fn, alpha_labeled, alpha_unlabeled, alpha_soft_uninfected, alpha_soft_infected,
            max_iterations, test, validation, use_gui)


def get_labels(label_array, labels):
    ordering = np.argmax(label_array, axis=0)
    return ordering + 1  # np.array([labels[o + 1] for o in ordering])


def generate_color_map(label_array, num_labeled, num_soft_labeled, unlabeled_initial_color=None):
    color_specification = np.array([[0.2, 0.17, 0.14], [0.4, 0.37, 0.34], [0.6, 0.57, 0.54], [0.9, 0.87, 0.84]])
    colors = np.linspace(0.3, 1, label_array.shape[0] + 1, endpoint=False)[1:]
    color_specification = np.vstack([colors, colors - 0.01, colors - 0.02]).T
    ordering = np.argmax(label_array, axis=0)
    color_map = np.concatenate([
        color_specification[ordering[:num_labeled]][:, 0],
        color_specification[ordering[num_labeled:num_labeled + num_soft_labeled]][:, 1],
        [unlabeled_initial_color] * (label_array.shape[1] - num_labeled - num_soft_labeled)
        if unlabeled_initial_color else color_specification[ordering[num_labeled + num_soft_labeled:]][:, 2]
    ])
    return color_map


class AnimationHandler:
    frame = 0
    animate = False
    animation_handler = None
    end = False

    def __init__(self, animation_handler):
        self.end = False
        self.animation_handler = animation_handler

    def toggle(self):
        self.animate = not self.animate

    def stop(self):
        self.animate = False
        self.end = True
        if self.animation_handler:
            self.animation_handler._stop()


def propagate_labels_SSL(feature_matrix, initial_labels, distance_metric, neighborhood_fn, alpha_vector,
                         max_iterations, num_labeled_points, num_soft_labeled_points, labels=hcs_labels, use_gui=True):
    assert type(initial_labels) is np.ndarray
    assert len(initial_labels.shape) == 2
    ___("labels: %s" % labels)
    assert initial_labels.shape[1] == len(labels)
    class_prior = np.sum(initial_labels[:num_labeled_points], axis=0) / num_labeled_points
    ___("class prior: %s" % class_prior)
    '''
    Implementation of the label spreading algorithm (Zhou et al., 2004) on a graph, represented by
    its similarity matrix.
    Parameters:
    [TODO: add parameter information]
    '''
    ___("calculating pairwise distances for %i datapoints, %i dimensions each..." %
        (feature_matrix.shape[0], feature_matrix.shape[1]))
    pairwise = pdist(feature_matrix)
    ___("  pairwise distances: %s... (%i distances)" % (str(pairwise[1:5]).replace(']', ''), len(pairwise)))
    ___("getting the square form...",)
    pairwise_matrix = squareform(pairwise)
    ___("%ix%i pairwise distances matrix created" % (pairwise_matrix.shape[0], pairwise_matrix.shape[1]))

    # Create weight matrix W:
    if neighborhood_fn == 'exp':
        exp_matrix = np.vectorize(lambda x: exp(-x))
        cutoff = None
        #cutoff = 0.1
        if cutoff:
            np.place(pairwise, pairwise > cutoff, 0)
        W = squareform(exp_matrix(pairwise))
    elif neighborhood_fn.startswith('knn'):
        k = int(neighborhood_fn[len('knn'):])
        ___("Using %i nearest neighbors" % k)
        knns = np.argsort(pairwise_matrix, axis=1)[:, :k + 1]
        W = np.array([[1 if j in knns[i] else 0 for j in range(pairwise_matrix.shape[0])]
                      for i in range(pairwise_matrix.shape[0])])
    np.fill_diagonal(W, 0)

    D_sqrt_inv = np.matrix(np.diag(1.0 / np.sqrt(np.sum(W, axis=1))))  # D^{-1/2}
    Alpha = np.matrix(np.diag(alpha_vector))
    OneMinusAlpha = np.matrix(np.diag(1 - np.array(alpha_vector)))
    Y_0 = initial_labels.T
    Y_t = initial_labels.T
    ___("Calculating Laplacian..."),
    Laplacian = D_sqrt_inv * W * D_sqrt_inv
    ___("Iterating up to %i times..." % max_iterations)

    # setup plot
    if use_gui:
        labeled_array = generate_color_map(Y_t, num_labeled_points, num_soft_labeled_points, 1)
        fig = plt.figure()
        splot_orig = fig.add_subplot(121)
        splot = fig.add_subplot(122)
        scat_orig_labeled = splot_orig.scatter(feature_matrix[: num_labeled_points, 0],
                                               feature_matrix[: num_labeled_points, 1],
                                               s=80, c=labeled_array[:num_labeled_points],
                                               cmap=cm.gist_ncar, vmin=0, vmax=1, alpha=0.65, marker='o')
        scat_orig_soft = splot_orig.scatter(feature_matrix[num_labeled_points:num_labeled_points + num_soft_labeled_points, 0],
                                            feature_matrix[num_labeled_points:num_labeled_points + num_soft_labeled_points, 1],
                                            s=35,
                                            c=labeled_array[num_labeled_points:num_labeled_points + num_soft_labeled_points],
                                            cmap=cm.gist_ncar, vmin=0, vmax=1, marker='p', alpha=0.6)
        scat_orig_unlabeled = splot_orig.scatter(feature_matrix[num_labeled_points + num_soft_labeled_points:, 0],
                                                 feature_matrix[num_labeled_points + num_soft_labeled_points:, 1], s=30,
                                                 c=labeled_array[num_labeled_points + num_soft_labeled_points:],
                                                 cmap=cm.gist_ncar, vmin=0, vmax=1, alpha=0.5)
        splot_orig.set_xlabel("feature 1")
        splot_orig.set_ylabel("feature 2")
        splot_orig.set_title("Original label assignment")
        splot.set_xlabel("feature 1")
        splot.set_ylabel("feature 2")
        splot.set_title("Current label assignment")
        scat_labeled = splot.scatter(feature_matrix[:num_labeled_points, 0], feature_matrix[:num_labeled_points, 1], s=80,
                                     c=labeled_array[:num_labeled_points], cmap=cm.gist_ncar, vmin=0, vmax=1, marker='o',
                                     alpha=0.55)
        scat_soft_labeled = splot.scatter(feature_matrix[num_labeled_points:num_labeled_points + num_soft_labeled_points, 0],
                                          feature_matrix[num_labeled_points:num_labeled_points + num_soft_labeled_points, 1],
                                          s=35,
                                          c=labeled_array[num_labeled_points:num_labeled_points + num_soft_labeled_points],
                                          cmap=cm.gist_ncar, vmin=0, vmax=1, marker='p', alpha=0.6)
        scat_unlabeled = splot.scatter(feature_matrix[num_labeled_points + num_soft_labeled_points:, 0],
                                       feature_matrix[num_labeled_points + num_soft_labeled_points:, 1], s=30,
                                       c=labeled_array[num_labeled_points + num_soft_labeled_points:], cmap=cm.gist_ncar,
                                       vmin=0, vmax=1, alpha=0.5)
        header = plt.figtext(0.5, 0.96, "Label propagation", weight="bold", size="large", ha="center")

        axStartAnimation = plt.axes([0.89, 0.005, 0.1, 0.055])
        bStartPauseAnimation = widgets.Button(axStartAnimation, 'Start')

        plt.subplots_adjust(bottom=0.17)
        rax_orig = plt.axes([0.225, 0.005, 0.155, 0.12])
        label_scatter_orig = {'Labeled': scat_orig_labeled,
                              'Soft-labeled': scat_orig_soft,
                              'Unlabeled': scat_orig_unlabeled}
        check_orig = widgets.CheckButtons(rax_orig, label_scatter_orig.keys(), (True, True, True))

        rax = plt.axes([0.65, 0.005, 0.155, 0.12])
        label_scatter = {'Labeled': scat_labeled,
                         'Soft-labeled': scat_soft_labeled,
                         'Unlabeled': scat_unlabeled}
        check = widgets.CheckButtons(rax, label_scatter.keys(), (True, True, True))

        def hide_show_orig(label):
            scatter_plot = label_scatter_orig[label]
            scatter_plot.set_visible(not scatter_plot.get_visible())
            plt.draw()

        def hide_show(label):
            scatter_plot = label_scatter[label]
            scatter_plot.set_visible(not scatter_plot.get_visible())
            plt.draw()

        check.on_clicked(hide_show)
        check_orig.on_clicked(hide_show_orig)

    assert Laplacian.shape[1] == Alpha.shape[0]

    def init_scatter():
        pass
        #scat.set_array(labeled_array)

    Y_scaled = Y_t.T.copy()

    # Animation frame
    def get_propagated_labels(t):
        if aHandler.animate:
            aHandler.frame = aHandler.frame + 1
            try:
                Y_old = Y_t.copy()
                for i in range(initial_labels.shape[1]):
                    Y_new = (Laplacian * Alpha).dot(Y_t[i]) + OneMinusAlpha.dot(Y_0[i])
                    #np.copyto(Y_t[i], Y_new)
                    Y_t[i] = Y_new

                # class mass normalization
                class_prior = np.sum(initial_labels[:num_labeled_points], axis=0) / float(num_labeled_points)
                class_mass = np.sum(initial_labels[num_labeled_points:], axis=0) / float(initial_labels.shape[0] - num_labeled_points)
                class_scaling = class_prior / class_mass
                #np.copyto(Y_scaled, class_scaling * Y_t.T)
                Y_scaled[:] = class_scaling * Y_t.T
                color_map = generate_color_map(Y_scaled.T, num_labeled_points, num_soft_labeled_points)
                #color_map = generate_color_map(np.array(Y_t), num_labeled_points, num_soft_labeled_points)
                if use_gui:
                    if t == 0:  # display first label assignment
                        scat_orig_unlabeled.set_array(color_map)
                    scat_labeled.set_array(color_map[:num_labeled_points])
                    scat_soft_labeled.set_array(color_map[num_labeled_points: num_labeled_points + num_soft_labeled_points])
                    scat_unlabeled.set_array(color_map[num_labeled_points + num_soft_labeled_points:])
                    header.set_text("Label propagation. Iteration %i" % aHandler.frame)
                ___([euclidean(Y_old[i], Y_t[i]) for i in
                     range(Y_old.shape[0])])
                if all([euclidean(Y_old[i], Y_t[i]) < 1e-2 for i in range(Y_old.shape[0])]) or aHandler.frame >= max_iterations:
                    if use_gui:
                        header.set_text("%s %s" % (header.get_text(), "[end]"))
                    aHandler.stop()
            except KeyboardInterrupt:
                aHandler.stop()
                return

    if use_gui:
        aHandler = AnimationHandler(animation.FuncAnimation(fig, get_propagated_labels, init_func=init_scatter))

        def start_stop(mouse_event):
            aHandler.toggle()
            bStartPauseAnimation.label.set_text("Pause" if aHandler.animate else "Start")
            plt.draw()

        bStartPauseAnimation.on_clicked(start_stop)
        plt.show()
    else:
        aHandler = AnimationHandler(None)
        aHandler.animate = True
        while(not aHandler.end):
            get_propagated_labels(0)
    return get_labels(Y_scaled.T, labels)


def normalize(M, class_weights=1):
    '''
    Student's t-statistic normalization, by using estimated mean and standard deviation
    '''
    assert type(class_weights) is int or type(class_weights) is np.ndarray and np.shape(class_weights)[0] == np.shape(M)[1]
    return class_weights * (M - np.mean(M, 0)) / np.std(M, 0)


def get_label_matrix(label_vector, class_labels):
    label_matrix = np.zeros([len(label_vector), len(class_labels)])
    for i in range(len(label_vector)):
        if label_vector[i] == -1:
            label_matrix[i, :] += 0.5
        else:
            label_matrix[i, label_vector[i] - 1] = 1
    return label_matrix


def setup_validation_matrix(labeled_file_references, soft_labeled_path, feature_list,
                            labeled_sample_size, class_sampling, alpha_labeled,
                            alpha_unlabeled, alpha_soft_labeled,
                            class_labels=hcs_soft_labels, ignore_labels=[6],
                            normalize_data=True):
    alpha_vector = []

    validation_data = [read_arff_file(input_file, feature_list, ignore_labels=ignore_labels)
                       for input_file in labeled_file_references]
    validation_points = np.concatenate(validation_data)
    np.random.shuffle(validation_points)
    labeled_points = validation_points[:labeled_sample_size]
    unlabeled_points = validation_points[labeled_sample_size:]
    expected_labels = unlabeled_points[:, -1].copy()
    unlabeled_points[:, -1] = -1
    soft_labeled_sample_size = len(unlabeled_points)

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
            class_sample_boundaries = np.rint(np.linspace(0, soft_labeled_sample_size, len(class_labels) + 1))
            class_sample_sizes = class_sample_boundaries[1:] - class_sample_boundaries[:-1]
            i = iter(class_sample_sizes)
            class_sample_size = {class_label: int(i.next()) for class_label in class_labels.keys()}
            soft_labeled_data = [get_sample(np.concatenate(soft_labeled_set), class_sample_size[soft_labeled_set_name])
                                 for soft_labeled_set_name, soft_labeled_set in soft_labeled_data.iteritems()]
        else:
            soft_labeled_data = np.concatenate(soft_labeled_data.values())
        soft_labeled_points = np.concatenate(soft_labeled_data)
        soft_labeled_alphas = [alpha_soft_labeled[label] for label in soft_labeled_points[:, -1]]
        alpha_vector += soft_labeled_alphas
    alpha_vector += [alpha_unlabeled] * len(unlabeled_points)

    if soft_labeled_points is None:
        soft_labeled_points = []
        M = np.concatenate([labeled_points, unlabeled_points])
    else:
        M = np.concatenate([labeled_points, soft_labeled_points, unlabeled_points])

    initial_labels = get_label_matrix(M[:, -1], class_labels)
    M = M[:, :-1]

    if normalize_data:
        # M = np.column_stack([normalize(M), initial_labels])
        scores = np.array([1.3275, 1.1739, 1.0605, 0.9868])
        weights = np.max(scores) / scores
        M = normalize(M, weights)

    return (M, initial_labels, alpha_vector, len(labeled_points), len(soft_labeled_points), expected_labels)


def setup_feature_matrix(unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list,
                         soft_labeled_sample_size, unlabeled_sample_size, labeled_sample_size, class_sampling,
                         num_labels, alpha_labeled=0.95, alpha_unlabeled=0.95, alpha_soft_labeled=hcs_soft_label_alphas,
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
            soft_labeled_data = [get_sample(np.concatenate(soft_labeled_set), class_sample_size[soft_labeled_set_name])
                                 for soft_labeled_set_name, soft_labeled_set in soft_labeled_data.iteritems()]
        else:
            soft_labeled_data = np.concatenate(soft_labeled_data.values())
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
        soft_labeled_points = []
        M = np.concatenate([labeled_points, unlabeled_points])
    else:
        M = np.concatenate([labeled_points, soft_labeled_points, unlabeled_points])

    initial_labels = get_label_matrix(M[:, -1], class_labels)
    M = M[:, :-1]

    if normalize_data:
        # M = np.column_stack([normalize(M), initial_labels])
        scores = np.array([1.3275, 1.1739, 1.0605, 0.9868])
        weights = np.max(scores) / scores
        M = normalize(M, weights)
    return (M, initial_labels, alpha_vector, len(labeled_points), len(soft_labeled_points))


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


def create_dummy_data(labeled_points=100, soft_labeled_points=100, unlabeled_points=100, normalize_data=True, num_labels=2):
    center_labeled_inf, width_labeled_inf = [1.5, 1.5], 0.75
    #center_labeled_inf, width_labeled_inf = [2.5, 2], 0.15
    center_labeled_noninf, width_labeled_noninf = [4.5, 4.5], 0.75
    center_soft_inf, width_soft_inf = [1.5, 1.5], 1.4
    center_soft_noninf, width_soft_noninf = 4.5, 1.4
    #center_unlabeled1, width_unlabeled1 = 3.0, 1.0
    #center_unlabeled2, width_unlabeled2 = 2.9, 0.1
    center_unlabeled1, width_unlabeled1 = 3.0, 3.0  # 75%
    center_unlabeled2, width_unlabeled2 = 3.0, 3.0  # 25%

    M = np.column_stack((get_uniform_sample(center_labeled_inf, width_labeled_inf, labeled_points / 2), [[0.0, 1.0]] * (labeled_points / 2)))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_labeled_noninf, width_labeled_noninf, labeled_points / 2), [[1.0, 0.0]] * (labeled_points / 2)))))

    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft_inf, width_soft_inf, soft_labeled_points / 2), [[0.0, 1.0]] * (soft_labeled_points / 2)))))
    #M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft_noninf, width_soft_inf, 3), [[0.0, 1.0]] * 3))))
    #M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft_noninf, width_soft_noninf, soft_labeled_points / 2 - 3), [[1.0, 0.0]] * (soft_labeled_points / 2 - 3)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft_noninf, width_soft_noninf, soft_labeled_points / 2), [[1.0, 0.0]] * (soft_labeled_points / 2)))))

    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_unlabeled1, width_unlabeled1, int(0.75 * unlabeled_points)), [[0.5, 0.5]] * int(0.75 * unlabeled_points)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_unlabeled2, width_unlabeled2, unlabeled_points - int(0.75 * unlabeled_points)), [[0.5, 0.5]] * (unlabeled_points - int(0.75 * unlabeled_points))))))

    alpha_vector = np.array([0.1] * labeled_points + [0.25] * soft_labeled_points + [0.75] * unlabeled_points)
    if normalize_data:
        M = np.column_stack([normalize(M[:, :-num_labels]), M[:, -num_labels:]])
    return M, alpha_vector, labeled_points, soft_labeled_points


def create_dummy_data2(labeled_points=100, soft_labeled_points=100, unlabeled_points=100, normalize_data=True, num_labels=2):
    center_labeled_1, width_labeled_1 = [1, 1], 2.5
    center_labeled_2, width_labeled_2 = [5, 5], 2.5
    center_labeled_3, width_labeled_3 = [1, 5], 2.5
    center_labeled_4, width_labeled_4 = [4, 1], 2.5
    center_soft1, width_soft1 = [1, 1], 2.8
    center_soft2, width_soft2 = [5, 5], 2.8
    center_soft3, width_soft3 = [1, 5], 2.8
    center_soft4, width_soft4 = [5, 1], 2.8
    num_mislabeled_points = 4

    center_unlabeled, width_unlabeled = 3.0, 5.0  # 25%

    M = np.column_stack((get_uniform_sample(center_labeled_1, width_labeled_1, labeled_points / 4),
                         [[1.0, 0.0, 0.0, 0.0]] * (labeled_points / 4)))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_labeled_2, width_labeled_2, labeled_points / 4),
                                            [[0.0, 1.0, 0.0, 0.0]] * (labeled_points / 4)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_labeled_3, width_labeled_3, labeled_points / 4),
                                            [[0.0, 0.0, 1.0, 0.0]] * (labeled_points / 4)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_labeled_4, width_labeled_4, labeled_points / 4),
                                            [[0.0, 0.0, 0.0, 1.0]] * (labeled_points / 4)))))

    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft1, width_soft1, soft_labeled_points / 4),
                                            [[1.0, 0.0, 0.0, 0.0]] * (soft_labeled_points / 4)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft2, width_soft2, soft_labeled_points / 4),
                                            [[0.0, 1.0, 0.0, 0.0]] * (soft_labeled_points / 4)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft3, width_soft3, soft_labeled_points / 4),
                                            [[0.0, 0.0, 1.0, 0.0]] * (soft_labeled_points / 4)))))
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft4, width_soft4, num_mislabeled_points),
                                            [[0.0, 1.0, 0.0, 0.0]] * num_mislabeled_points))))
    remaining_points = soft_labeled_points - 3 * (soft_labeled_points / 4) - num_mislabeled_points
    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_soft4, width_soft4, remaining_points),
                                            [[0.0, 0.0, 0.0, 1.0]] * remaining_points))))

    M = np.concatenate((M, np.column_stack((get_uniform_sample(center_unlabeled, width_unlabeled, unlabeled_points),
                                            [[0.5, 0.5, 0.5, 0.5]] * (unlabeled_points)))))

    alpha_vector = np.array([0.05] * labeled_points + [0.2] * soft_labeled_points + [0.75] * unlabeled_points)
    initial_labels = M[:, -num_labels:]
    if normalize_data:
        M = normalize(M[:, :-num_labels])
    else:
        M = M[:, :-num_labels]
    return M, initial_labels, alpha_vector, labeled_points, soft_labeled_points


def main(argv):
    '''
    Reads the parameters from the command line, loads the referred labeled and unlabeled files, and
    applies the label propagation algorithm iteratively
    '''
    unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list, soft_labeled_sample_size, \
        unlabeled_sample_size, labeled_sample_size, class_sampling, distance_metric, neighborhood_fn, alpha_labeled, \
        alpha_unlabeled, alpha_soft_uninfected, alpha_soft_infected, max_iterations, test, \
        validation, use_gui = process_cmdline(argv)

    seed(7283)
    labels = hcs_labels
    expected_labels = None

    if test:
        labels = dummy_labels2
        M, initial_labels, alpha_vector, labeled_points, soft_labeled_points = \
            create_dummy_data2(labeled_sample_size if labeled_sample_size > 0 else 200,
                               unlabeled_sample_size / 2, unlabeled_sample_size / 2, num_labels=len(labels), normalize_data=True)
    elif validation:
        alpha_soft_labeled = {1: alpha_soft_uninfected,
                              2: alpha_soft_infected,
                              3: alpha_soft_infected,
                              4: alpha_soft_infected,
                              5: alpha_soft_infected}
        M, initial_labels, alpha_vector, labeled_points, soft_labeled_points, expected_labels = \
            setup_validation_matrix(labeled_file_references, soft_labeled_path, feature_list, labeled_sample_size,
                                    class_sampling, alpha_labeled, alpha_unlabeled, alpha_soft_labeled,
                                    ignore_labels=['6'], normalize_data=True)
    else:
        M, initial_labels, alpha_vector, labeled_points, soft_labeled_points = \
            setup_feature_matrix(unlabeled_file_references, soft_labeled_path, labeled_file_references,
                                 feature_list, soft_labeled_sample_size, unlabeled_sample_size,
                                 labeled_sample_size,
                                 class_sampling, alpha_labeled=0.1, alpha_unlabeled=0.9, ignore_labels=['6'],
                                 num_labels=len(labels), normalize_data=True)

    Y = propagate_labels_SSL(M, initial_labels, distance_metric, neighborhood_fn, alpha_vector, max_iterations,
                             labeled_points, soft_labeled_points, labels=labels, use_gui=use_gui)
    if validation and type(expected_labels) is np.ndarray:
        Y_unlabeled = Y[-len(expected_labels):]
        classwise_precision = {label: np.sum(np.all([expected_labels == label, expected_labels == Y_unlabeled], axis=0)) /
                               (1. * np.sum(Y_unlabeled == label)) for label in labels}
        classwise_recall = {label: np.sum(np.all([expected_labels == label, expected_labels == Y_unlabeled], axis=0)) /
                            (1. * np.sum(expected_labels == label)) for label in labels}
        print "a-labeled, %f, a-unlabeled, %f, a-soft-uninf, %f, a-soft-inf, %f, nf, %s, precision, %s, recall, %s" % \
            (alpha_labeled, alpha_unlabeled, alpha_soft_uninfected, alpha_soft_infected, neighborhood_fn,
             classwise_precision, classwise_recall)
    return Y


def load_test():
    '''
    Creates and uses dummy data to test the label propagation algorithm
    '''
    unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list, soft_labeled_sample_size, \
        unlabeled_sample_size, labeled_sample_size, class_sampling, distance_metric, neighborhood_fn, alpha, \
        max_iterations, test = generate_test_data()

    M, initial_labels, alpha_vector, labeled_points, soft_labeled_points = \
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

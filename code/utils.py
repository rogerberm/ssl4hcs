import sys
from glob import glob
from random import sample
from os.path import basename, dirname


MESSAGES = {'NO_LABELED_FILE_MESSAGE': 'Invalid or no labeled file specified',
            'NO_UNLABELED_FILE_MESSAGE': 'Invalid or no unlabeled file specified',
            'INVALID_FEATURE_LIST_MESSAGE': 'Invalid feature list',
            'INVALID_SOFT_LABELED_FILE_PATH': 'Invalid soft labeled file path'}


hcs_soft_labels = {'bPEGI-29': 1,  # non-infected = 1
                   'bPEGI-26': 2,  # inGroup1 = 2
                   'bPEGI-27': 3,  # inGroup2 = 3
                   'bPEGI-25': 4,  # inGroup3 = 4
                   'bPEGI-28': 5}  # inGroup4 = 5


dummy_soft_labels = {'non-infected': 1,
                     'infected4': 5}


def print_help():
    print('''
	        ---help message here---
            TODO: include here the help message
            make sure to include quotation marks when using wildcards for file names
            ''')


def assert_argument(obj, message):
    if not obj:
        print message
        sys.exit()


def parentdir(file_path):
    return basename(dirname(file_path))


def get_files(path, recursive=True):
    return glob("%s*/*.txt" % path)


def get_sample(data_set, sample_size):
    if sample_size > len(data_set) or sample_size < 0:
        return data_set
    else:
        return data_set[sample(range(len(data_set)), sample_size)]


def generate_test_data():
    unlabeled_file_references = ["../dummydata/unlabeled/bDummy/bDummy1.txt",
                                 "../dummydata/unlabeled/bDummy/bDummy2.txt",
                                 "../dummydata/unlabeled/bDummy/bDummy3.txt"]
    soft_labeled_path = "../dummydata/soft/"
    labeled_file_references = ["../dummydata/labeled/labeled.arff"]
    # Defaults:
    feature_list = [1, 2]
    sample_size = 3000
    soft_labeled_sample_size = sample_size / 2
    unlabeled_sample_size = sample_size / 2
    class_sampling = True
    distance_metric = 'euclidean'
    neighborhood_function = 'exp'
    alpha = 0.94
    max_iterations = 1000

    return (unlabeled_file_references, soft_labeled_path, labeled_file_references, feature_list,
            soft_labeled_sample_size, unlabeled_sample_size, class_sampling, distance_metric,
            neighborhood_function, alpha, max_iterations)

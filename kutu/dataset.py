from abc import ABC


class Dataset(ABC):
    """
    Dataset class. The most base class for dataset construction.
    Variables are following:
        _root_folder: Path to root folder that dataset will be constructed.
        _num_samples: The number of samples in the dataset.
        _num_train_samples: The number of train samples in the dataset.
        _num_val_samples: The number of validation samples in the dataset.
        _num_test_samples: The number of test samples in the dataset.

    All variables are implemented as private. Therefore, they have getter and setter properties.
    """

    _root_folder: str
    _num_samples: int
    _num_train_samples: int
    _num_val_samples: int
    _num_test_samples: int

    def __init__(self):
        self._root_folder = ""
        self._num_samples = 0
        self._num_train_samples = 0
        self._num_val_samples = 0
        self._num_test_samples = 0

    @property
    def root_folder(self):
        return self._root_folder

    @root_folder.setter
    def root_folder(self, r):
        if type(r) is not str:
            raise TypeError('Arg (path to source file of the dataset) must be String but got type %s' % type(r))
        elif r == "":
            raise ValueError('Arg (root_folder) cannot be empty string')
        self._root_folder = r

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, n):
        if type(n) is not int:
            raise TypeError('Arg (num_samples) must be integer but got type %s' % type(n))
        elif n < 0:
            raise ValueError(('Arg (num_samples) cannot be negative but got value %d' % n))
        self._num_samples = n

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @num_train_samples.setter
    def num_train_samples(self, n):
        if type(n) is not int:
            raise TypeError('Arg (num_train_samples) must be integer but got type %s' % type(n))
        elif n < 0:
            raise ValueError(('Arg (num_train_samples) cannot be negative but got value %d' % n))
        self._num_train_samples = n

    @property
    def num_val_samples(self):
        return self._num_val_samples

    @num_val_samples.setter
    def num_val_samples(self, n):
        if type(n) is not int:
            raise TypeError('Arg (num_val_samples) must be integer but got type %s' % type(n))
        elif n < 0:
            raise ValueError(('Arg (num_val_samples) cannot be negative but got value %d' % n))
        self._num_val_samples = n

    @property
    def num_test_samples(self):
        return self._num_test_samples

    @num_test_samples.setter
    def num_test_samples(self, n):
        if type(n) is not int:
            raise TypeError('Arg (num_test_samples) must be integer but got type %s' % type(n))
        elif n < 0:
            raise ValueError(('Arg (num_test_samples) cannot be negative but got value %d' % n))
        self._num_test_samples = n

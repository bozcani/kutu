import numpy as np
import cv2
from abc import ABC, abstractmethod


class Dataset(ABC):

    def __init__(self):
        self._source = None
        self._root_folder = None
        self._num_sample = None
        self._num_vis = None
        self._num_train_samples = None
        self._num_val_samples = None
        self._num_test_samples = None

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, s):
        if type(s) is not str:
            raise TypeError('Arg (path to source file of the dataset) must be String but got type %s' % type(s))
        self._source = s

    @property
    def root_folder(self):
        return self._root_folder

    @root_folder.setter
    def root_folder(self, r):
        if type(r) is not str:
            raise TypeError('Arg (path to source file of the dataset) must be String but got type %s' % type(r))
        self._root_folder = r


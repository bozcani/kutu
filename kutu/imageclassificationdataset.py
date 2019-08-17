from kutu.dataset import Dataset


class ImageClassificationDataset(Dataset):
    """
    Class for image classification datasets. Based on Dataset object instance.
    Variables are following:
        _image_shape: Shape of the image samples. Should be a tuple of integers, and have size of 2 or 3.
            2 for grayscale images which have format (num. of rows, num. of cols)
            3 for colored images which have format (num. of rows, num. of cols, num. of channels)

        _train_mean: The mean value of training samples. May be useful for pre-processing.
        _num_classes: The number of classes in the dataset.
        _class_labels: List of class labels. Should be list of strings.
    """

    _image_shape: tuple
    _train_mean: float
    _num_classes: int
    _class_labels: list

    def __init__(self):
        super(ImageClassificationDataset, self).__init__()
        self._image_shape = (0, 0)
        self._train_mean = 0.
        self._num_classes = 0
        self._class_labels = []

    @property
    def image_shape(self):
        return self._image_shape

    @image_shape.setter
    def image_shape(self, s):
        if type(s) is not tuple:
            raise TypeError('Arg (image_shape) must be tuple but got type %s' % type(s))
        elif len(s) == 2 or len(s) == 3:
            self._image_shape = s
        else:
            raise ValueError('Arg (image_shape) must have length of 2 or 3. But has length %s ' % str(len(s)))

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, n):
        if type(n) is not int:
            raise TypeError('Arg (num_classes) must be int but got type %s' % type(n))
        elif n < 2:
            raise ValueError('Arg (num_classes) cannot be lower than 2. But got value %d ' % n)
        else:
            self._num_classes = n

    @property
    def class_labels(self):
        return self._class_labels

    @class_labels.setter
    def class_labels(self, cl):
        if type(cl) is not list:
            raise TypeError('Arg (num_classes) must be list but got type %s' % type(cl))
        elif len(cl) < 2:
            raise ValueError('Arg (class_labels) cannot have lower size than 2. But has size %d ' % len(cl))
        else:
            self._class_labels = cl

    def calculate_train_mean(self):
        raise NotImplemented()

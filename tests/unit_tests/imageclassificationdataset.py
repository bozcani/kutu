import unittest
from kutu.imageclassificationdataset import ImageClassificationDataset


class TestImageClassificationDataset(unittest.TestCase):

    def test_image_shape(self):
        """
        Test image_shape variable.
        """
        dataset = ImageClassificationDataset()

        # Check TypeError for invalid inputs.
        with self.assertRaises(TypeError):
            dataset.image_shape = 3
        with self.assertRaises(TypeError):
            dataset.image_shape = 3.0
        with self.assertRaises(TypeError):
            dataset.image_shape = [3, 3]
        with self.assertRaises(TypeError):
            dataset.image_shape = "3"
        with self.assertRaises(TypeError):
            dataset.image_shape = None

        # Check ValueError for invalid input shapes.
        with self.assertRaises(ValueError):
            dataset.image_shape = (128,)
        with self.assertRaises(ValueError):
            dataset.image_shape = (128, 128, 3, 3)

        # Set input_shape variable.
        dataset.image_shape = (64, 64, 3)
        self.assertEqual(dataset.image_shape, (64, 64, 3))

        dataset.image_shape = (64, 64, 4)
        self.assertEqual(dataset.image_shape, (64, 64, 4))

        dataset.image_shape = (128, 128)
        self.assertEqual(dataset.image_shape, (128, 128))

    def test_class_labels(self):
        """
        Test class_labels variable.
        """
        dataset = ImageClassificationDataset()

        # Check TypeError for invalid inputs.
        with self.assertRaises(TypeError):
            dataset.class_labels = 3
        with self.assertRaises(TypeError):
            dataset.class_labels = 3.0
        with self.assertRaises(TypeError):
            dataset.class_labels = (3, 3)
        with self.assertRaises(TypeError):
            dataset.class_labels = "3"
        with self.assertRaises(TypeError):
            dataset.class_labels = None

        # Check ValueError for invalid input shapes.
        with self.assertRaises(ValueError):
            dataset.class_labels = []
        with self.assertRaises(ValueError):
            dataset.class_labels = ['only_1_class']

        # Set input_shape variable.
        dataset.class_labels = ['class_name_1', 'class_name_2']
        self.assertEqual(dataset.class_labels, ['class_name_1', 'class_name_2'])

        dataset.class_labels = ['class_name_1', 'class_name_2', 'class_name_3', 'class_name_4']
        self.assertEqual(dataset.class_labels, ['class_name_1', 'class_name_2', 'class_name_3', 'class_name_4'])


if __name__ == '__main__':
    unittest.main()

import unittest
from kutu.dataset import Dataset


class TestDataset(unittest.TestCase):
    """
    Unit tests for Dataset class.
    Run 5 tests in total.
    """

    def test_root_folder(self):
        """
        Test root_folder variable.
        :return:
        """
        dataset = Dataset()

        # Check TypeError for invalid inputs.
        with self.assertRaises(TypeError):
            dataset.root_folder = 3
        with self.assertRaises(TypeError):
            dataset.root_folder = 3.0
        with self.assertRaises(TypeError):
            dataset.root_folder = [3, 3]
        with self.assertRaises(TypeError):
            dataset.root_folder = (3, 3)
        with self.assertRaises(TypeError):
            dataset.root_folder = None

        # Check ValueError for empty string.
        with self.assertRaises(ValueError):
            dataset.source = ""

        # Set source variable.
        dataset.root_folder = 'path/to/root_folder'
        self.assertEqual(dataset.root_folder, 'path/to/root_folder')

        dataset.root_folder = 'path2/to/root_folder2'
        self.assertEqual(dataset.root_folder, 'path2/to/root_folder2')

    def test_num_samples(self):
        """
        Test num_samples variable.
        :return:
        """
        dataset = Dataset()

        # Check TypeError for invalid inputs.
        with self.assertRaises(TypeError):
            dataset.num_samples = "3"
        with self.assertRaises(TypeError):
            dataset.num_samples = 3.
        with self.assertRaises(TypeError):
            dataset.num_samples = [3, 3]
        with self.assertRaises(TypeError):
            dataset.num_samples = (3, 3)
        with self.assertRaises(TypeError):
            dataset.num_samples = None

        # Check ValueError for negative num_samples input.
        with self.assertRaises(ValueError):
            dataset.num_samples = -1
        with self.assertRaises(ValueError):
            dataset.num_samples = -9999999

        # Set num_samples variable.
        dataset.num_samples = 3
        self.assertEqual(dataset.num_samples, 3)

        dataset.num_samples = 4
        self.assertEqual(dataset.num_samples, 4)

        # Set large values.
        dataset.num_samples = 99999999
        self.assertEqual(dataset.num_samples, 99999999)

    def test_num_train_samples(self):
        """
        Test num_train_samples variable.
        :return:
        """
        dataset = Dataset()

        # Check TypeError for invalid inputs.
        with self.assertRaises(TypeError):
            dataset.num_train_samples = "3"
        with self.assertRaises(TypeError):
            dataset.num_train_samples = 3.
        with self.assertRaises(TypeError):
            dataset.num_train_samples = [3, 3]
        with self.assertRaises(TypeError):
            dataset.num_train_samples = (3, 3)
        with self.assertRaises(TypeError):
            dataset.num_train_samples = None

        # Check ValueError for negative num_train_samples input.
        with self.assertRaises(ValueError):
            dataset.num_train_samples = -1
        with self.assertRaises(ValueError):
            dataset.num_train_samples = -9999999

        # Set num_train_samples variable.
        dataset.num_train_samples = 3
        self.assertEqual(dataset.num_train_samples, 3)

        dataset.num_train_samples = 4
        self.assertEqual(dataset.num_train_samples, 4)

        # Set large values.
        dataset.num_train_samples = 99999999
        self.assertEqual(dataset.num_train_samples, 99999999)

    def test_num_val_samples(self):
        """
        Test num_val_samples variable.
        :return:
        """
        dataset = Dataset()

        # Check TypeError for invalid inputs.
        with self.assertRaises(TypeError):
            dataset.num_val_samples = "3"
        with self.assertRaises(TypeError):
            dataset.num_val_samples = 3.
        with self.assertRaises(TypeError):
            dataset.num_val_samples = [3, 3]
        with self.assertRaises(TypeError):
            dataset.num_val_samples = (3, 3)
        with self.assertRaises(TypeError):
            dataset.num_val_samples = None

        # Check ValueError for negative num_val_samples input.
        with self.assertRaises(ValueError):
            dataset.num_val_samples = -1
        with self.assertRaises(ValueError):
            dataset.num_val_samples = -9999999

        # Set num_val_samples variable.
        dataset.num_val_samples = 3
        self.assertEqual(dataset.num_val_samples, 3)

        dataset.num_val_samples = 4
        self.assertEqual(dataset.num_val_samples, 4)

        # Set large values.
        dataset.num_val_samples = 99999999
        self.assertEqual(dataset.num_val_samples, 99999999)

    def test_num_test_samples(self):
        """
        Test num_test_samples variable.
        :return:
        """
        dataset = Dataset()

        # Check TypeError for invalid inputs.
        with self.assertRaises(TypeError):
            dataset.num_test_samples = "3"
        with self.assertRaises(TypeError):
            dataset.num_test_samples = 3.
        with self.assertRaises(TypeError):
            dataset.num_test_samples = [3, 3]
        with self.assertRaises(TypeError):
            dataset.num_test_samples = (3, 3)
        with self.assertRaises(TypeError):
            dataset.num_test_samples = None

        # Check ValueError for negative num_test_samples input.
        with self.assertRaises(ValueError):
            dataset.num_test_samples = -1
        with self.assertRaises(ValueError):
            dataset.num_test_samples = -9999999

        # Set num_test_samples variable.
        dataset.num_test_samples = 3
        self.assertEqual(dataset.num_test_samples, 3)

        dataset.num_test_samples = 4
        self.assertEqual(dataset.num_test_samples, 4)

        # Set large values.
        dataset.num_val_samples = 99999999
        self.assertEqual(dataset.num_val_samples, 99999999)


if __name__ == '__main__':
    unittest.main()

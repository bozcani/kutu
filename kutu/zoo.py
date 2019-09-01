import numpy as np
import wget
import os
import tarfile
import cv2
import pickle


def get_cifar10(origin_url, root_folder, force_download=False):
    """
    Download CIFAR10 from the origin and create the folder structures as a classification dataset.
    :param origin_url: (str) Origin url of the dataset. Usually belongs to the dataset publisher organization.
    :param root_folder: (str) Root folder of the created folder structure.
    :param force_download: (bool) True for downloading anyways. False for skipping download if it has been done before.
    :return: Path to the root folder. Same with the root_folder parameter.
    """

    print('[INFO] ------- CIFAR10 ------- ')

    if force_download:
        path_to_source_zip = wget.download(origin_url)  # Use wget for downloading process.

    else:
        path_to_source_zip = origin_url.split('/')[-1]

        if os.path.isfile(path_to_source_zip):
            print('[INFO] Downloading is skipped. The dataset has been already downloaded: %s.' % path_to_source_zip)
        else:
            path_to_source_zip = wget.download(origin_url)  # Use wget for downloading process.

    print('[INFO] Folder structuring process STARTED.')

    # Create the root folder.
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)
    else:
        print('[INFO] The root folder already exists: %s' % root_folder)

    # Create sub folders.
    train_folder = os.path.join(root_folder, 'train')
    test_folder = os.path.join(root_folder, 'test')

    sub_folders = [train_folder, test_folder]
    num_classes = 10
    for i in range(num_classes):
        sub_folders.append(os.path.join(train_folder, str(i)))
        sub_folders.append(os.path.join(test_folder, str(i)))

    for sub_folder in sub_folders:
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
        else:
            print('[INFO] The sub folder already exists: %s' % os.path.join(sub_folder))

    # Save images into each subclass.
    # Extract the zip file (source_file) into the root folder.
    print('[INFO] The downloaded source file (%s) is being extracted into the root folder (%s) as new folder: %s.'
          % (path_to_source_zip, root_folder, 'cifar-10-batches-py'))
    if path_to_source_zip.endswith("tar.gz"):
        tar = tarfile.open(path_to_source_zip, "r:gz")
        tar.extractall(path=root_folder)
        tar.close()

    cifar_10_batches_py = os.path.join(root_folder, 'cifar-10-batches-py')
    print('[INFO] The source file has been extracted: %s' % cifar_10_batches_py)

    batch_labels = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

    print('[INFO] Train images are being saved into sub folders...')
    cnt = 0
    for batch_label in batch_labels:
        print('........%s is loading...' % batch_label)
        with open(os.path.join(cifar_10_batches_py, batch_label), 'rb') as fo:
            loaded_dict = pickle.load(fo, encoding='latin-1')

            data = loaded_dict['data']
            labels = loaded_dict['labels']

            for i in range(len(data)):
                img = data[i].reshape((3, 32, 32)).transpose(1, 2, 0)  # Transpose, channels first to channels last.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB color space.

                if batch_label.startswith('data'):
                    cv2.imwrite(os.path.join(train_folder, str(labels[i]), str(cnt) + '.png'), img.astype(np.uint8))

                elif batch_label.startswith('test'):
                    cv2.imwrite(os.path.join(test_folder, str(labels[i]), str(cnt) + '.png'), img.astype(np.uint8))

                cnt += 1

    return root_folder


#get_cifar10("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", 'cifar-10')

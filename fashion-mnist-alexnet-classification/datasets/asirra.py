import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def read_asirra_subset(subset_dir, one_hot=True, sample_size=None):
    """
    Load the Asirra Dogs vs. Cats data subset from disk
    and perform preprocessing for training AlexNet.
    :param subset_dir: str, path to the directory to read.
    :param one_hot: bool, whether to return one-hot encoded labels.
    :param sample_size: int, sample size specified when we are not using the entire set.
    :return: X_set: np.ndarray, shape: (N, H, W, C).
             y_set: np.ndarray, shape: (N, num_channels) or (N,).
    """

    # 일괄적으로 이미지를 256*256으로 리사이징하고 numpy의 np.ndarray 형태로 반환함함
   # Read train val data
    filename_list = os.listdir(subset_dir)
    set_size = len(filename_list)

    if sample_size is not None and sample_size < set_size:
        # Randomly sample subset of data when sample_size is specified
        # 샘플 크기가 정해져있으면 랜덤으로 데이터에서 Subset을 채택함
        filename_list = np.random.choice(filename_list, size=sample_size, replace=False) # 랜덤으로 뽑아옴
        set_size = sample_size
    else:
        # Just shuffle the filename list
        np.random.shuffle(filename_list) # 그렇지 않으면 그냥 섞는다

    # Pre-allocate data arrays
    X_set = np.empty((set_size, 256, 256, 3), dtype=np.float32)    # (N, H, W, 3)
    y_set = np.empty((set_size), dtype=np.uint8)                   # (N,)
    for i, filename in enumerate(filename_list):
        if i % 1000 == 0:
            print('Reading subset data: %d / %d...' % (i, set_size))
        label = filename.split('.')[0]
        if label == 'cat':
            y = 0
        else:  # label == 'dog'
            y = 1
        file_path = os.path.join(subset_dir, filename)
        img = imread(file_path)    # shape: (H, W, 3), range: [0, 255]
        img = resize(img, (256, 256), mode='constant').astype(np.float32)    # (256, 256, 3), [0.0, 1.0]
        X_set[i] = img
        y_set[i] = y

    if one_hot:
        # Convert labels to one-hot vectors, shape: (N, num_classes)
        y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
        y_set_oh[np.arange(set_size), y_set] = 1
        y_set = y_set_oh
    print('\nDone')

    return X_set, y_set


def random_crop_reflect(images, crop_l):
    """
    Perform random cropping and reflection from images. 랜덤으로 잘라서 쓰기
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region. 크롭하는 크기
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3] # 이미지 크기
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        # Randomly crop patch
        y = np.random.randint(H-crop_l)
        x = np.random.randint(W-crop_l)
        image = image[y:y+crop_l, x:x+crop_l]    # (h, w, C)

        # Randomly reflect patch horizontally
        reflect = bool(np.random.randint(2))
        if reflect:
            image = image[:, ::-1] # 좌우 반전 일 경우

        augmented_images.append(image)
    return np.stack(augmented_images)    # shape: (N, h, w, C)


def corner_center_crop_reflect(images, crop_l):
    """
    Perform 4 corners and center cropping and reflection from images,
    좌측 우측 상단 좌측 우측 하단, 중심 위치로 부터 각각 5개의 패치들을 추출함
    이미지 하나 당 총 10개의 패치를 반환
    resulting in 10x augmented patches.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, 10, h, w, C).
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        aug_image_orig = []
        # Crop image in 4 corners
        aug_image_orig.append(image[:crop_l, :crop_l])
        aug_image_orig.append(image[:crop_l, -crop_l:])
        aug_image_orig.append(image[-crop_l:, :crop_l])
        aug_image_orig.append(image[-crop_l:, -crop_l:])
        # Crop image in the center
        aug_image_orig.append(image[H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                                    W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
        aug_image_orig = np.stack(aug_image_orig)    # (5, h, w, C)

        # Flip augmented images and add it
        aug_image_flipped = aug_image_orig[:, :, ::-1]    # (5, h, w, C)
        aug_image = np.concatenate((aug_image_orig, aug_image_flipped), axis=0)    # (10, h, w, C)
        augmented_images.append(aug_image)
    return np.stack(augmented_images)    # shape: (N, 10, h, w, C)


def center_crop(images, crop_l):
    """
    Perform center cropping of images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3]
    cropped_images = []
    for image in images:    # image.shape: (H, W, C)
        # Crop image in the center
        cropped_images.append(image[H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                              W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
    return np.stack(cropped_images)


class DataSet(object):
    def __init__(self, images, labels=None):
        """
        Construct a new DataSet object. 생성자
        :param images: np.ndarray, shape: (N, H, W, C).
        :param labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if labels is not None:
            assert images.shape[0] == labels.shape[0], (
                'Number of examples mismatch, between images and labels.'
            )
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels    # NOTE: this can be None, if not given.
        self._indices = np.arange(self._num_examples, dtype=np.uint)    # image/label indices(can be permuted)
        self._reset()

    def _reset(self):
        """Reset some variables."""
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True,
                   fake_data=False):
        """
        Return the next `batch_size` examples from this dataset.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        미니배치 추출에 앞서 현재 데이터셋 내 이미지들의 순서를 랜덤으로 섞을지
        :param augment: bool, whether to perform data augmentation while sampling a batch.
        미니배치 추출할 때, 데이터 증강을 수행할 것인지
        :param is_train: bool, current phase for sampling.
        미니배치 추출을 위한 현재의 상황
        :param fake_data: bool, whether to generate fake data (for debugging).
        디버깅 목적으로 가짜 이미지 데이터를 생성할 것인지
        :return: batch_images: np.ndarray, shape: (N, h, w, C) or (N, 10, h, w, C).
                 batch_labels: np.ndarray, shape: (N, num_classes) or (N,).
        """
        if fake_data: # 디버깅용 가짜 이미지 데이터임
            fake_batch_images = np.random.random(size=(batch_size, 227, 227, 3))
            fake_batch_labels = np.zeros((batch_size, 2), dtype=np.uint8)
            fake_batch_labels[np.arange(batch_size), np.random.randint(2, size=batch_size)] = 1
            return fake_batch_images, fake_batch_labels

        start_index = self._index_in_epoch

        # 첫번째 epoch 에서는 전체 데이터 셋을 랜덤하게 섞음
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # Go to the next epoch, if current index goes beyond the total number of examples
        # 현재 인덱스가 전체 이미지 수를 넘어간 경우, 다음 epoch을 진행함
        if start_index + batch_size > self._num_examples:
            # Increment the number of epochs completed
            # 완료된 epoch 수 하나 추가
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            # 이전에 남은 이미지들을 가져옴
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # Shuffle the dataset, after finishing a single epoch
            # epoch이 끗나면 전체 데이터셋을 한번 섞음
            if shuffle:
                np.random.shuffle(self._indices)

            # Start the next epoch
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self.images[indices_rest_part]
            images_new_part = self.images[indices_new_part]
            batch_images = np.concatenate((images_rest_part, images_new_part), axis=0)
            if self.labels is not None:
                labels_rest_part = self.labels[indices_rest_part]
                labels_new_part = self.labels[indices_new_part]
                batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self.images[indices]
            if self.labels is not None:
                batch_labels = self.labels[indices]
            else:
                batch_labels = None

        if augment and is_train:
            # 학습상황에서의 데이터 증강을 수행함
            # 원본 크기의 이미지로 부터 227*227 patch를 추출해옴. 이미지 하나당 하나의 패치
            # 랜덤으로 데이터를 잘라서 선택해옴
            batch_images = random_crop_reflect(batch_images, 28)
        elif augment and not is_train:
            # Perform data augmentation, for evaluation phase(10x)
            # 예측상황에서의 데이터 증강을 수행함
            batch_images = corner_center_crop_reflect(batch_images, 28)
        else:
            # Don't perform data augmentation, generating center-cropped patches
            batch_images = center_crop(batch_images, 28)

        return batch_images, batch_labels

import time
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from models.layers import conv_layer, max_pool, fc_layer


class ConvNet(object):
    """Base class for Convolutional Neural Networks."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        Model initializer.
        :param input_shape: tuple, the shape of inputs (H, W, C), ranged [0.0, 1.0].
        :param num_classes: int, the number of classes.
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(tf.float32, [None] + [num_classes])
        self.is_train = tf.placeholder(tf.bool)

        # Build model and loss function
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        This should be implemented.
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
            - batch_size: int, batch size for each iteration.
            - augment_pred: bool, whether to perform augmentation for prediction.
        :return _y_pred: np.ndarray, shape: (N, num_classes).
        """
        batch_size = kwargs.pop('batch_size', 256)
        augment_pred = kwargs.pop('augment_pred', True)

        if dataset.labels is not None:
            assert len(dataset.labels.shape) > 1, 'Labels must be one-hot encoded.'
        num_classes = int(self.y.get_shape()[-1])
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps*batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False,
                                      augment=augment_pred, is_train=False)
            # if augment_pred == True:  X.shape: (N, 10, h, w, C)
            # else:                     X.shape: (N, h, w, C)

            # If performing augmentation during prediction,
            if augment_pred:
                y_pred_patches = np.empty((_batch_size, 10, num_classes),
                                          dtype=np.float32)    # (N, 10, num_classes)
                # compute predictions for each of 10 patch modes,
                for idx in range(10):
                    y_pred_patch = sess.run(self.pred,
                                            feed_dict={self.X: X[:, idx],    # (N, h, w, C)
                                                       self.is_train: False})
                    y_pred_patches[:, idx] = y_pred_patch
                # and average predictions on the 10 patches
                y_pred = y_pred_patches.mean(axis=1)    # (N, num_classes)
            else:
                # Compute predictions
                y_pred = sess.run(self.pred,
                                  feed_dict={self.X: X,
                                             self.is_train: False})    # (N, num_classes)

            _y_pred.append(y_pred)
        if verbose:
            print('Total prediction time(sec): {}'.format(time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)    # (N, num_classes)

        return _y_pred


class AlexNet(ConvNet):
    """AlexNet class."""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building AlexNet.
            - image_mean: np.ndarray, mean image for each input channel, shape: (C,).
            - dropout_prob: float, the probability of dropping out each unit in FC layer.
        :return d: dict, containing outputs on each layer.
        """
        d = dict()    # Dictionary to save intermediate values returned from each layer.
        X_mean = kwargs.pop('image_mean', 0.0)
        dropout_prob = kwargs.pop('dropout_prob', 0.0)
        num_classes = int(self.y.get_shape()[-1])

        # The probability of keeping each unit for dropout layers
        keep_prob = tf.cond(self.is_train,
                            lambda: 1. - dropout_prob,
                            lambda: 1.)

        # input
        X_input = self.X - X_mean    # perform mean subtraction

        # First Convolution Layer
        # conv1 - relu1 - pool1

        with tf.variable_scope('conv1'):
            # conv_layer(x, side_l, stride, out_depth, padding='SAME', **kwargs):
            d['conv1'] = conv_layer(X_input, 3, 1, 64, padding='SAME',
                                    weights_stddev=0.01, biases_value=1.0)
            print('conv1.shape', d['conv1'].get_shape().as_list())
        d['relu1'] = tf.nn.relu(d['conv1'])
        # max_pool(x, side_l, stride, padding='SAME'):
        d['pool1'] = max_pool(d['relu1'], 2, 1, padding='SAME')
        d['drop1'] = tf.nn.dropout(d['pool1'], keep_prob)
        print('pool1.shape', d['pool1'].get_shape().as_list())

        # Second Convolution Layer
        # conv2 - relu2 - pool2
        with tf.variable_scope('conv2'):
            d['conv2'] = conv_layer(d['pool1'], 3, 1, 128, padding='SAME',
                                    weights_stddev=0.01, biases_value=1.0)
            print('conv2.shape', d['conv2'].get_shape().as_list())
        d['relu2'] = tf.nn.relu(d['conv2'])
        d['pool2'] = max_pool(d['relu2'], 2, 1, padding='SAME')
        d['drop2'] = tf.nn.dropout(d['pool2'], keep_prob)
        print('pool2.shape', d['pool2'].get_shape().as_list())

        # Third Convolution Layer
        # conv3 - relu3
        with tf.variable_scope('conv3'):
            d['conv3'] = conv_layer(d['pool2'], 3, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=1.0)
            print('conv3.shape', d['conv3'].get_shape().as_list())
        d['relu3'] = tf.nn.relu(d['conv3'])
        d['pool3'] = max_pool(d['relu3'], 2, 1, padding='SAME')
        d['drop3'] = tf.nn.dropout(d['pool3'], keep_prob)
        print('pool3.shape', d['pool3'].get_shape().as_list())


        # Flatten feature maps
        f_dim = int(np.prod(d['drop3'].get_shape()[1:]))
        f_emb = tf.reshape(d['drop3'], [-1, f_dim])

        # fc4
        with tf.variable_scope('fc4'):
            d['fc4'] = fc_layer(f_emb, 1024,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu4'] = tf.nn.relu(d['fc4'])
        print('fc4.shape', d['relu4'].get_shape().as_list())

        # fc5
        with tf.variable_scope('fc5'):
            d['fc5'] = fc_layer(d['relu4'], 1024,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu5'] = tf.nn.relu(d['fc5'])
        print('fc5.shape', d['relu5'].get_shape().as_list())
        d['logits'] = fc_layer(d['relu5'], num_classes,
                               weights_stddev=0.01, biases_value=0.0)
        print('logits.shape', d['logits'].get_shape().as_list())

        # softmax
        d['pred'] = tf.nn.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments for regularization term.
            - weight_decay: float, L2 weight decay regularization coefficient.
        :return tf.Tensor.
        """
        weight_decay = kwargs.pop('weight_decay', 0.0005)
        variables = tf.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables])

        # Softmax cross-entropy loss function
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        softmax_loss = tf.reduce_mean(softmax_losses)

        return softmax_loss + weight_decay*l2_reg_loss

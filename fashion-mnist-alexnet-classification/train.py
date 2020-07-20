import os
import numpy as np
import tensorflow as tf
from datasets import asirra as dataset
from models.nn import AlexNet as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import AccuracyEvaluator as Evaluator
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
print(tf.__version__)
fashion_mnist = keras.datasets.mnist

""" 1. Load and split datasets """
# Load train val set and split into train/validation
print('Load Data')

# Cats_ and Dogs_
# C:\Users\Chan\Downloads\train
'''
root_dir = os.path.join('C:/','Users', 'Chan', 'Downloads')
trainval_dir = os.path.join(root_dir, 'train')
X_trainval, y_trainval = dataset.read_asirra_subset(trainval_dir, one_hot=True)  # Load train val set
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.2)    # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])
'''

# fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.astype(np.float32) / 255
test_images = test_images.astype(np.float32) / 255

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Sanity check
'''
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())
print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())
print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())
'''
print('Training set stats:')
print(train_images.shape)
print(train_images.min(), train_images.max())
# print((train_labels[:, 1] == 0).sum(), (train_labels[:, 1] == 1).sum())
print('Validation set stats:')
print(test_images.shape)
print(test_images.min(), test_images.max())
# print((test_labels[:, 1] == 0).sum(), (test_labels[:, 1] == 1).sum())



""" 2. Set training hyperparameters """
hp_d = dict()
image_mean = train_images.mean(axis=(0, 1, 2))    # mean image
print(image_mean)
# np.save('/tmp/asirra_mean.npy', image_mean)    # save mean image
hp_d['image_mean'] = image_mean

# FIXME: Training hyperparameters
hp_d['batch_size'] = 1
hp_d['num_epochs'] = 1

hp_d['augment_train'] = False
hp_d['augment_pred'] = True

hp_d['init_learning_rate'] = 0.01
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: Regularization hyperparameters
hp_d['weight_decay'] = 0.0005
hp_d['dropout_prob'] = 0.8

# FIXME: Evaluation hyperparameters
hp_d['score_threshold'] = 1e-4

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

""" 3. Build graph, initialize a session and start training """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([28, 28, 1], 10, **hp_d)

evaluator = Evaluator()

train_set = dataset.DataSet(train_images, train_labels)
test_set = dataset.DataSet(test_images, test_labels)

# train_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=1000000).batch(hp_d['batch_size'])
# test_set = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(hp_d['batch_size'])

optimizer = Optimizer(model, train_set, evaluator, test_set=test_set, **hp_d)

sess = tf.Session(graph=graph, config=config)
train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
from scipy import ndimage
import PIL
from PIL import Image
import sklearn
import sklearn.metrics
import argparse
import cv2
import math
import copy
import random

import imageio
import imgaug as ia
from imgaug import augmenters as iaa


class DataGenerator(tf.keras.utils.Sequence):
    '''Generates data for Keras'''
    def __init__(self, list_IDs, labels, batch_size=10,
                 side_size=128, depth_size=64, n_channels=1,
                 n_classes=4, shuffle=True, phase='train', augmentation=False,
                 class_weights=None):

        '''Initialization'''
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.side_size = side_size
        self.depth_size = depth_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.phase = phase
        self.augmentation = augmentation
        self.class_weights = class_weights

        self.on_epoch_end()

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle==True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        '''Generates data containing batch_size samples''' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, self.side_size, self.side_size, self.depth_size, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.n_classes), dtype=np.int32)

        if self.phase=='train':
            weights = np.empty((self.batch_size), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store class
            y[i,] = self.labels[ID]

            # Store sample
            X_aux = self._process_scan(ID, self.side_size, self.depth_size)

            if self.phase=='train':
                X[i,] = self._train_preprocessing(X_aux)
                weights[i,] = self.class_weights[np.argmax(self.labels[ID])]

            else:
                X[i,] = self._val_preprocessing(X_aux)

        if self.phase=='train':
            return X, y, weights
        else:
            return X, y

    def __len__(self):
      '''Denotes the number of batches per epoch'''
      return int(np.floor(len(self.list_IDs)/self.batch_size))

    def __getitem__(self, index):
      '''Generate one batch of data'''
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]

      return self.__data_generation(list_IDs_temp)

    def _read_image_folder(self, folderpath):
        """folderpath: path to a folder containing the CT images of a patient"""

        ct_names = os.listdir(folderpath)
        #ct_names = [file_name.zfill(10) for file_name in ct_names]
        ct_names.sort()

        matrix = []

        for filename in ct_names:
            filepath = os.path.join(folderpath, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print('Wrong path:', filepath)
            else:
                img = cv2.resize(img, dsize=(self.side_size,self.side_size), interpolation=cv2.INTER_AREA)
                matrix.append(img)

        matrix = np.array(matrix)

        if self.augmentation:
            scale = np.random.uniform(0.0, 20.0)
            sigma = np.random.uniform(0.0, 2.0)
            rotate = np.random.uniform(-15.0, 15.0)

            seq = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0.0, scale=scale)),
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=sigma)),
                iaa.Rotate(rotate=rotate)
            ], random_order=True)

            matrix = seq(images=matrix)

        matrix = np.transpose(matrix, [1,2,0])

        # remove the first matrix (empty)
        return matrix

    def _normalize(self, matrix):
        return np.array(matrix/255.0, dtype=np.float32)

    def _process_scan(self, folderpath, desired_side=128, desired_depth=64):
        """Read and resize volume"""
        # Read scan
        volume = self._read_image_folder(folderpath)

        # Normalize to (0,1)
        volume = self._normalize(volume)

        return volume

    def _train_preprocessing(self, volume):
        """Process training data by rotating and adding a channel."""
        # Rotate volume
        volume = tf.expand_dims(volume, axis=3)
        return volume

    def _val_preprocessing(self, volume):
        """Process validation data by only adding a channel."""
        volume = tf.expand_dims(volume, axis=3)
        return volume

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.L2(l2=0.05),
		      kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation='relu',
		      kernel_regularizer=tf.keras.regularizers.L2(l2=0.05),
                      kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation='relu',
		      kernel_regularizer=tf.keras.regularizers.L2(l2=0.10),
                      kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.L2(l2=0.10),
                      kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=4, activation='softmax')(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name='3dcnn_severity')

    return model


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--side_size', type=int, default=224, help='Height and width of the image')
    parser.add_argument('--depth_size', type=int, default=64, help='Depth of the CT')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--train_dir', type=str, default='MIA_sev_224x224x64/train',
                        help='Directory to the train dataset')
    parser.add_argument('--val_dir', type=str, default='MIA_sev_224x224x64/val',
                        help='Directory to the validation dataset')
    parser.add_argument('--work_directory', type=str,
                        default='/home/lovelace/proj/proj882/givendra/MIA-challenge-2023/dataset/',
                        help='Work directory')
    parser.add_argument('--checkpoint_path', type=str, default='severity_6/cp.ckpt',
                        help='Path to checkpoint')
    parser.add_argument('--patience', type=int, default=200, help='Patience parameter of early stopping')
    parser.add_argument('--augmentation', type=bool, default=True, help='To use data augmentation on the training phase')

    args = parser.parse_args()

    side_size = args.side_size
    depth_size = args.depth_size
    batch_size = args.batch_size
    initial_learning_rate = args.initial_learning_rate
    num_epochs = args.num_epochs
    patience = args.patience
    augmentation = args.augmentation


    train_dir = args.train_dir
    val_dir = args.val_dir

    work_directory = args.work_directory
    n_classes = 4
    n_channels = 1

    checkpoint_path = args.checkpoint_path
    checkpoint_dir = os.path.dirname(checkpoint_path)

    print(f'side_size: {side_size}')
    print(f'depth_size: {depth_size}')
    print(f'batch_size: {batch_size}')
    print(f'initial_learning_rate: {initial_learning_rate}')
    print(f'num_epochs: {num_epochs}')
    print(f'patience: {patience}')
    print(f'checkpoint_path: {checkpoint_path}')
    print()

    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    print(os.getenv('TF_GPU_ALLOCATOR'))
    print()

    print(os.getcwd())
    os.chdir(work_directory)
    ''' @ /home/lovelace/proj/proj882/igorkion/testeMIA'''
    print(os.getcwd())
    print()

    names = os.listdir(train_dir)

    train_folder_1 = os.path.join(train_dir,'1')
    train_folder_2 = os.path.join(train_dir,'2')
    train_folder_3 = os.path.join(train_dir,'3')
    train_folder_4 = os.path.join(train_dir,'4')

    val_folder_1 = os.path.join(val_dir,'1')
    val_folder_2 = os.path.join(val_dir,'2')
    val_folder_3 = os.path.join(val_dir,'3')
    val_folder_4 = os.path.join(val_dir,'4')

    print(train_folder_1)
    print(train_folder_2)
    print(train_folder_3)
    print(train_folder_4)
    print()

    print(val_folder_1)
    print(val_folder_2)
    print(val_folder_3)
    print(val_folder_4)
    print()

    train_paths_1 = [os.path.join(train_folder_1,folderpath) for folderpath in os.listdir(train_folder_1)]
    train_paths_2 = [os.path.join(train_folder_2,folderpath) for folderpath in os.listdir(train_folder_2)]
    train_paths_3 = [os.path.join(train_folder_3,folderpath) for folderpath in os.listdir(train_folder_3)]
    train_paths_4 = [os.path.join(train_folder_4,folderpath) for folderpath in os.listdir(train_folder_4)]
    train_paths = train_paths_1 + train_paths_2 + train_paths_3 + train_paths_4

    val_paths_1 = [os.path.join(val_folder_1,folderpath) for folderpath in os.listdir(val_folder_1)]
    val_paths_2 = [os.path.join(val_folder_2,folderpath) for folderpath in os.listdir(val_folder_2)]
    val_paths_3 = [os.path.join(val_folder_3,folderpath) for folderpath in os.listdir(val_folder_3)]
    val_paths_4 = [os.path.join(val_folder_4,folderpath) for folderpath in os.listdir(val_folder_4)]
    val_paths = val_paths_1 + val_paths_2 + val_paths_3 + val_paths_4

    print(f'Shape of train_paths_1: {len(train_paths_1)}')
    print(f'Shape of train_paths_2: {len(train_paths_2)}')
    print(f'Shape of train_paths_3: {len(train_paths_3)}')
    print(f'Shape of train_paths_4: {len(train_paths_4)}')
    print(f'Shape of train_paths: {len(train_paths)}')

    print(f'Shape of val_paths_1: {len(val_paths_1)}')
    print(f'Shape of val_paths_2: {len(val_paths_2)}')
    print(f'Shape of val_paths_3: {len(val_paths_3)}')
    print(f'Shape of val_paths_4: {len(val_paths_4)}')
    print(f'Shape of val_paths: {len(val_paths)}')

    print()

    partition = {}
    partition['train'] = train_paths
    partition['val'] = val_paths

    #print(partition)

    # Read and process the scans.
    # Each scan is resized across height, width, and depth and rescaled.

    # For the CT scans having presence of covid assign 1, for the non-covid ones assign 0.
    train_labels_1 = np.array([0 for _ in range(len(train_paths_1))])
    train_labels_2 = np.array([1 for _ in range(len(train_paths_2))])
    train_labels_3 = np.array([2 for _ in range(len(train_paths_3))])
    train_labels_4 = np.array([3 for _ in range(len(train_paths_4))])

    val_labels_1 = np.array([0 for _ in range(len(val_paths_1))])
    val_labels_2 = np.array([1 for _ in range(len(val_paths_2))])
    val_labels_3 = np.array([2 for _ in range(len(val_paths_3))])
    val_labels_4 = np.array([3 for _ in range(len(val_paths_4))])

    # Split data in the ratio 70-30 for training and validation.
    #x_train = np.concatenate((cov_train_scans, non_train_scans), axis=0)
    #y_train = np.concatenate((cov_train_labels, non_train_labels), axis=0)
    y_train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4), axis=0)
    y_train = np.zeros((y_train_labels.size, y_train_labels.max()+1))
    y_train[np.arange(y_train_labels.size),y_train_labels] = 1

    #x_val = np.concatenate((cov_val_scans, non_val_scans), axis=0)
    #y_val = np.concatenate((cov_val_labels, non_val_labels), axis=0)
    y_val_labels = np.concatenate((val_labels_1, val_labels_2, val_labels_3, val_labels_4), axis=0)
    y_val = np.zeros((y_val_labels.size, y_val_labels.max()+1))
    y_val[np.arange(y_val_labels.size),y_val_labels] = 1

    labels = {}

    for folderpath in train_paths_1:
        labels[folderpath] = [1.,0.,0.,0.]
    for folderpath in train_paths_2:
        labels[folderpath] = [0.,1.,0.,0.]
    for folderpath in train_paths_3:
        labels[folderpath] = [0.,0.,1.,0.]
    for folderpath in train_paths_4:
        labels[folderpath] = [0.,0.,0.,1.]

    for folderpath in val_paths_1:
        labels[folderpath] = [1.,0.,0.,0.]
    for folderpath in val_paths_2:
        labels[folderpath] = [0.,1.,0.,0.]
    for folderpath in val_paths_3:
        labels[folderpath] = [0.,0.,1.,0.]
    for folderpath in val_paths_4:
        labels[folderpath] = [0.,0.,0.,1.]

    print(labels)

    # class weights
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train_labels), y=y_train_labels)

    print(f'np.unique(y_train_labels): {np.unique(y_train_labels)}')
    print(f'y_train_labels: {y_train_labels}')
    print(f'np.unique(y_train): {np.unique(y_train)}')

    print(class_weights)
    class_weights = dict(zip(np.unique(y_train_labels), class_weights))

    print(class_weights)
    print()

    #exit()

    params_train = {'side_size': side_size,
                    'depth_size': depth_size,
                    'batch_size': batch_size,
                    'n_classes': n_classes,
                    'n_channels': n_channels,
                    'shuffle': True,
                    'phase': 'train',
                    'class_weights': class_weights,
                    'augmentation': augmentation}

    params_val = {'side_size': side_size,
                  'depth_size': depth_size,
                  'batch_size': batch_size,
                  'n_classes': n_classes,
                  'n_channels': n_channels,
                  'shuffle': False,
                  'phase': 'val',
                  'class_weights': class_weights,
                  'augmentation': False}

    train_generator = DataGenerator(partition['train'], labels, **params_train)
    val_generator = DataGenerator(partition['val'], labels, **params_val)

    print(len(train_generator))
    print(len(val_generator))
    print()

    # Build model.
    model = get_model(width=side_size, height=side_size, depth=depth_size)
    model.summary()

    # Compile model.
    model.compile(
        loss='categorical_crossentropy',
        #optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        #optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
        metrics=['accuracy','Precision','Recall',
                tfa.metrics.F1Score(num_classes=n_classes, average=None, name='f1'),
                tfa.metrics.F1Score(num_classes=n_classes, average='micro', name='micro_f1'),
                tfa.metrics.F1Score(num_classes=n_classes, average='macro', name='macro_f1')])

    # Define callbacks.
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(work_directory,checkpoint_path),
                    monitor='val_macro_f1', mode='max', save_weights_only=True,
                    verbose=1, save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_macro_f1',
                        min_delta=1e-4, patience=patience, mode='max', verbose=1)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_macro_f1',
                factor=0.8, patience=50, min_lr=1e-8, mode='max', verbose=1)

    # Train the model, doing validation at the end of each epoch
    hist = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=num_epochs,
        #use_multiprocessing=True,
        #workers=6
        verbose=2,
        #class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr]
    )

    model.load_weights(os.path.join(work_directory,checkpoint_path))

    #### validation after training  ####
    params_val_2 = {'side_size': side_size,
                    'depth_size': depth_size,
                    'batch_size': 1,
                    'n_classes': n_classes,
                    'n_channels': n_channels,
                    'shuffle': False,
                    'phase': 'val',
                    'class_weights': class_weights}

    val_generator_2 = DataGenerator(partition['val'], labels, **params_val_2)

    score = model.evaluate(val_generator_2, verbose=2)

    print(f'Validation loss:      {score[0]:.4f}')
    print(f'Validation accuracy:  {score[1]:.4f}')
    print(f'Validation precision: {score[2]:.4f}')
    print(f'Validation recall:    {score[3]:.4f}')
    print(f'Validation F1 (1):    {score[4][0]:.4f}')
    print(f'Validation F1 (2):    {score[4][1]:.4f}')
    print(f'Validation F1 (3):    {score[4][1]:.4f}')
    print(f'Validation F1 (4):    {score[4][1]:.4f}')
    print(f'Validation micro-F1:  {score[5]:.4f}')
    print(f'Validation macro-F1:  {score[6]:.4f}')
    print()

    print(score)
    print()

    preds = model.predict(val_generator_2, verbose=2)
    y_val_preds = np.argmax(preds, axis=1)
    print(y_val_labels)
    print(y_val_preds)

    conf_mat = tf.math.confusion_matrix(
        y_val_labels,
        y_val_preds,
        num_classes=n_classes,
        dtype=tf.dtypes.int32
    )
    print(conf_mat)
    print()

    accuracy_score = sklearn.metrics.accuracy_score(y_val_labels, y_val_preds)
    precision_score = sklearn.metrics.precision_score(y_val_labels, y_val_preds, average='micro')
    recall_score = sklearn.metrics.recall_score(y_val_labels, y_val_preds, average='micro')
    f1_score_micro = sklearn.metrics.f1_score(y_val_labels, y_val_preds, average='micro')
    f1_score_macro = sklearn.metrics.f1_score(y_val_labels, y_val_preds, average='macro')

    print('Sklearn metrics:')
    print(f'Validation accuracy: {accuracy_score:.4f}')
    print(f'Validation precision: {precision_score:.4f}')
    print(f'Validation recall: {recall_score:.4f}')
    print(f'Validation micro-F1: {f1_score_micro:.4f}')
    print(f'Validation macro-F1: {f1_score_macro:.4f}')
    print()

    conf_mat_sk = sklearn.metrics.confusion_matrix(y_val_labels, y_val_preds)
    print(conf_mat_sk)
    print()

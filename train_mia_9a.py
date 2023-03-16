import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import sklearn
import sklearn.metrics
import argparse
import cv2

import imageio
import imgaug as ia
from imgaug import augmenters as iaa


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self, list_IDs, labels, batch_size=10,
            side_size=128, depth_size=64, n_channels=1,
            n_classes=2, shuffle=True, phase='train', augmentation=False,
            class_weights=None   
        ):

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
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle==True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.side_size, self.side_size, self.depth_size, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.n_classes), dtype=np.int32)

        if self.phase=='train':
            weights = np.empty((self.batch_size), dtype=np.float32)

        for i, ID in enumerate(list_IDs_temp):
            y[i,] = self.labels[ID]
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
      return int(np.floor(len(self.list_IDs)/self.batch_size))

    def __getitem__(self, index):
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
      list_IDs_temp = [self.list_IDs[k] for k in indexes]

      return self.__data_generation(list_IDs_temp)

    def _read_image_folder(self, folderpath):
        ct_names = os.listdir(folderpath)
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
            rotate = np.random.uniform(-30.0, 30.0)
            flip_lr = np.random.choice([0.,1.])
            flip_ud = np.random.choice([0.,1.])

            seq = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0.0, scale=scale)),
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=sigma)),
                iaa.Rotate(rotate=rotate), 
                iaa.Fliplr(flip_lr),
                iaa.Flipud(flip_ud),
                iaa.Cutout(nb_iterations=(0,4), size=0.2, squared=False),
                iaa.Sometimes(0.5, iaa.GammaContrast((0.5, 2.0)))
            ], random_order=True)

            matrix = seq(images=matrix)

        matrix = np.transpose(matrix, [1,2,0])

        return matrix

    def _normalize(self, matrix):
        return np.array(matrix/255.0, dtype=np.float32)

    def _process_scan(self, folderpath, desired_side=128, desired_depth=64):
        volume = self._read_image_folder(folderpath)
        volume = self._normalize(volume)

        return volume

    def _train_preprocessing(self, volume):
        volume = tf.expand_dims(volume, axis=3)
        return volume

    def _val_preprocessing(self, volume):
        volume = tf.expand_dims(volume, axis=3)
        return volume


def get_model(width=128, height=128, depth=64):

    inputs = tf.keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(),
                      bias_regularizer=tf.keras.regularizers.L2(),
		              kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, input_shape=x.shape)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation='relu', padding='same',
		              kernel_regularizer=tf.keras.regularizers.L2(),
                      bias_regularizer=tf.keras.regularizers.L2(),
                      kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, input_shape=x.shape)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation='relu', padding='same',
		              kernel_regularizer=tf.keras.regularizers.L2(l2=0.05),
                      bias_regularizer=tf.keras.regularizers.L2(),
                      kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, input_shape=x.shape)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(l2=0.05),
                      bias_regularizer=tf.keras.regularizers.L2(),
                      kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, input_shape=x.shape)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(l2=0.05),
                      bias_regularizer=tf.keras.regularizers.L2(),
                      kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, input_shape=x.shape)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.L2(l2=0.05),
                      bias_regularizer=tf.keras.regularizers.L2(),
                      kernel_initializer=tf.keras.initializers.HeNormal())(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, input_shape=x.shape)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=2, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='3dcnn')

    return model


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--side_size', type=int, default=224, help='Height and width of the image')
    parser.add_argument('--depth_size', type=int, default=64, help='Depth of the CT')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--train_dir', type=str, default='MIA2023_224x224x64/train',
                        help='Directory to the train dataset')
    parser.add_argument('--val_dir', type=str, default='MIA2023_224x224x64/val',
                        help='Directory to the validation dataset')
    parser.add_argument('--work_directory', type=str,
                        default='/home/lovelace/proj/proj882/igorkion/testeMIA2023',
                        help='Work directory')
    parser.add_argument('--training_id', type=str, default='training2023_9a',
                        help='Training id')
    parser.add_argument('--patience', type=int, default=80, help='Patience parameter of early stopping')
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
    n_classes = 2
    n_channels = 1

    training_id = args.training_id
    checkpoint_path = args.training_id + '/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    print(os.getenv('TF_GPU_ALLOCATOR'))

    os.chdir(work_directory)

    names = os.listdir(train_dir)

    cov_train_folder = os.path.join(train_dir,'covid')
    non_train_folder = os.path.join(train_dir,'non-covid')
    cov_val_folder = os.path.join(val_dir,'covid')
    non_val_folder = os.path.join(val_dir,'non-covid')

    cov_train_paths = [os.path.join(cov_train_folder,folderpath) for folderpath in os.listdir(cov_train_folder)]
    non_train_paths = [os.path.join(non_train_folder,folderpath) for folderpath in os.listdir(non_train_folder)]
    train_paths = cov_train_paths + non_train_paths

    cov_val_paths = [os.path.join(cov_val_folder,folderpath) for folderpath in os.listdir(cov_val_folder)]
    non_val_paths = [os.path.join(non_val_folder,folderpath) for folderpath in os.listdir(non_val_folder)]
    val_paths = cov_val_paths + non_val_paths

    partition = {}
    partition['train'] = train_paths
    partition['val'] = val_paths


    cov_train_labels = np.array([1 for _ in range(len(cov_train_paths))])
    non_train_labels = np.array([0 for _ in range(len(non_train_paths))])

    cov_val_labels = np.array([1 for _ in range(len(cov_val_paths))])
    non_val_labels = np.array([0 for _ in range(len(non_val_paths))])

    y_train_labels = np.concatenate((cov_train_labels, non_train_labels), axis=0)
    y_train = np.zeros((y_train_labels.size, y_train_labels.max()+1))
    y_train[np.arange(y_train_labels.size),y_train_labels] = 1

    y_val_labels = np.concatenate((cov_val_labels, non_val_labels), axis=0)
    y_val = np.zeros((y_val_labels.size, y_val_labels.max()+1))
    y_val[np.arange(y_val_labels.size),y_val_labels] = 1

    labels = {}
    for folderpath in cov_train_paths:
        labels[folderpath] = [0.,1.]
    for folderpath in non_train_paths:
        labels[folderpath] = [1.,0.]
    for folderpath in cov_val_paths:
        labels[folderpath] = [0.,1.]
    for folderpath in non_val_paths:
        labels[folderpath] = [1.,0.]

    class_weights = sklearn.utils.class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train_labels), y=y_train_labels)

    class_weights = dict(zip(np.unique(y_train), class_weights))

    params_train = {'side_size': side_size,
                    'depth_size': depth_size,
                    'batch_size': batch_size,
                    'n_classes': n_classes,
                    'n_channels': n_channels,
                    'shuffle': True,
                    'phase': 'train',
                    'class_weights': class_weights,
                    'augmentation': True}

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

    model = get_model(width=side_size, height=side_size, depth=depth_size)
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tfa.optimizers.RectifiedAdam(learning_rate=initial_learning_rate),
        metrics=['accuracy','Precision','Recall',
                 tfa.metrics.F1Score(num_classes=2, average='micro', name='micro_f1'),
                 tfa.metrics.F1Score(num_classes=2, average='macro', name='macro_f1')]
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(work_directory,checkpoint_path),
                    monitor='val_macro_f1', mode='max', save_weights_only=True,
                    verbose=1, save_best_only=True)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_macro_f1',
                        min_delta=1e-4, patience=patience, mode='max', verbose=1)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_macro_f1',
                factor=0.5, patience=20, min_lr=1e-8, mode='max', verbose=1)

    log_dir = work_directory+'/logs/'+training_id
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    hist = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=num_epochs,
        verbose=2,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr, tensorboard_callback]
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
                    'class_weights': class_weights,
                    'augmentation': False}

    val_generator_2 = DataGenerator(partition['val'], labels, **params_val_2)

    score = model.evaluate(val_generator_2, verbose=2)

    print(f'Validation loss:      {score[0]:.4f}')
    print(f'Validation accuracy:  {score[1]:.4f}')
    print(f'Validation precision: {score[2]:.4f}')
    print(f'Validation recall:    {score[3]:.4f}')
    print(f'Validation micro-F1:  {score[4]:.4f}')
    print(f'Validation macro-F1:  {score[5]:.4f}')
    print()

    preds = model.predict(val_generator_2, verbose=2)
    y_val_preds = np.argmax(preds, axis=1)
    print(y_val_labels)
    print(y_val_preds)

    conf_mat = tf.math.confusion_matrix(
        y_val_labels,
        y_val_preds,
        num_classes=2,
        dtype=tf.dtypes.int32
    )
    print(conf_mat)
    print()


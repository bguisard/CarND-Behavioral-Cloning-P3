import csv
import cv2
import numpy as np
import sklearn
import pandas as pd

from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Cropping2D, Convolution2D
from keras.layers import BatchNormalization, Input, Lambda, Dropout, Activation
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger

from sklearn.model_selection import train_test_split
from random import shuffle
from skimage import transform


# Define path for input data
path = './data/data/'

# We now import the data from the csv file - OBSOLETE
samples = []
with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Import data and remove most straight line driving using pandas:
data = pd.read_csv(path + 'driving_log.csv',
                   dtype={'center_img': np.str, 'left_img': np.str,
                          'rignt_img': np.str, 'steering': np.float64,
                          'throttle': np.float64, 'brake': np.float64,
                          'speed': np.float64}, header=None)

# We will remove 90% of the straight line driving:
remove_n = np.round(data[data[:][3] == 0][3].count() * 0.70, 0).astype(np.uint64)
drop_indices = np.random.choice(data[data[:][3] == 0].index, remove_n, replace=False)
data_subset = data.drop(drop_indices)

# And also remove 70% of the data between -0.25 and +0.25 degrees
f1 = data[:][3] > -0.25
f2 = data[:][3] < 0.25
remove_n = np.round(data_subset[f1 & f2][3].count() * 0.70, 0).astype(np.uint64)
drop_indices = np.random.choice(data_subset[f1 & f2].index, remove_n, replace=False)
data_subset2 = data_subset.drop(drop_indices)


# And split into training and validation, keeping 20% for validation
trn_samples, val_samples = train_test_split(samples, test_size=0.2)
# trn_samples, val_samples = train_test_split(data_subset.values, test_size=0.2)
# trn_samples, val_samples = train_test_split(data_subset2.values, test_size=0.2)


# We will first define the augmentation functions that we will use in
# our generator.


def flip_image(img, angle):
    img = cv2.flip(img, 1)
    angle *= -1.0

    return img, angle


def translate_image(img, angle, trans_limit=60, trans_adj=0.2):

    # LATERAL TRANSLATION (simulates car being in different positions on track - like using multiple cameras)
    trans_x = np.random.uniform(-trans_limit, trans_limit)

    # the lesson suggested that we should add or subtract 0.2 to the steering angle for the left and right
    # images. They are translated by about 60 pixels, giving us an estimate of 0.004 radians per pixel.
    angle += trans_x * trans_adj / 60.

    # VERTICAL TRANSLATION (simulates car going up or down slope, no need to adjust steering angle)
    # I reduced the translation limit to reflect the fact that lateral shifts are wider than vertical shifts.
    trans_y = np.random.uniform(-trans_limit / 2, trans_limit / 2)

    # Now we combine both transformations into one transformation matrix
    tform = transform.AffineTransform(translation=(trans_x, trans_y))

    img = transform.warp(img, tform, mode='constant')
    img = np.array(img * 255).astype(np.uint8)

    return img, angle


def disturb_brightness(img, strength=0.50):

    # will create a random brightness factor between [strenght, 1+strenght] to apply to the brightness channel
    rnd_brightness = strength + np.random.uniform()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img).astype(np.float64)
    img[:, :, 2] = np.clip(img[:, :, 2] * rnd_brightness, 0, 255)
    img = np.array(img).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img


def random_shadow(img, strength=0.50):
    """
    Random shawdow augmentation implementation as suggested by:
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    """

    rows, cols, _ = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    x_lo, x_hi = 0, rows

    rand_y = (cols * np.random.uniform(), cols * np.random.uniform())
    y_lo, y_hi = np.min(rand_y), np.max(rand_y)

    shadow_mask = 0 * img[:, :, 1]
    X_msk = np.mgrid[0:rows, 0:cols][0]
    Y_msk = np.mgrid[0:rows, 0:cols][1]
    shadow_mask[((X_msk - x_lo) * (y_lo - y_hi) - (x_hi - x_lo) * (Y_msk - y_hi) >= 0)] = 1

    if np.random.randint(2) == 1:
        random_bright = strength
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            img[:, :, 1][cond1] = img[:, :, 1][cond1] * random_bright
        else:
            img[:, :, 1][cond0] = img[:, :, 1][cond0] * random_bright
    img = np.array(img).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)

    return img


# And now the generator
def generator(samples, batch_size=32, use_side_cam=False, aug=False, adj=0.20):
    """
    This generator receives a list of image filenames and steering angles
    and shuffles the data to feed the model.

    Input:
    samples - list from csv file with image filenames and steering angles.

    Arguments:
    batch_size - size of the mini batch
    aug - Data Augmentation flag, if set to True we randomly transform the image.
    adj - steering correction value when using images from the side cameras.

    Output:
    X_train and y_train
    """
    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if use_side_cam:
                    img_idx = np.random.choice([0, 1, 2])
                else:
                    img_idx = 0

                if img_idx == 0:
                    adjustment = 0
                if img_idx == 1:
                    adjustment = adj
                if img_idx == 2:
                    adjustment = adj * -1.0

                name = path + 'IMG/' + batch_sample[img_idx].split('/')[-1]
                image = cv2.imread(name)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[3]) + adjustment

                if aug:
                    p = np.random.uniform(0., 1.)
                    if p < 0.66:
                        image = random_shadow(image)

                    p = np.random.uniform(0., 1.)
                    if p < 0.66:
                        image = disturb_brightness(image)

                    p = np.random.uniform(0., 1.)
                    if p < 0.66:
                        image, angle = translate_image(image, angle, trans_adj=adj)

                    p = np.random.uniform(0., 1.)
                    if p < 0.50:
                        image, angle = flip_image(image, angle)

                # image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


# We now define the parameters we are going to use for our generator:
aug = True
side = True
reg = 1e-3
lr = 1e-3
angle_adj = 0.20
optimizer = Adam(lr=lr)
resume_training = False


# And assign the generator to the training samples and validation samples.
trn_generator = generator(trn_samples, batch_size=128, aug=aug, use_side_cam=side, adj=angle_adj)
val_generator = generator(val_samples, batch_size=128)

# This piece of code accounts for the size of the training dataset depending
# on how many cameras we are using and if we are using data augmentation.
if aug:
    sample_epoch = len(trn_samples) * 5
else:
    sample_epoch = len(trn_samples)


def get_NVIDIA(input_shape, crop=False):
    """
    This function will return a convolutional neural network as described on
    "End to End Learning for Self-Driving Cars" by NVIDIA
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

    I made a small change to the network and added BatchNormalization to help
    the model train faster and avoid overfitting.

    """
    i_rows, i_cols, i_channels = input_shape

    inputs = Input(((i_rows, i_cols, i_channels)))
    if crop:
        x = Cropping2D(cropping=((30, 25), (0, 0)))(inputs)
        x = BatchNormalization()(x)
    else:
        x = BatchNormalization()(inputs)
    x = Convolution2D(24, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(48, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='valid')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='valid')(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu', W_regularizer=l2(reg))(x)
    x = Dense(50, activation='relu', W_regularizer=l2(reg))(x)
    x = Dense(10, activation='relu', W_regularizer=l2(reg))(x)
    output = Dense(1)(x)

    model = Model(input=inputs, output=output)

    return model


def get_commaAI(input_shape, crop=False):

    i_rows, i_cols, i_channels = input_shape

    inputs = Input(((i_rows, i_cols, i_channels)))

    x = Lambda(lambda x: x / 127.5 - 1.)(inputs)

    if crop:
        x = Cropping2D(cropping=((30, 25), (0, 0)))(x)

    x = Convolution2D(16, 8, 8, activation='elu', border_mode='same', subsample=(4, 4))(x)
    x = Convolution2D(32, 5, 5, activation='elu', border_mode='same', subsample=(2, 2))(x)
    x = Convolution2D(64, 5, 5, activation='elu', border_mode='same', subsample=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(.2)(x)
    x = Activation('elu')(x)
    x = Dense(512, activation='elu')(x)
    x = Dropout(.5)(x)
    output = Dense(1)(x)

    model = Model(input=inputs, output=output)

    return model


model = get_NVIDIA(input_shape=(160, 320, 3), crop=True)
# model = get_NVIDIA(input_shape=(64, 64, 3), crop=False)
# model = get_commaAI(input_shape=(64, 64, 3), crop=False)

# I trained the model with the default learning rate for Adam and it behaved
# so well that I didn't feel I had much to gain by changing the initial Learning
# rate.
model.compile(optimizer=optimizer, loss='mse')

checkpoint = ModelCheckpoint('./checkpoints/model_{epoch:02d}.h5')
csv_logger = CSVLogger('./checkpoints/train_log.csv')

if resume_training:
    model = load_model('model_temp.h5')
    model.compile(optimizer=optimizer, loss='mse')
    model.optimizer.lr = 1e-4

model.fit_generator(trn_generator,
                    nb_epoch=10,
                    samples_per_epoch=sample_epoch,
                    validation_data=val_generator,
                    nb_val_samples=len(val_samples),
                    callbacks=[checkpoint, csv_logger],
                    verbose=1)

model.save('model_new.h5')

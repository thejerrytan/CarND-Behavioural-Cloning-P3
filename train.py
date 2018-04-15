import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

TRAINING_DIR = "harder_data"
samples = []
def load_data():
    with open('./{TRAINING_DIR!s}/driving_log.csv'.format(**globals())) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    images = []
    measurements = []
    return train_test_split(samples, test_size=0.2)

def visualize_data():
    # Shows the distribution of steering angles among the full unaugmented dataset
    angles = []
    for line in samples:
        angles.append(float(line[3]))
        angles.append(float(line[3]) + CORRECTION_FACTOR)
        angles.append(float(line[3]) - CORRECTION_FACTOR)
    hist, bin_edges = np.histogram(angles, bins=32)
    plt.hist(angles, bins=32)
    plt.show()
    return hist, bin_edges

def random_shift(img, angle, shift_range):
    STEERING_ANGLE_SHIFT_PER_PIXEL = 0.002
    dx = np.random.uniform(-shift_range, shift_range, 1)
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return (dst, STEERING_ANGLE_SHIFT_PER_PIXEL * dx + angle)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    new_batch_size = batch_size // 9
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, new_batch_size): # Because each loop generates x6 more images
            batch_samples = samples[offset:offset+new_batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3])
                preprocess_data('./{TRAINING_DIR!s}/IMG/'.format(**globals()) + batch_sample[0], measurement, images, angles)
                # Add left camera
                measurement_left = measurement + CORRECTION_FACTOR
                preprocess_data('./{TRAINING_DIR!s}/IMG/'.format(**globals()) + batch_sample[1], measurement_left, images, angles)
                # Add right camera
                measurement_right = measurement - CORRECTION_FACTOR
                preprocess_data('./{TRAINING_DIR!s}/IMG/'.format(**globals()) + batch_sample[2], measurement_right, images, angles)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Data augmentation
def preprocess_data(source_path, measurement, images, angles):
    image = cv2.imread('./{TRAINING_DIR!s}/IMG/'.format(**globals()) + source_path.split(os.sep)[-1])
    b,g,r = cv2.split(image)       # get b,g,r
    image = cv2.merge([r,g,b])     # switch it to rgb
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) # resize image by half
    
    images.append(image)
    angles.append(measurement)
    
    # Flip horizontally
    measurement_flipped = -measurement
    flipped = np.fliplr(image)
    images.append(flipped)
    angles.append(measurement_flipped)

    # Translate horizontally
    translated, new_steering_angle = random_shift(image, measurement, 10)
    images.append(translated)
    angles.append(new_steering_angle)
    

CORRECTION_FACTOR = 0.20

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Cropping2D
from keras.models import load_model


def main():
    train_samples, validation_samples = load_data()

    visualize_data()

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # Compile and train model
    # if os.path.isfile("model_hard.h5"):
    if False:
        model = load_model('model_hard.h5')
    else:
        model = Sequential()
        model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(80,160,3)))
        model.add(Cropping2D(cropping=((35,13), (0,0))))
        model.add(Convolution2D(6,5,5, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Convolution2D(6,5,5, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))
        model.add(Flatten(input_shape=(37,160,3)))
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))

    adam = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=adam)
    model.fit_generator(train_generator, nb_epoch=20, samples_per_epoch=len(train_samples), validation_data=validation_generator, \
                        nb_val_samples=len(validation_samples))

    model.save('model_hard.h5')

if __name__ == "__main__":
    main()
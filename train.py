import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split


samples = []
with open('./more_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

images = []
measurements = []
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size//6): # Because each loop generates x6 more images
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
            	measurement = float(batch_sample[3])
            	preprocess_data('./more_data/IMG/' + batch_sample[0], measurement, images, angles)
            	# Add left camera
            	measurement_left = measurement + CORRECTION_FACTOR
            	preprocess_data('./more_data/IMG/' + batch_sample[1], measurement_left, images, angles)
            	# Add right camera
            	measurement_right = measurement - CORRECTION_FACTOR
            	preprocess_data('./more_data/IMG/' + batch_sample[2], measurement_right, images, angles)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def preprocess_data(source_path, measurement, images, angles):
	image = cv2.imread('./more_data/IMG/' + source_path.split(os.sep)[-1])
	b,g,r = cv2.split(image)       # get b,g,r
	image = cv2.merge([r,g,b])     # switch it to rgb
	image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) # resize image by half
	flipped = np.fliplr(image)
	images.append(image)
	images.append(flipped)
	measurement_flipped = -measurement
	angles.append(measurement)
	angles.append(measurement_flipped)

CORRECTION_FACTOR = 0.20

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.models import load_model


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

if os.path.isfile("model.h5"):
    model = load_model('model.h5')
else:
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(80,160,3)))
    model.add(Cropping2D(cropping=((35,13), (0,0))))
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten(input_shape=(37,160,3)))
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, nb_epoch=5, samples_per_epoch=len(train_samples), validation_data=validation_generator, \
					nb_val_samples=len(validation_samples))

model.save('model.h5')
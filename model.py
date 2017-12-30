import cv2
import csv
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout

path = './data/'  # fill in the path to your training IMG directory

#-----------------------------------------------------------------------------------------

 # READ CSV and LOAD DATA

car_images = []
steering_angles = []
with open(path + 'driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        steering_center = float(line[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2  # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        img_center = line[0]
        img_left = line[1]
        img_right = line[2]

        # add images and angles to data set
        car_images.append(img_center)
        car_images.append(img_left)
        car_images.append(img_right)
        steering_angles.append(steering_center)
        steering_angles.append(steering_left)
        steering_angles.append(steering_right)

#-----------------------------------------------------------------------------------------

def CNNModel():
    # Modified Nvidia Model
    # Difference to the Nvidia Model is the cropping and the dropout layers
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
    # normalize data
    model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160,320,3)))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#-----------------------------------------------------------------------------------------

def generatorData(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping image, correcting measurement and adding that measuerement
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

#-----------------------------------------------------------------------------------------
# Splitting into train and valdiation data

print('Total Images: {}'.format( len(car_images)))

# Splitting samples and creating generators.
samples = list(zip(car_images, steering_angles))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

#-----------------------------------------------------------------------------------------

training_generator = generatorData(train_samples, batch_size=32)
validation_generator = generatorData(validation_samples, batch_size=32)

print('Training model...')

model = CNNModel()
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(training_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=15, verbose=1)

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

#-----------------------------------------------------------------------------------------

print('Saving model...')

model.save("model.h5")

with open("model.json", "w") as json_file:
  json_file.write(model.to_json())

print("Model Saved.")

import sys  # to access the system
from os import listdir
from os.path import isfile, join, isdir
import tensorflow as tf
import numpy as np
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui


class Model:
    def __init__(self):
        self.test_images_folder_path = "dataset/manual_test"
        # Preprocessing dataset
        self.train_data_gen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        self.training_set = self.train_data_gen.flow_from_directory(
            'dataset/training_set',
            target_size=(64, 64),
            batch_size=24,
            class_mode='categorical'
        )
        self.test_set = self.train_data_gen.flow_from_directory(
            'dataset/test_set',
            target_size=(64, 64),
            batch_size=24,
            class_mode='categorical'
        )
        self.classes = {}
        for key, value in self.training_set.class_indices.items():
            self.classes[value] = (key[:-1] if key != "buses" else key[:-2]).capitalize()
        if not isdir("saved_models/model") or not listdir("saved_models/model"):
            self.model = self.construct_cnn()
        else:
            self.model = self.load_model()

        # last saved model evaluation 72% ___ 25 epochs
        self.model.evaluate(self.test_set)

    def construct_cnn(self):
        cnn = tf.keras.models.Sequential()
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=1))
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=1))
        cnn.add(tf.keras.layers.Flatten())
        cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
        cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))
        cnn.compile(optimizer="adam",
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        cnn.fit(x=self.training_set, validation_data=self.test_set, epochs=20)
        cnn.save('saved_models/model')

        return cnn

    def load_model(self):
        return tf.keras.models.load_model("saved_models/model")

    def predict_single(self, img_name):
        img_path = self.test_images_folder_path + "/" + img_name

        y_result = {0: 0, 1: 0, 2: 0, 3: 0}
        weird_number = 0

        test_image = utils.load_img(img_path, target_size=(64, 64))
        test_image = utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        predictions = (tf.nn.softmax(self.model.predict(test_image)[0])).numpy()
        result = self.model.predict(test_image)[0]
        itemindex = np.where(result == 1)

        print("--- ---")
        print("Image name: ", img_name)
        print(predictions)
        if len(itemindex[0]):
            # y_result[itemindex[0][0]] += 1
            print("Guess: ", self.classes[itemindex[0][0]])
        else:
            weird_number += 1
        for index, prob in enumerate(result):
            print(self.classes[index], ": ", prob)

        return predictions

    def predict_multiple(self):
        test_images = listdir(self.test_images_folder_path)

        # y_result = {0: 0, 1: 0, 2: 0, 3: 0}
        # weird_number = 0
        for img in test_images:
            self.predict_single(img)

        # print(y_result)
        # print("WN: ", weird_number)

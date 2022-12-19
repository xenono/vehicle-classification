import sys  # to access the system
import cv2
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Preprocessing dataset
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1. / 255)
test_set = train_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Convolution Neural Network
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
# path = "/home/xenono/AI/AI_Coursework/dataset/cars/"
# images = [f for f in listdir(path)[:10] if isfile(join(path, f))]
#
# for img in images:
#     img = cv2.imread(path + "/" + img, cv2.IMREAD_ANYCOLOR)
#     cv2.imshow("Sheep", img)
#     cv2.waitKey(0)
# # sys.exit()
# cv2.destroyAllWindows()  # destroy all windows
# while True:
#     cv2.imshow("Sheep", img)
#     cv2.waitKey(0)
#     sys.exit() # to exit from all the processes

# cv2.destroyAllWindows() # destroy all windows

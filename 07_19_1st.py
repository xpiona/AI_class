import keras
import matplotlib.pyplot as plt
print(keras.__version__)

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
a = train_images[2]
plt.imshow(a, cmap=plt.cm.binary)
plt.show()

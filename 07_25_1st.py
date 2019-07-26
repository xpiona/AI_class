from keras.models import Sequential
from keras import layers
from keras import models
from keras.layers import Flatten, Dense, Embedding
from keras import optimizers
from keras.datasets import imdb
from keras import preprocessing
import numpy as np
import io
import sys
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import cv2

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


!curl -L https://www.dropbox.com/s/v4225z25x0zz4y3/pool.jpg \
  -o pool.jpg
img = cv2.imread('pool.jpg')
numpy.ndarray
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
img2 = 255 -img
plt.
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img3, cmap='gray')

[i for i in dir(cv2) if i.startswith('COLOR_')]

cv2.imwrite('pool_2.jpg', img)
plt.subplot(1,2,1) # 1행 2열 중 첫번째
plt.axis('off')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

img2 = cv2.flip(img,1)
plt.subplot(1,2,2) # 1행 2열 중 2번째
plt.axis('off')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

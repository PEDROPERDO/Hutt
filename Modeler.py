import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomTranslation, RandomZoom

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

class Helper(Model):
  def __init__(self):
    super(Helper, self).__init__()
    self.flip_left_right = RandomFlip()
    self.random_rotation = RandomRotation(0.4)
    self.random_translation = RandomTranslation(0.2, 0.2)
    self.random_size = RandomZoom(0.4)
  
  def call(self, i):
    x = self.flip_left_right(i)
    x = self.random_rotation(x)
    x = self.random_translation(x)
    x = self.random_size(x)
    return x

class ModelFile(Model):
  def __init__(self):
    super(Hantu, self).__init__()
    self.imasatu = Conv2D(16, 3, activation="relu", padding="same")
    self.poolone = MaxPool2D((2, 2))
    self.imanida = Conv2D(32, 3, activation="relu", padding="same")
    self.pooltwo = MaxPool2D((2, 2))
    self.imatiga = Conv2D(64, 3, activation="relu", padding="same")
    self.pooliga = MaxPool2D((2, 2))
    self.imafour = Conv2D(128, 3, activation="relu", padding="same")
    self.pooling = MaxPool2D((2, 2))
    self.dropout = Dropout(0.3)
    self.flatten = Flatten()
    self.denseri = Dense(128, activation="relu")
    self.classes = Dense(3, activation="softmax")

  def call(self, i):
    x = self.poolone(self.imasatu(i))
    x = self.pooltwo(self.imanida(x))
    x = self.pooliga(self.imatiga(x))
    x = self.pooling(self.imafour(x))
    x = self.dropout(self.flatten(x))
    x = self.denseri(x)
    j = self.classes(x)
    return j

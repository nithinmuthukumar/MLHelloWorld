import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data() # 28*28 images of hand-written digits 0-9
x_train=tf.keras.utils.normalize(x_train,axis=1)#sigmoid function
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
model = tf.keras.models.Sequential()#Sequential only feeds forward
# the pixel arrays need to be flattened into 1d
model.add(tf.keras.layers.Flatten())

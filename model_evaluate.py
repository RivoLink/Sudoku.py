# import os for path
import os

# keras imports for the dataset
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')
X_test /= 255

n_classes = 10
Y_test = np_utils.to_categorical(y_test, n_classes)

save_dir = os.getcwd() + "/models/"
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)

mnist_model = load_model(model_path)
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])
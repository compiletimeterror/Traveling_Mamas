import tensorflow.keras as K
import tensorflow as tf
import numpy as np
# Train, Test = K.datasets.cifar10.load_data()
Train, Test = K.datasets.fashion_mnist.load_data()

(x_train, y_train) = Train
(x_test, y_test) = Test

from sklearn.model_selection import train_test_split
x_train, X_test, y_train, Y_test = train_test_split(x_train,y_train, test_size=0.75, random_state=42)
# x_train, X_test, y_train, Y_test = train_test_split(x_train,y_train, test_size=0.50, random_state=42)

def preprocess_data(X, Y):
    """
    * X is a numpy.ndarray of shape (m, 32, 32, 3) containing
        the CIFAR 10 data, where m is the number of data points
    * Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
        labels for X
        Returns: X_p, Y_p
    * X_p is a numpy.ndarray containing the preprocessed X
    * Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.resnet.preprocess_input(X)    
    Y_p = K.utils.to_categorical(y=Y, num_classes=10)
    return (X_p, Y_p)
    
    
   xp_train, yp_train = preprocess_data(x_train, y_train)
xp_test, yp_test = preprocess_data(x_test, y_test)

lay_init = K.initializers.he_normal()
# pre_model  
entrada = K.Input(shape=(28, 28, 1))
resize = K.layers.Lambda(lambda image: tf.image.resize(image, (155, 155)))(entrada)
resnet50 = K.applications.ResNet50(include_top=False, weights=None, input_tensor=resize)
# dense169.trainable = False
out_pre = resnet50(resize)
 
vector = K.layers.Flatten()(out_pre)
drop1 = K.layers.Dropout(0.3)(vector)
norm_lay1 = K.layers.BatchNormalization()(drop1)
FC1 = K.layers.Dense(units=510, activation='relu', kernel_initializer=lay_init)(norm_lay1)
norm_lay2 = K.layers.BatchNormalization()(FC1)
out = K.layers.Dense(units=10, activation='softmax', kernel_initializer=lay_init)(norm_lay2)
 
model = K.models.Model(inputs=entrada, outputs=out)
   
#learn_dec = K.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
early = K.callbacks.EarlyStopping(patience=5)
#save = K.callbacks.ModelCheckpoint(filepath='cifar10.h5', save_best_only=True, monitor='val_loss', mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=xp_train.reshape([-1,28, 28,1]), y=yp_train, batch_size=32, epochs=200, verbose=1, validation_data=(xp_test.reshape([-1,28, 28,1]), yp_test), callbacks=[early])

model.evaluate(xp_test.reshape([-1,28, 28,1]), yp_test, batch_size=32, verbose=1)

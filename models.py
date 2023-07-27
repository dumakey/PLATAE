import os
import numpy as np
import tensorflow as tf

def swish(x, beta=1):
    return (x * tf.keras.backend.sigmoid(beta * x))

        
def conv2D_block(X, num_channels, f, p, s, dropout, **kwargs):

    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'

    if p != 0:
        net = tf.keras.layers.ZeroPadding2D(p)(X)
    else:
        net = X
    net = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s,padding='valid',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    
    if activation == 'leakyrelu':
        rate = 0.1
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU()(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)

    return net

def dense_layer(X, units, activation, dropout, l1_reg, l2_reg):
    net = tf.keras.layers.Dense(units=units,activation=activation,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X)
    net = tf.keras.layers.BatchNormalization()(net)
    if activation == 'leakyrelu':
        rate = 0.1
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU()(net)
    else:
        net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)

    return net

def get_padding(f,s,nin,nout):

    padding = []
    for i in range(f.__len__()):
        p = int(np.floor(0.5 * ((nout-1)*s[i] + f[i] - nin)))
        nchout = int(np.floor((nin + 2*p - f[i])/s[i] + 1))
        if nchout != nout:
            padding.append(p+1)
        else:
            padding.append(p)

    return padding

def vehicle_detection_alexnet_model(image_shape, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    input_shape = tuple(image_shape.as_list() + [3])
    X_input = tf.keras.layers.Input(shape=input_shape)
    net = conv2D_block(X_input,num_channels=96,f=11,p=0,s=4,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=3,strides=2)(net)
    net = conv2D_block(net,num_channels=256,f=5,p=2,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=3,strides=2)(net)
    net = conv2D_block(net,num_channels=384,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = conv2D_block(net,num_channels=384,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = conv2D_block(net,num_channels=256,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=3,strides=2)(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = dense_layer(net,512,activation,dropout,l1_reg,l2_reg)
    net = tf.keras.layers.Dense(units=9,activation='softmax',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    model = tf.keras.Model(inputs=X_input,outputs=net,name='CNNScanner')
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics = [tf.keras.metrics.CategoricalAccuracy(),
                             ])

    return model

def vehicle_detection_cnn_model(image_shape, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    input_shape = tuple(image_shape.as_list() + [3])
    X_input = tf.keras.layers.Input(shape=input_shape)
    net = conv2D_block(X_input,num_channels=17,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = conv2D_block(net,num_channels=39,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = conv2D_block(net,num_channels=87,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = conv2D_block(net,num_channels=87,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = conv2D_block(net,num_channels=87,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Flatten()(net)
    net = dense_layer(net,512,activation,dropout,l1_reg,l2_reg)
    net = tf.keras.layers.Dense(units=9,activation='softmax',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    model = tf.keras.Model(inputs=X_input,outputs=net,name='CNNScanner')
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics = [tf.keras.metrics.CategoricalAccuracy()])

    return model

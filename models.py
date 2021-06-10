import math

import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, MaxPool2D, Conv2DTranspose, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import initializers 


def cnn(in_shape, n_classes, opt):
    inp = Input(shape= in_shape)
    fe = Conv2D(filters=32, kernel_size=(1,5), kernel_initializer='random_normal', bias_initializer='zeros')(inp)
    fe = Activation('relu')(fe)
    fe = Conv2D(filters=32, kernel_size=(1,5), kernel_initializer='random_normal', bias_initializer='zeros')(fe)
    fe = Activation('relu')(fe)
    fe = Conv2D(filters=32, kernel_size=(1,5), kernel_initializer='random_normal', bias_initializer='zeros')(fe)
    fe = Activation('relu')(fe)
    fe = Flatten()(fe)
    fe = Dense(128, activation='relu')(fe)
    fe = Dense(n_classes)(fe)
    #Classifer #, kernel_regularizer='l2'
    c_out_layer = Activation('softmax')(fe)
    c_model = Model(inp, c_out_layer)
    c_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return c_model

def semi_cnn(in_shape, n_classes, k, alpha, beta, DC_opt, C_opt):
    inp = Input(shape= in_shape)
    fe = Conv2D(filters=32, kernel_size=(1,5), kernel_initializer='random_normal', bias_initializer='zeros')(inp)
    fe = Activation('relu')(fe)
    fe = Conv2D(filters=32, kernel_size=(1,5), kernel_initializer='random_normal', bias_initializer='zeros')(fe)
    fe = Activation('relu')(fe)
    fe = Conv2D(filters=32, kernel_size=(1,5), kernel_initializer='random_normal', bias_initializer='zeros')(fe)
    fe = Activation('relu')(fe)
    fe = Flatten()(fe)
    #Deep Clustering
    #DC_fe = Dense(128, activation='relu')(fe)
    #DC_fe = Dense(k, kernel_regularizer='l2')(DC_fe)
    DC_fe = Dense(k, kernel_regularizer='l2')(fe)
    DC_out_layer = Activation('softmax')(DC_fe)
    DC_model = Model(inp, DC_out_layer)
    DC_model.compile(loss=dc_loss(beta = beta), optimizer=DC_opt, metrics=['accuracy'])
    #Classidier
    #C_fe = Dense(128, activation='relu')(fe)
    #C_fe = Dense(n_classes, kernel_regularizer='l2')(C_fe)
    C_fe = Dense(n_classes, kernel_regularizer='l2')(fe)  
    C_out_layer = Activation('softmax')(C_fe)
    C_model = Model(inp, C_out_layer)
    C_model.compile(loss=cls_loss(alpha = alpha), optimizer=C_opt, metrics=['accuracy'])

    return DC_model, C_model     

def dc_loss(beta):
    def loss(pseudo_y, pred_y):   
        return beta * K.categorical_crossentropy(pseudo_y, pred_y)
    return loss 

def cls_loss(alpha):
    def loss(y, pred_y):
        return alpha * K.categorical_crossentropy(y, pred_y)
    return loss


def mnistnet(in_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, input_shape=(1, 28, 28), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    opt= Adam(lr=0.0002, beta_1=0.5)
    inp = Input(shape= in_shape)
    fe = Conv2D(filters=32, kernel_size=3, kernel_initializer='random_normal', bias_initializer='zeros')(inp)
    fe = Activation('relu')(fe)
    fe = MaxPool2D(pool_size=2, data_format='channels_first')
    fe = Flatten()(fe)
    #Deep Clustering
    DC_fe = Dense(256, activation='relu')(fe)
    DC_fe = Dense(k, kernel_regularizer='l2')(DC_fe)
    DC_out_layer = Activation('softmax')(DC_fe)
    DC_model = Model(inp, DC_out_layer)
    DC_model.compile(loss=dc_loss(alpha = alpha), optimizer=opt, metrics=['accuracy'])
    #Classidier
    C_fe = Dense(256, activation='relu')(fe)
    C_fe = Dense(n_classes, kernel_regularizer='l2')(C_fe) 
    C_out_layer = Activation('softmax')(C_fe)
    C_model = Model(inp, C_out_layer)
    C_model.compile(loss=cls_loss(alpha = alpha), optimizer=opt, metrics=['accuracy'])

    return model

def semi_AE(in_shape, n_classes, z, alpha):
    opt= Adam(lr=0.0002, beta_1=0.5)
    #Encoder
    inp = Input(shape= in_shape)
    en = Conv2D(filters=32, kernel_size=(1,5), strides=1, kernel_initializer='random_normal', bias_initializer='zeros')(inp)
    en = Activation('relu')(en)
    en = Conv2D(filters=32, kernel_size=(1,5), strides=1, kernel_initializer='random_normal', bias_initializer='zeros')(en)
    en = Activation('relu')(en)
    en = Conv2D(filters=32, kernel_size=(1,5), strides=1, kernel_initializer='random_normal', bias_initializer='zeros')(en)
    en = Activation('relu')(en)
    en = Flatten()(en)
    #
    latent = Dense(z)(en)
    #Decoder
    fe = Dense(108 * z * 1, activation='relu')(latent)
    fe = Reshape((1, 108, z))(fe)
    fe = Conv2DTranspose(filters = 32, 
                         kernel_size=(1,5), 
                         strides=1, 
                         activation='relu',
                         kernel_initializer='random_normal', 
                         bias_initializer='zeros')(fe)
    fe = Conv2DTranspose(filters = 32, 
                         kernel_size=(1,5), 
                         strides=1, 
                         activation='relu',
                         kernel_initializer='random_normal', 
                         bias_initializer='zeros')(fe)
    fe = Conv2DTranspose(filters = 32, 
                         kernel_size=(1,5), 
                         strides=1, 
                         activation='relu',
                         kernel_initializer='random_normal', 
                         bias_initializer='zeros')(fe)
    fe = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(fe)
    #Autoencoder
    AE_model = Model(inp, fe)
    AE_model.compile(loss=ae_loss(alpha = alpha), optimizer=opt, metrics=['accuracy'])
    #Classidier
    C_fe = Dense(128, activation='relu')(latent)
    C_fe = Dense(n_classes, kernel_regularizer='l2')(C_fe) 
    C_out_layer = Activation('softmax')(C_fe)
    C_model = Model(inp, C_out_layer)
    C_model.compile(loss=cls_loss(alpha = alpha), optimizer=opt, metrics=['accuracy'])

    return AE_model, C_model    
def ae_loss(alpha): 
    def loss(x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = K.mean(K.square(x_decoded_mean - x))
        return (1- alpha) * xent_loss 
    return loss

def cls_loss(alpha):
    def loss(y, pred_y):
        return alpha * K.categorical_crossentropy(y, pred_y)
    return loss


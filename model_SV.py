
from keras.layers import Conv2D, MaxPooling3D, Dropout, UpSampling3D, Input, concatenate, Dense, Concatenate, Multiply, \
    multiply
from keras.layers import Reshape, Flatten, Reshape, RepeatVector, Permute, Add, Average
from keras.layers.convolutional import ZeroPadding3D, UpSampling3D
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import BatchNormalization, Conv3D, ReLU, MaxPooling3D, AveragePooling3D, Conv3DTranspose, Activation, \
    Add, GlobalAveragePooling3D
from sklearn.decomposition import PCA
from keras import regularizers
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.layers.core import Lambda


def random_normal(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)


kernel_init = "he_normal"  # "glorot_uniform" #'random_uniform'


def BN(input_conv):
    conv = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-06, center=True, scale=True)(input_conv)
    return conv



def MSpeRB(X, filters, stride, validkernel=(1, 1, 1)):
    F1, F2, F3 = filters
    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=stride, kernel_initializer=kernel_init)(X)
    X = BN(X)
    X_shortcut = X
    branch_1 = Conv3D(filters=F2, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='same',kernel_initializer=kernel_init)(X)
    branch_2 = Conv3D(filters=F2, kernel_size=(1, 1, 5), strides=(1, 1, 1), padding='same',kernel_initializer=kernel_init)(X)
    branch_3 = Conv3D(filters=F2, kernel_size=(1, 1, 7), strides=(1, 1, 1), padding='same',kernel_initializer=kernel_init)(X)
    X = concatenate([branch_1,branch_2,branch_3],axis=4)
    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), kernel_initializer=kernel_init)(X)
    X = BN(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = Conv3D(filters=F3, kernel_size=validkernel, strides=(1, 1, 1), padding='valid', kernel_initializer=kernel_init)(X)
    X = BN(X)
    X = Activation('relu')(X)
    return X


from keras.regularizers import l2
def MSpaRB(X, filters, stride, validkernel=(1, 1, 1)):
    F1, F2, F3 = filters
    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=stride, kernel_initializer=kernel_init)(X)
    # X = Conv3D(filters= F3, kernel_size= (1, 1, 1), strides= stride, kernel_initializer= kernel_init, kernel_regularizer= l2(0.001))(X)
    X = BN(X)
    X_shortcut = X
    branch_1 = Conv3D(filters=F2, kernel_size=(3, 3, 1), strides=(1, 1, 1), padding='same',kernel_initializer=kernel_init)(X)
    branch_2 = Conv3D(filters=F2, kernel_size=(5, 5, 1), strides=(1, 1, 1), padding='same',kernel_initializer=kernel_init)(X)
    branch_3 = Conv3D(filters=F2, kernel_size=(7, 7, 1), strides=(1, 1, 1), padding='same',kernel_initializer=kernel_init)(X)
    X = concatenate([branch_1, branch_2, branch_3], axis=4)
    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), kernel_initializer=kernel_init)(X)
    X = BN(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = Conv3D(filters=F3, kernel_size=validkernel, strides=(1, 1, 1), padding='valid', kernel_initializer=kernel_init)(
        X)
    X = BN(X)
    X = Activation('relu')(X)
    return X

def SCAM(X):

    a, b = X.shape[3], X.shape[4]

    conv_GAP = GlobalAveragePooling3D(data_format='channels_last')(X)
    conv_GAP = Dense(512, activation="relu")(conv_GAP)
    conv_GAP = Dense(int(b), activation="sigmoid")(conv_GAP)
    channel_attention = Reshape((1, 1, 1, int(b)))(conv_GAP)
    #    print("wwwww",channel_attention)
    #    channel_attention = K.resize_volumes(channel_attention, 4, 4, a,"channels_last")

    conv_band4x4 = Conv3D(1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', kernel_initializer=kernel_init)(X)
    conv_band2x2 = Conv3D(1, kernel_size=(int(patch_size / 2), int(patch_size / 2), 1),
                          strides=(int(patch_size / 2), int(patch_size / 2), 1), padding='valid',
                          kernel_initializer=kernel_init)(X)
    # conv_band1x1 = Conv3D(1, kernel_size=(patch_size / 4, patch_size / 4, 1), strides=(patch_size / 4, patch_size / 4, 1),
    #                       padding='valid', kernel_initializer=kernel_init)(X)

    conv_band4x4 = Activation("sigmoid")(conv_band4x4)
    conv_band2x2 = Activation("sigmoid")(conv_band2x2)
    # conv_band1x1 = Activation("sigmoid")(conv_band1x1)

    conv_band2x2 = UpSampling3D(size=(int(patch_size / 2), int(patch_size / 2), 1))(conv_band2x2)
    # conv_band1x1 = UpSampling3D(size=(int(patch_size / 4), int(patch_size / 4), 1))(conv_band1x1)

    attention = Average()([channel_attention, conv_band2x2, conv_band4x4])
    # attention = Average()([channel_attention, conv_band2x2, conv_band4x4, conv_band1x1])

    attention = Multiply()([attention, X])

    amc = Add()([attention, X])
    return amc


def SAM(X):
    conv_band1x1 = Conv3D(1, kernel_size=(patch_size, patch_size, 1), strides=(patch_size, patch_size, 1),
                          padding='valid', kernel_initializer=kernel_init)(X)
    conv_band1x1 = Activation("sigmoid")(conv_band1x1)
    conv_band1x1 = UpSampling3D(size=(int(patch_size), int(patch_size), 1))(conv_band1x1)
    conv2n = Multiply()([conv_band1x1, X])
    conv2n = Add()([conv2n, X])
    return conv2n


patch_size = 8


def unet(pretrained_weights=None, input_size=(patch_size, patch_size, 204, 1)):
    inputs = Input(shape=input_size, name='inputs')
    strides111 = (1, 1, 1)
    strides112 = (1, 1, 2)
    strides113 = (1, 1, 3)
    # channel 1
    conv00c = SCAM(inputs)
    conv00s = SAM(inputs)
    conv00u = concatenate([inputs,conv00s],axis=4)
    conv00p = concatenate([inputs,conv00c],axis=4)
    conv0a = MSpaRB(conv00u, filters=[32, 32, 64], stride=strides112,
                                 validkernel=(1, 1, 3))
    conv0b = MSpeRB(conv00p, filters=[32, 32, 64], stride=strides112, validkernel=(1, 1, 3))

    conv11c = SCAM(conv0a)
    conv11s = SAM(conv0b)
    conv11u = concatenate([conv0a,conv11s],axis=4)
    conv11p = concatenate([conv0b,conv11c],axis=4)
    conv1a = MSpaRB(conv11u, filters=[32, 32, 64], stride=strides112)
    conv1b = MSpeRB(conv11p, filters=[32, 32, 64], stride=strides112)

    conv22c = SCAM(conv1a)
    conv22s = SAM(conv1b)
    conv22u = concatenate([conv1a,conv22s],axis=4)
    conv22p = concatenate([conv1b,conv22c],axis=4)
    conv2a = MSpaRB(conv22u, filters=[64, 64, 128], stride=strides112,
                                 validkernel=(1, 1, 2))
    conv2b = MSpeRB(conv22p, filters=[64, 64, 128], stride=strides112, validkernel=(1, 1, 2))

    conv33c = SCAM(conv2a)
    conv33s = SAM(conv2b)
    conv33u = concatenate([conv2a,conv33s],axis=4)
    conv33p = concatenate([conv2b,conv33c],axis=4)
    conv3a = MSpaRB(conv33u, filters=[64, 64, 128], stride=strides112)
    conv3b = MSpeRB(conv33p, filters=[64, 64, 128], stride=strides112)

    conv44c = SCAM(conv3a)
    conv44s = SAM(conv3b)
    conv44u = concatenate([conv3a,conv44s],axis=4)
    conv44p = concatenate([conv3b,conv44c],axis=4)
    conv4a = MSpaRB(conv44u, filters=[128, 128, 256], stride=strides112)
    conv4b = MSpeRB(conv44p, filters=[128, 128, 256], stride=strides112)

    conv55c = SCAM(conv4a)
    conv55s = SAM(conv4b)
    conv55u = concatenate([conv4a,conv55s],axis=4)
    conv55p = concatenate([conv4b,conv55c],axis=4)
    conv5a = MSpaRB(conv55u, filters=[128, 128, 256], stride=strides112)
    conv5b = MSpeRB(conv55p, filters=[128, 128, 256], stride=strides112)

    conv6 = concatenate([conv5a, conv5b], axis=4)

    conv_1 = AveragePooling3D(pool_size=(1, 1, 3), strides=None, padding='valid', data_format=None)(conv6)

    _conv1 = Conv3D(16, (1, 1, 1), strides=(1, 1, 1), activation='softmax', padding='valid',
                    kernel_initializer=kernel_init, name='out_conv1')(conv_1)

    model = Model(inputs=inputs, outputs=_conv1)

    model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

















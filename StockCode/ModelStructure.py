import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation,Dense ,Dropout ,GRU, ConvLSTM2D ,LSTM ,Bidirectional,TimeDistributed,Flatten, Conv1D, MaxPooling1D,BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow import keras

height = 30
width = 89
channel = 1
output_shape = 2
inputs = keras.Input(shape=(height, width ,channel), name="input")

"""
Basic Conv2D
"""
def Conv2D():
    model = Sequential([
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu')(inputs),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    # model.summary()
    return model

"""
Basic Conv2D with BatchNormalization
"""
def Conv2DBatchNorm():
    model = Sequential([
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu')(inputs),
        keras.layers.Dropout(0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    # model.summary()
    return model

"""
Basic Conv2D with LayerNormalization
"""
def Conv2DLayerNorm():
    model = Sequential([
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu')(inputs),
        keras.layers.Dropout(0.5),
        keras.layers.LayerNormalization(axis=1 , center=True , scale=True),
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    # model.summary()
    return model

"""
Basic Conv2D with InstanceNormalization
"""
def Conv2DInstanceNorm():
    model = Sequential([
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu')(inputs),
        keras.layers.Dropout(0.5),
        tfa.layers.InstanceNormalization(axis=1),
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu'),
        keras.layers.Dropout(0.5),
        tfa.layers.InstanceNormalization(axis=1),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    # model.summary()
    return model

"""
Basic Conv2D with GroupNormalization
"""
def Conv2DGroupNorm():
    model = Sequential([
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu')(inputs),
        keras.layers.Dropout(0.5),
        tfa.layers.GroupNormalization(groups=(X.shape[2] - ((3,3)[0] - 1)), axis=2),
        keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu'),
        keras.layers.Dropout(0.5),
        tfa.layers.GroupNormalization(groups=(X.shape[2] - ((3,3)[0] - 1)*2), axis=2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    # model.summary()
    return model

"""
Basic Conv2D with WeightNormalization
"""
def Conv2DWeightNorm():
    model = Sequential([
        tfa.layers.WeightNormalization(keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu'))(inputs),
        keras.layers.Dropout(0.5),
        tfa.layers.WeightNormalization((keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu'))),
        keras.layers.Dropout(0.5),
        tfa.layers.WeightNormalization((keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation='relu'))),
        keras.layers.Flatten(),
        tfa.layers.WeightNormalization((keras.layers.Dense(128, activation='relu'))),
        tfa.layers.WeightNormalization((keras.layers.Dense(output_shape, activation='softmax')))
    ])
    # model.summary()
    return model

"""
Simple Restnet Network
"""
def SimpleRestNet():
    inputs = keras.Input(shape=(height, width ,channel), name="input_standar")
    x = layers.Conv2D(64, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    block_1_output = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(output_shape)(x)

    model = keras.Model(inputs, outputs, name="SimpleRestNet")
    # model.summary()
    return model

"""
Multi Input Network
"""
def MultiInput():
    # define two sets of inputs
    inputA = keras.Input(shape=(height, width ,channel), name="input_ema")
    inputB = keras.Input(shape=(height, width ,channel), name="input_sma")
    # the first branch operates on the first input
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputA)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = keras.Model(inputs=inputA, outputs=x)
    # the second branch opreates on the second input
    y = layers.Conv2D(32, 3, activation="relu", padding="same")(inputB)
    y = layers.Conv2D(64, 3, activation="relu", padding="same")(y)
    y = layers.BatchNormalization()(y)
    y = keras.Model(inputs=inputB, outputs=y)
    # combine the output of the two branches
    combined = layers.concatenate([x.output, y.output])
    # combined outputs
    z = layers.Conv2D(32, 3, activation="relu", padding="same")(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dense(256, activation="relu")(z)
    z = layers.Flatten()(z)
    z = layers.Dropout(0.5)(z)
    outputs = layers.Dense(output_shape, activation='softmax', name='output')(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = keras.Model(inputs=[x.input, y.input], outputs=[outputs] , name = 'MultiInput')
    return model

"""
Multi Input II Network
"""

def lstm_conv1d(x, filter, kernel_size, strides=1, padding='same'):
    x = layers.LSTM(filter, return_sequences=True, dropout=0.2 )(x)
    x = layers.Conv1D(filter, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def concat_lc(x,name=None):
    combined = layers.concatenate(x,name=name)
    combined = lstm_conv1d(combined,64,3)
    return combined

def MultiInputII():
    inputA = keras.Input(shape=(height, width ), name="input_ratio")
    inputB = keras.Input(shape=(height, width ), name="input_ema")
    inputC = keras.Input(shape=(height, width ), name="input_ema_10")

    a = lstm_conv1d(inputA, 64, 3)
    a_o = keras.Model(inputs=[inputA], outputs=a)

    x = lstm_conv1d(inputB, 64, 3)
    x_o = keras.Model(inputs=[inputB], outputs=x)

    y = lstm_conv1d(inputC, 64, 3)
    y_o = keras.Model(inputs=[inputC], outputs=y)

    combined = concat_lc([a_o.output,x_o.output, y_o.output], name= 'combine')

    a_1 = lstm_conv1d(a, 64, 3)
    a_1_o = keras.Model(inputs=[a_o.input], outputs=a_1)

    x_1 = lstm_conv1d(x, 64, 3)
    x_1_o = keras.Model(inputs=[x_o.input], outputs=x_1)

    y_1 = lstm_conv1d(y, 64, 3)
    y_1_o = keras.Model(inputs=[y_o.input], outputs=y_1)

    combined = lstm_conv1d(combined, 64, 3)

    combined_1 = concat_lc([a_1_o.output,x_1_o.output, y_1_o.output], name = 'combine_1')

    combined_3 = layers.add([combined,combined_1],name= 'combine_3')

    z = layers.Dense(256, activation="relu")(combined_3)
    z = layers.Flatten()(z)
    z = layers.Dropout(0.2)(z)

    outputs = layers.Dense(output_shape, activation='softmax', name='output')(z)
    model = keras.Model(inputs=[a_1_o.input, x_1_o.input, y_1_o.input], outputs=[outputs] , name = 'MultiInput')

    return model

"""
Restnet-18 Network
"""
def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = layers.Conv2D(nb_filter, kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          kernel_regularizer = keras.regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def shortcut(input, residual):
    """
    shortcut, which is identity mapping
    """
    input_shape = keras.backend.int_shape(input)
    residual_shape = keras.backend.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    identity = input
    # 如果維度不同，則使用1x1卷積進行調整
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = layers.Conv2D(filters=residual_shape[3],
                           kernel_size=(1, 1),
                           strides=(stride_width, stride_height),
                           padding="valid",
                           kernel_regularizer = keras.regularizers.l2(0.0001))(input)

    return layers.add([identity, residual])


def basic_block(nb_filter, strides=(1, 1)):
    """
    Basic ResNet building bloc
    """
    def f(input):

        conv1 = conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
        residual = conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))

        return shortcut(input, residual)

    return f


def residual_block(nb_filter, repetitions, is_first_layer=False):
    """
    build residual block，對應論文引數統計表中的conv2_x -> conv5_x
    """
    def f(input):
        for i in range(repetitions):
            strides = (1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2)
            input = basic_block(nb_filter, strides)(input)
        return input

    return f


def resnet_18():
    input_ = keras.Input(shape=(height, width ,channel),name = 'input_standar')

    conv1 = conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

    conv2 = residual_block(64, 2, is_first_layer=True)(pool1)
    conv3 = residual_block(128, 2, is_first_layer=True)(conv2)
    conv4 = residual_block(256, 2, is_first_layer=True)(conv3)
    conv5 = residual_block(512, 2, is_first_layer=True)(conv4)

    pool2 = layers.GlobalAvgPool2D()(conv5)
    output_ = layers.Dense(output_shape, activation='softmax')(pool2)

    model = keras.Model(inputs=input_, outputs=output_)

    return model

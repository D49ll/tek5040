import tensorflow as tf
from tensorflow.keras import layers, models

def simple_model(input_shape):

    height, width, channels = input_shape
    image = layers.Input(input_shape)
    x = layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu')(image)
    x = layers.Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 1, padding='same', activation=None)(x)
    # resize back into same size as regularization mask
    x = tf.image.resize(x, [height, width])
    x = tf.keras.activations.sigmoid(x)

    model = models.Model(inputs=image, outputs=x)

    return model

def conv2d_3x3(filters):
    conv = layers.Conv2D(
        filters, kernel_size=(3, 3), activation='relu', padding='same'
    )
    return conv

def up_conv(image, filter1, filter2):
    c = conv2d_3x3(filter1)(image)
    c = conv2d_3x3(filter2)(c)
    p = layers.UpSampling2D(
        filter2, kernel_size=(2,2),padding='same'
    )
    return c, p


def max_pool():
    return layers.MaxPooling2D((2, 2), strides=2, padding='same')

def down_conv(image, filters):
    c = conv2d_3x3(filters)(image)
    c = conv2d_3x3(filters)(c)
    p = max_pool()(c)

    return c, p

def unet(input_shape):

    image = layers.Input(shape=input_shape)

    #Down
    c1, p1 = down_conv(image, 8)
    c2, p2 = down_conv(p1, 16)
    c3, p3 = down_conv(p2, 32)
    c4, p4 = down_conv(p3, 64)


    #Up
    c5, p5 = up_conv(c4, p4, 128, 64)
    


    # Fill the layers from 2 to 9.
    # .........................

    probs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=image, outputs=probs)

    return model
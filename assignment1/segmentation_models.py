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

def conv2d_3x3(filter):
    return layers.Conv2D(filter, kernel_size=(3, 3), activation='relu', padding='same')

def max_pool():
    return layers.MaxPooling2D((2, 2), strides=2, padding='same')

def down_conv(image, filter):
    c = conv2d_3x3(filter)(image)
    c = conv2d_3x3(filter)(c)
    p = max_pool()(c)

    return c, p

def up_conv(image, filter):
    c = conv2d_3x3(filter*2)(image)
    c = conv2d_3x3(filter*2)(c)
    
    return layers.Conv2DTranspose(filter, kernel_size=(2,2), strides=2, padding='same')(c)


def output_seg(image):
    c = conv2d_3x3(8)(image)
    c = conv2d_3x3(8)(c)

    return layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(c)

def unet(input_shape):
    image = layers.Input(shape=input_shape)
    
    #Downwards
    c1, p1 = down_conv(image, 8)
    c2, p2 = down_conv(p1, 16)
    c3, p3 = down_conv(p2, 32)
    c4, p4 = down_conv(p3, 64)

    #Bottom
    c5 = up_conv(p4, 64)

    #Upwards
    c6 = up_conv(tf.concat([c4, c5], 3), 32)
    c7 = up_conv(tf.concat([c3, c6], 3), 16)
    c8 = up_conv(tf.concat([c2, c7], 3), 8)

    #Output segment
    probs = output_seg(tf.concat([c1,c8],3))

    model = models.Model(inputs=image, outputs=probs)
    print(model.summary())

    return model
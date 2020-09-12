import tensorflow as tf


@tf.function
def shift_left_one_hot(vec, shift=-1):
    P = tf.eye(tf.shape(vec)[0])
    P = tf.roll(P, shift=shift, axis=0)

    vec = tf.expand_dims(vec, 0)

    return vec @ P


@tf.function
def dot(x, y):
    r = tf.multiply(x, y)
    return tf.reduce_sum(r, -1)

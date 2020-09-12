import tensorflow as tf


@tf.function
def to_prob_dist_all(v):
    v2 = tf.sqrt(tf.square(v)+1e-9)
    # v2 = tf.sqrt(tf.square(v))
    m = tf.expand_dims(tf.reduce_sum(v2, axis=-1), -1)
    n = tf.math.divide_no_nan(v2, m)
    return n


@tf.function
def cross_entropy(x, y, epsilon=1e-9):
    return -2 * tf.reduce_mean(y * tf.math.log(x + epsilon), -1) / tf.math.log(2.)


@tf.function
def entropy(x):
    return cross_entropy(x, x)

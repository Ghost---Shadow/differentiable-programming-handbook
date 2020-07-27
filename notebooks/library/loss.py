import tensorflow as tf


@tf.function
def bistable_loss(x):
    a = (x ** 2)
    b = (x - 1) ** 2

    return a * b


@tf.function
def permute_matrix_loss(P, cycle_length=1, cycle_weight=0):
    loss = 0

    P_square = tf.math.square(P)
    axis_1_sum = tf.reduce_sum(P_square, axis=1)
    axis_0_sum = tf.reduce_sum(P_square, axis=0)

    # Penalize axes not adding up to one
    loss += tf.nn.l2_loss(axis_1_sum - 1)
    loss += tf.nn.l2_loss(axis_0_sum - 1)

    # Penalize numbers outside 0 or 1
    loss += tf.math.reduce_sum(bistable_loss(P))

    # Cycle loss
    Q = P
    for _ in tf.range(cycle_length - 1):
        Q = P @ Q
    cycle_loss = tf.nn.l2_loss(Q - tf.eye(tf.shape(Q)[0]))
    loss += cycle_loss * cycle_weight

    return loss

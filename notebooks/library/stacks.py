import tensorflow as tf

from .array_ops import assign_index_vectored, superposition_lookup_vectored


def new_stack(stack_shape, is_learnable=False):
    buffer = tf.zeros(stack_shape, dtype=tf.float32)
    index = tf.one_hot(0, stack_shape[0], dtype=tf.float32)

    if is_learnable:
        buffer = tf.Variable(buffer)
        index = tf.Variable(index)

    stack = (buffer, index)
    return stack


def new_stack_from_buffer(buffer, is_learnable=False):
    stack_shape = tf.shape(buffer)
    index = tf.one_hot(0, stack_shape[0], dtype=tf.float32)

    if is_learnable:
        buffer = tf.Variable(buffer)
        index = tf.Variable(index)

    stack = (buffer, index)
    return stack


@tf.function
def stack_push(state, element):
    buffer, index = state
    buffer = assign_index_vectored(buffer, index, element)
    index = tf.roll(index, shift=1, axis=0)
    state = (buffer, index)
    return state


@tf.function
def stack_pop(state):
    buffer, index = state
    index = tf.roll(index, shift=-1, axis=0)
    element = superposition_lookup_vectored(buffer, index)
    state = (buffer, index)
    return state, element


@tf.function
def stack_peek(stack):
    buffer, index = stack
    index = tf.roll(index, shift=-1, axis=0)
    element = superposition_lookup_vectored(buffer, index)
    return element

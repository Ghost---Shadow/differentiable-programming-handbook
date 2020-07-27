import tensorflow as tf

from .array_ops import assign_index_vectored, superposition_lookup_vectored


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

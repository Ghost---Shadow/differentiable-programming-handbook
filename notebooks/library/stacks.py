import tensorflow as tf

from library.array_ops import assign_index


@tf.function
def stack_push(state, element):
    buffer, index = state
    buffer = assign_index(buffer, index, element)
    index += 1
    
    state = (buffer, index)
    return state


@tf.function
def stack_pop(state):
    buffer, index = state
    index -= 1
    element = buffer[index]
    
    state = (buffer, index)
    return state, element

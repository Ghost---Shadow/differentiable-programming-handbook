import tensorflow as tf


@tf.function
def assign_index(arr, index, element):
    arr_shape = tf.shape(arr)
    
    pos_mask = tf.eye(arr_shape[0])[index]
    pos_mask = tf.transpose(tf.expand_dims(pos_mask, 0))
    neg_mask = 1 - pos_mask
    
    tiled_element = tf.reshape(tf.tile(element, [arr_shape[0]]), arr_shape)
    
    arr = arr * neg_mask + tiled_element * pos_mask
    
    return arr


@tf.function
def assign_index_vectored(arr, index, element):
    arr_shape = tf.shape(arr)
    
    pos_mask = tf.transpose(tf.expand_dims(index, 0))
    neg_mask = 1 - pos_mask
    
    tiled_element = tf.reshape(tf.tile(element, [arr_shape[0]]), arr_shape)

    arr = arr * neg_mask + tiled_element * pos_mask
    
    return arr


@tf.function
def naive_lookup(arr, index):
    index = tf.round(index)
    index = tf.cast(index, tf.int32)
    result = arr[index]
    return result


@tf.function
def interp_factor(index):
    t1 = tf.math.floor(index)
    t2 = tf.math.ceil(index)
    
    t = tf.math.divide_no_nan((index - t1), (t2 - t1))
    
    i1 = tf.cast(t1, tf.int32)
    i2 = tf.cast(t2, tf.int32)
    
    return t, i1, i2


@tf.function
def linear_lookup(arr, index):
    t, i1, i2 = interp_factor(index)
    
    # Linear interpolation
    result = t * arr[i1] + (1 - t) * arr[i2]
    
    return result


@tf.function
def superposition_lookup_vectored(arr, indices):
    if tf.rank(arr) == 1:
        arr = tf.expand_dims(arr, -1)
    indices = tf.expand_dims(indices, -1)
    result = arr * indices
    return tf.reduce_sum(result, axis=0)


@tf.function
def bandwidthify(index, bandwidth):
    t, i1, i2 = interp_factor(index)
    
    # Prevent array out of bounds
    i1 = tf.clip_by_value(i1, 0, bandwidth - 1)
    i2 = tf.clip_by_value(i2, 0, bandwidth - 1)
    t = tf.clip_by_value(t, 0, 1)
    
    # Linear interpolation
    eye = tf.eye(bandwidth)
    result = t * eye[i1] + (1 - t) * eye[i2]
    
    return result


@tf.function
def bulk_bandwidthify(indices, bandwidth):
    num_indices = tf.shape(indices)[0]
    
    indices = tf.unstack(indices)
    result = tf.zeros((num_indices, bandwidth), dtype=tf.float32)
    result = tf.unstack(result)
    
    for i, index in enumerate(indices):
        b_index = bandwidthify(index, bandwidth)
        result[i] += b_index
    
    result = tf.stack(result)
    return result


@tf.function
def superposition_lookup(arr, index):
    bandwidth = tf.shape(arr)[0]
    vectored_index = bandwidthify(index, bandwidth)
    result = superposition_lookup_vectored(arr, vectored_index)
    
    return result


@tf.function
def residual_lookup(arr, index):
    i = tf.round(index)
    residue = index - i
    i = tf.cast(i, tf.int32)
    
    result = arr[i]
    
    return result, residue


@tf.function
def tensor_lookup_2d(arr, x_index, y_index):
    # Calculate outer product
    mask = tf.tensordot(x_index, y_index, axes=0)
    
    # Broadcast the mask to match dimensions with arr
    kernel_shape = tf.shape(arr)[:2]
    ones = tf.ones(tf.rank(arr) - 2, dtype=tf.int32)
    new_shape = tf.concat((kernel_shape, ones),axis=0)
    mask = tf.reshape(mask, new_shape)
    
    # Multiply the mask
    masked_arr = mask * arr
    element = tf.math.reduce_max(masked_arr, axis=[0,1])
    return element


@tf.function
def tensor_write_2d(arr, element, x_index, y_index):
    arr_shape = tf.shape(arr)
    mask = tf.tensordot(x_index, y_index, axes=0)
    
    # Broadcast the mask to match dimensions with arr
    kernel_shape = arr_shape[:2]
    ones = tf.ones(tf.rank(arr) - 2, dtype=tf.int32)
    new_shape = tf.concat((kernel_shape, ones),axis=0)
    mask = tf.reshape(mask, new_shape)
    
    element = tf.reshape(element,[1,1,-1])
    element = tf.tile(element, [arr_shape[0], arr_shape[1], 1])
    
    result = (1.0 - mask) * arr + mask * element
    
    return result

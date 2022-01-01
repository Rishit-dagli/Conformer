import tensorflow as tf

class Swish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)

class GLU(tf.keras.layers.Layer):
    def __init__(self, diim, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.dim = dim

    def call(self, inputs):
        out, gate = tf.split(inputs, 2, axis=self.dim)

class DepthwiseLayer(tf.keras.layers.Layer):
    def __init__(self, chan_in, chan_out, kernel_size, padding, **kwargs):
        super(DepthwiseLayer, self).__init__(**kwargs)
        self.padding = padding
        self.conv = tf.keras.layers.Conv1D(chan_in, chan_out, kernel_size, groups = chan_in)
    
    def call(self, inputs):
        pad_list = []
        for _ in range(inputs.shape.rank - 1):
            pad_list.append([0, 0])
        pad_list.append(list(self.padding))
        inputs = tf.pad(inputs, pad_list)
        return self.conv(inputs)

class Scale(tf.keras.layers.Layer):
    def __init__(self, scale, fn, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.scale = scale
        self.fn = fn
    
    def call(self, inputs, **kwargs):
        return self.fn(inputs, **kwargs) * self.scale

class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn, **kwargs):
        super(PreNorm, self).__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.fn = fn
    
    def call(self, inputs, **kwargs):
        inputs = self.norm(inputs)
        return self.fn(inputs, **kwargs)


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




"""
    * GradientTape is released as part of Tensorflow-2.0 to support automatic
    * differentiation. This promotes and eases custom training batches.
    * Reference: https://www.tensorflow.org/api_docs/python/tf/GradientTape
"""
import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x
dy_dx = g.gradient(y, x) # Will compute to 6.0

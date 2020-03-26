"""
    * GradientTape is released as part of Tensorflow-2.0 to support automatic
    * differentiation. This promotes and eases custom training batches.
    * Reference:
    *   https://www.tensorflow.org/api_docs/python/tf/GradientTape
    *   https://www.pyimagesearch.com/2020/03/23/using-tensorflow-and-gradienttape-to-train-a-keras-model/
"""
import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x
dy_dx = g.gradient(y, x) # Will compute to 6.0


"""
    * Below is a snippet for custom training using gradient tape.
    * Call this method on custom loop for n_epochs.
"""

def train_step(X, y):
	# keep track of our gradients
	with tf.GradientTape() as tape:
		# make a prediction using the model and then calculate the loss
		prediction = model(X)
		loss = categorical_crossentropy(y, prediction)
	# calculate the gradients using our tape and then update the model weights
	gradients = tape.gradient(loss, model.trainable_variables)
	opt.apply_gradients(zip(gradients, model.trainable_variables))

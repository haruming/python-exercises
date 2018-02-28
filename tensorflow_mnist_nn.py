import tensorflow as tf
import numpy as np

# use the MNIST data set in tensorflow
from tensorflow.examples.tutorials.mnist import input_data

data_path = os.path.join('.', 'mnist_data')
# one-hot representation, here we use number 0-9 which means 10 entrances
mnist = input_data.read_data_sets(data_path, one_hot=True)

input_layer = 28 * 28  # input layer

hidden_layer_1 = 500   # hidden layer 1
hidden_layer_2 = 1000  # hidden layer 2
hidden_layer_3 = 300   # hideen layer 3

output_layer = 10      # output layer

# normally, linear --> 1   non-linear --> 2   super non-liear --> 3+

def neural_network(data):
	# define the weights and biases of the first second third layers
	layer_1_w_b = {'w_':tf.Variable(tf.random_normal([input_layer, hidden_layer_1])), 'b_':tf.Variable(tf.random_normal(hidden_layer_1))}
	layer_2_w_b = {'w_':tf.Variable(tf.random_normal([hidden_layer_1, hidden_layer_2])), 'b_':tf.Variable(tf.random_normal(hidden_layer_2))}
	layer_3_w_b = {'w_':tf.Variable(tf.random_normal([hidden_layer_2, hidden_layer_3])), 'b_':tf.Variable(tf.random_normal(hidden_layer_3))}
	# define the weights and biases between the third and output layer
	layer_output_w_b = {'w_':tf.Variable(tf.random_normal([hidden_layer_3, output_layer])), 'b_':tf.Variable(tf.random_normal(output_layer))}

	# before neuron value = weight * input + bias
	layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['_b'])
	# after neuron value, activation function, here we use ReLU
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['_b'])
	layer_2 = tf.nn.relu(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['_b'])
	layer_3 = tf.nn.relu(layer_3)

	# no need to activate the last layer
	layer_output = tf.add(tf.matmul(layer_3, layer_output_w_b['w_']), layer_output_w_b['_b'])

	return layer_output


# we use 100 records as a batch for each epoch
batch_size = 100

# the second parameter is the matrix size for two sizes
X = tf.placeholder('float', [None, 28*28])    # input
Y = tf.placeholder('float')   # output

# use data to train the neural network
def train_neural_network(X, Y):
	# get the predict value of after neural network
	predict = neural_network(X)
	# calculate the loss function with softmax and cross entropy
	loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
	optimizer = tf.train.AdamOptimizer().minimize(loss_function) # learning rate = 0.001

	# define the epochs
	epochs = 13
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		epoch_loss = 0
		for epoch in range(epochs):
			for i in range(int(mnist.train.num_examples/batch_size)):
				x, y = mnist.train.next_batch(batch_size)
				_, c = session.run([optimizer, loss_function], feed_dict={X:x, Y:y})
				epoch_loss += c
			print(epoch, ' : ', epoch_loss)

		correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('accuracy: ', accuracy.eval({X:mnist.test.images, Y:mnist.test.labels}))

train_neural_network(X,Y)
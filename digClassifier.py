#Loading MNIST dataset

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#importing tensorflow library

import tensorflow as tf

#defining variables

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])



#defing cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#defining the descent optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()


#creating and running the session
sess = tf.Session()
sess.run(init)


#using batch gradient descent for faster training from training slot of MNIST
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#Defing accuracy nodes

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#running the nodes by testing the trained model on test model
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


#close the session
sess.close()
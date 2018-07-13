import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class Model(object):
    def __init__(self, learning_rate, epochs, batch_size, display_step, no_layers, no_nodes):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.no_layers = no_layers
        self.no_nodes = no_nodes
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        self.weights = list()
        self.bias = []
        self.activation_val = [self.x]
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    

    def make_layer(self):
        for p in range(self.no_layers-1):
            self.weights.append(tf.Variable(tf.truncated_normal(shape=(self.no_nodes[p], self.no_nodes[p+1]), mean=0, stddev=0.1)))
            self.bias.append(tf.Variable(tf.zeros(self.no_nodes[p+1])))
            if p == 0:
                linear_val = tf.matmul(self.x, self.weights[-1]) + self.bias[-1]
            else:
                linear_val = tf.matmul(self.activation_val[-1], self.weights[-1]) + self.bias[-1]
            if p == self.no_layers-1:
                self.activation_val.append(tf.nn.softmax(linear_val))
            else:
                self.activation_val.append(tf.nn.sigmoid(linear_val))
        lr = tf.placeholder(tf.float32)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=linear_val, labels=self.y))

        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

        correct_predictions = tf.equal(tf.argmax(self.activation_val[-1], 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        self.run_session(cross_entropy, optimizer, accuracy, lr)

    def run_session(self, cross_entropy, optimizer, accuracy, lr):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print(f'''
            Running session with:
            Epochs: {self.epochs:>3d}
            Learning Rate: {self.learning_rate:>6.3f}
            Batch Size: {self.batch_size:>3d}''')

            for epoch in range(self.epochs-1):
                # print("Oye;oednf;jes[figjeokwvm['wiejrgfio")
                for batch in range(self.mnist.train.num_examples//self.batch_size):
                    x_train, y_train = self.mnist.train.next_batch(self.batch_size)
                    # print(batch)
                    feed_dict = {
                        self.x : x_train,
                        self.y : y_train,
                        lr : self.learning_rate }
                    _, loss = sess.run([optimizer, cross_entropy], feed_dict = feed_dict)

                validation_acc = sess.run(accuracy, feed_dict= {
                    self.x : self.mnist.validation.images,
                    self.y : self.mnist.validation.labels})

                print(f'Epoch: {epoch:>5d}; Loss: {loss: >10.3f}; Validation Accuracy: {validation_acc:>1.4f}')


            test_accuracy = sess.run(accuracy, feed_dict = {
                self.x : self.mnist.test.images,
                self.y : self.mnist.test.labels})

            print(f'Final test accuracy: {test_accuracy:>2.2f}')

        tf.reset_default_graph()

if __name__ == '__main__':
    obj = Model(0.1, 20, 100, 2, 4, [784, 100, 100, 10])
    obj.make_layer()


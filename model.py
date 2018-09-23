import tensorflow as tf
import numpy as np
import time
import dataset

class Model(object):
    with tf.Graph().as_default():
        def __init__(self, learning_rate, no_layers, input_nodes, logits, no_nodes, data):
            self.learning_rate = learning_rate
            self.epochs = 250
            self.batch_size = 100
            self.display_step = 2
            self.no_layers = no_layers
            self.no_nodes = no_nodes
            self.input_nodes = input_nodes
            self.logits = logits
            self.x = tf.placeholder(tf.float32, shape=[None, input_nodes])
            self.y = tf.placeholder(tf.float32, shape=[None, logits])
            self.weights = list()
            self.bias = []
            self.activation_val = [self.x]
            self.data = data #read_data_sets("MNIST_data/", one_hot=True)


        def make_layer(self):
            for p in range(self.no_layers-1):
                if p == 0:
                    self.weights.append(tf.Variable(tf.truncated_normal(shape=(self.input_nodes, self.no_nodes[p]), mean=0, stddev=0.1)))
                elif p == self.no_layers-2:
                    self.weights.append(tf.Variable(tf.truncated_normal(shape=(self.no_nodes[p-1], self.logits), mean=0, stddev=0.1)))
                else:
                    self.weights.append(tf.Variable(tf.truncated_normal(shape=(self.no_nodes[p-1], self.no_nodes[p]), mean=0, stddev=0.1)))
                
                if p == self.no_layers-2:
                    self.bias.append(tf.Variable(tf.zeros(self.logits)))
                else:
                    self.bias.append(tf.Variable(tf.zeros(self.no_nodes[p])))

                if p == 0:
                    linear_val = tf.matmul(self.x, self.weights[-1]) + self.bias[-1]
                else:
                    linear_val = tf.matmul(self.activation_val[-1], self.weights[-1]) + self.bias[-1]

                if p == self.no_layers-2:
                    self.activation_val.append(tf.nn.softmax(linear_val))
                else:
                    self.activation_val.append(tf.nn.sigmoid(linear_val))

            lr = tf.placeholder(tf.float32)

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=linear_val, labels=self.y))

            optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

            correct_predictions = tf.equal(tf.argmax(self.activation_val[-1], 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            return(self.run_session(cross_entropy, optimizer, accuracy, lr))


        def run_session(self, cross_entropy, optimizer, accuracy, lr):
            t0 = time.time()
            test_accuracy = 0.0
            time_taken = 0.0
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                print(f'''
                Running session with:
                Epochs: {self.epochs:>3d}
                Learning Rate: {self.learning_rate:>6.3f}
                Batch Size: {self.batch_size:>3d}''')

                for epoch in range(self.epochs-1):
                    # print("Oye;oednf;jes[figjeokwvm['wiejrgfio")
                    # for batch in range(self.mnist.train.num_examples//self.batch_size):
                    x_train, y_train = self.data.train_list, self.data.train_label
                    # print(batch)
                    feed_dict = {
                        self.x : x_train,
                        self.y : y_train,
                        lr : self.learning_rate }
                    _, loss = sess.run([optimizer, cross_entropy], feed_dict = feed_dict)

                    validation_acc = sess.run(accuracy, feed_dict= {
                        self.x : self.data.validation_list,
                        self.y : self.data.validation_label})

                    print(f'Epoch: {epoch:>5d}; Loss: {loss: >10.3f}; Validation Accuracy: {validation_acc:>1.4f}')

                test_accuracy = sess.run(accuracy, feed_dict = {
                    self.x : self.data.test_list,
                    self.y : self.data.test_label})
                print(f'Final test accuracy: {test_accuracy:>2.2f}')
                sess.close()
            t1 = time.time()
            time_taken = t1-t0
            test_accuracy*=100
            return(test_accuracy/time_taken)

if __name__ == '__main__':
    data = dataset
    obj = Model(0.1, 4, 10936, 2, [1000, 1000], data)
    val = obj.make_layer()

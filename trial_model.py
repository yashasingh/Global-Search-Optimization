import tensorflow as tf
import dataset
# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[ None, 10936])
y = tf.placeholder(tf.float32, shape=[ None, 2])

#x_reshaped = tf.reshape(x, shape=[ -1, 784 ])

# Layer 1
w1 = tf.Variable(tf.truncated_normal(shape=(10936, 100), mean=0, stddev=0.1))
b1 = tf.Variable(tf.zeros([100]))

linear_1 = tf.matmul(x, w1) + b1

act_1 = tf.nn.sigmoid(linear_1)

# Layer 2
w2 = tf.Variable(tf.truncated_normal(shape=(100, 100), mean=0, stddev=0.1))
b2 = tf.Variable(tf.zeros([100]))

linear_2 = tf.matmul(act_1, w2) + b2

act_2 = tf.nn.sigmoid(linear_2)

# Layer 3
w3 = tf.Variable(tf.truncated_normal(shape=(100, 2), mean=0, stddev=0.1))
b3 = tf.Variable(tf.zeros([2]))

logits = tf.matmul(act_2, w3) + b3

prediction = tf.nn.softmax(logits)
#act_3 = tf.nn.sigmoid(linear_3)



lr = tf.placeholder(tf.float32)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

epochs = 50
learning_rate = 0.1
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(f'''
        Running session with:
        Epochs: {epochs:>3d}
        Learning Rate: {learning_rate:>6.3f}
        Batch Size: {batch_size:>3d}''')

    for epoch in range(epochs):
        # for batch in range(mnist.train.num_examples//batch_size):
        x_train, y_train = dataset.train_list, dataset.train_label

        feed_dict = {
                x : x_train,
                y : y_train,
                lr : learning_rate }

        _, loss = sess.run([optimizer, cross_entropy], feed_dict = feed_dict)

        # Calculate validation accuracy every epoch.

        valid_acc = sess.run(accuracy, feed_dict = {
            x : dataset.validation_list,
            y : dataset.validation_label})

        print(f'Epoch: {epoch:>5d}; Loss: {loss: >10.3f}; Validation Accuracy: {valid_acc:>1.4f}')
        #import pdb;pdb.set_trace()

    test_accuracy = sess.run(accuracy, feed_dict = {
        x : dataset.test_list,
        y : dataset.test_label})

    print(f'Final test accuracy: {test_accuracy:>2.2f}')

tf.reset_default_graph()

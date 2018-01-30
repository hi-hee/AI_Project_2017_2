from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


LEARNING_RATE = 0.001
TRAIN_EPOCH = 15
BATCH_SIZE = 100

#import MNIST data
mnist = input_data.read_data_sets('./',one_hot=True)

x_tran = mnist.train.images
y_tran = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels

# x_tran = tf.reshape(x_tran, [-1,28,28,1])  #28*28*1
# x_test = tf.reshape(x_test, [-1,28,28,1])  #28*28*1

input_placeholder = tf.placeholder('float', [None, 28, 28, 1])
output_placeholder = tf.placeholder('float', [None, 10])


#step 6

#convolutional layer_1
conv1 = tf.layers.conv2d(inputs= input_placeholder, filters = 32, kernel_size= [3, 3], strides= 1, padding ='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2,2],strides=2, padding='same')

#convolutional layer_2
conv2 = tf.layers.conv2d(inputs = pool1, filters= 64, kernel_size=[3,3], strides=1, padding='same',activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size=[2,2], strides=2, padding='same')

#FC layer
w_fc = tf.Variable(tf.truncated_normal([7*7*64,10]))
b_fc = tf.Variable(tf.truncated_normal([10]))
flat = tf.reshape(pool2, [-1, 7*7*64])

logits =tf.add(tf.matmul(flat,w_fc),b_fc)
cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_placeholder)
loss =  tf.reduce_mean(cost)

train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


'''learning'''
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    #training
    for epoch in range(TRAIN_EPOCH):
        total_batch = int(mnist.train.num_examples / BATCH_SIZE)
        # Loop over all batches
        for i in range(total_batch):
            _, c = sess.run([train, loss], feed_dict={input_placeholder: x_tran, output_placeholder: y_tran})


    #testing
    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(output_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({input_placeholder: x_test, output_placeholder: y_test}))
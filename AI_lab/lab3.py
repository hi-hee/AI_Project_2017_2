from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('./',one_hot=True)

INPUT = 784
HIDDEN_1 = 256
HIDDEN_2 = 256
HIDDEN_3 = 256
HIDDEN_4 = 256
OUTPUT = 10
LEARNING_RATE = 0.001
TRAIN_EPOCH = 15
BATCH_SIZE = 100

#initialize

# WEIGHT_HIDDEN_1 = tf.Variable(tf.random_normal([INPUT, HIDDEN_1]))
# WEIGHT_HIDDEN_2 = tf.Variable(tf.random_normal([HIDDEN_1,HIDDEN_2]))
# WEIGHT_OUTPUT = tf.Variable(tf.random_normal([HIDDEN_2, OUTPUT]))
#
# BIAS_HIDDEN_1 = tf.Variable(tf.random_normal([HIDDEN_1]))
# BIAS_HIDDEN_2 = tf.Variable(tf.random_normal([HIDDEN_2]))
# BIAS_OUTPUT = tf.Variable(tf.random_normal([OUTPUT]))

#Step_3
# initializer = tf.contrib.layers.xavier_initializer()
#
# WEIGHT_HIDDEN_1 = tf.Variable(initializer([INPUT, HIDDEN_1]))
# WEIGHT_HIDDEN_2 = tf.Variable(initializer([HIDDEN_1,HIDDEN_2]))
# WEIGHT_OUTPUT = tf.Variable(initializer([HIDDEN_2, OUTPUT]))
#
# BIAS_HIDDEN_1 = tf.Variable(initializer([HIDDEN_1]))
# BIAS_HIDDEN_2 = tf.Variable(initializer([HIDDEN_2]))
# BIAS_OUTPUT = tf.Variable(initializer([OUTPUT]))


#Step_4
initializer = tf.contrib.layers.xavier_initializer()

WEIGHT_HIDDEN_1 = tf.Variable(initializer([INPUT, HIDDEN_1]))
WEIGHT_HIDDEN_2 = tf.Variable(initializer([HIDDEN_1,HIDDEN_2]))
WEIGHT_HIDDEN_3 = tf.Variable(initializer([HIDDEN_2, HIDDEN_3]))
WEIGHT_HIDDEN_4 = tf.Variable(initializer([HIDDEN_3,HIDDEN_4]))
WEIGHT_OUTPUT = tf.Variable(initializer([HIDDEN_4, OUTPUT]))

BIAS_HIDDEN_1 = tf.Variable(initializer([HIDDEN_1]))
BIAS_HIDDEN_2 = tf.Variable(initializer([HIDDEN_2]))
BIAS_HIDDEN_3 = tf.Variable(initializer([HIDDEN_3]))
BIAS_HIDDEN_4 = tf.Variable(initializer([HIDDEN_4]))
BIAS_OUTPUT = tf.Variable(initializer([OUTPUT]))


input_placeholder = tf.placeholder("float", shape=[None, INPUT])  # 자료형 선언만
output_placeholder = tf.placeholder("float", shape=[None, OUTPUT])

#Step1_
# hidden_layer_1 = tf.nn.softmax(tf.add(tf.matmul(input_placeholder, WEIGHT_HIDDEN_1), BIAS_HIDDEN_1))
# logits = tf.add(tf.matmul(hidden_layer_1, WEIGHT_OUTPUT), BIAS_OUTPUT)

#Step2 & 3
# hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(input_placeholder, WEIGHT_HIDDEN_1), BIAS_HIDDEN_1))
# hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1, WEIGHT_HIDDEN_2), BIAS_HIDDEN_2))
# logits = tf.add(tf.matmul(hidden_layer_2, WEIGHT_OUTPUT), BIAS_OUTPUT)

#Step4
'''overfitting'''
# hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(input_placeholder, WEIGHT_HIDDEN_1), BIAS_HIDDEN_1))
# hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1, WEIGHT_HIDDEN_2), BIAS_HIDDEN_2))
# hidden_layer_3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_2, WEIGHT_HIDDEN_3), BIAS_HIDDEN_3))
# hidden_layer_4 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_3, WEIGHT_HIDDEN_4), BIAS_HIDDEN_4))
# logits = tf.add(tf.matmul(hidden_layer_2, WEIGHT_OUTPUT), BIAS_OUTPUT)


#Step5
'''dropout'''
keep_prob=tf.placeholder(tf.float32)

hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(input_placeholder, WEIGHT_HIDDEN_1), BIAS_HIDDEN_1))
hidden_layer_1 = tf.nn.dropout(hidden_layer_1,keep_prob)

hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1, WEIGHT_HIDDEN_2), BIAS_HIDDEN_2))
hidden_layer_2 = tf.nn.dropout(hidden_layer_2,keep_prob)

hidden_layer_3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_2, WEIGHT_HIDDEN_3), BIAS_HIDDEN_3))
hidden_layer_3 = tf.nn.dropout(hidden_layer_3,keep_prob)

hidden_layer_4 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_3, WEIGHT_HIDDEN_4), BIAS_HIDDEN_4))
hidden_layer_4 = tf.nn.dropout(hidden_layer_4,keep_prob)

logits = tf.add(tf.matmul(hidden_layer_4, WEIGHT_OUTPUT), BIAS_OUTPUT)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_placeholder))
train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)


'''learning'''
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    #training
    for epoch in range(TRAIN_EPOCH):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / BATCH_SIZE)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train, loss], feed_dict={input_placeholder: batch_x, output_placeholder: batch_y, keep_prob: 0.8})
            # Compute average loss
            avg_cost += c / total_batch

    #testing
    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(output_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({input_placeholder: mnist.test.images, output_placeholder: mnist.test.labels,keep_prob: 1.0}))


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read data
mnist_data = input_data.read_data_sets('../../data/mnist', one_hot=True)

# mnist: 28*28*1

def conv_layer(x, kernel: list, strides: list, out_channel: int, padding: str, name, do_act=True):
    # x: NHWC
    # kernel: [k_h, k_w]
    with tf.variable_scope(name):
        w = tf.Variable(tf.random_normal(
            [*kernel, x.get_shape().as_list()[-1], out_channel]
            ), name='w')
        
        # strides: [1,s,s,1]
        output = tf.nn.conv2d(x, w, strides=strides, padding=padding, name='conv')

        if do_act:
            output = tf.nn.relu(output, name='relu')
    return output

def fully_connect(x, out_num, name, do_act=True):
    with tf.variable_scope(name):
        w = tf.Variable(
            tf.random_normal([x.get_shape().as_list()[-1], out_num]),
            name='w'
            )
        b = tf.Variable(
            tf.random_normal([1, out_num]),
            name='b'
            )
        output = tf.matmul(x, w) + b

        if do_act:
            output = tf.nn.relu(output)
        return output

def build_network(x):
    # x: 28*28*1
    output = conv_layer(x, [3, 3], [1, 2, 2, 1], 32, padding='SAME', name='conv1')
    # 14*14*32
    output = conv_layer(output, [3, 3], [1, 2, 2, 1], 64, padding='SAME', name='conv2')
    # 7*7*64
    output = tf.reshape(output, [BATCH_SIZE, -1])
    output = fully_connect(output, 100, name='fc1')
    output = fully_connect(output, 10, name='fc2', do_act=False)
    return output

BATCH_SIZE = 100
EPOCH = 100
X = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 28, 28, 1], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 10], name='Y')
output = build_network(X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

def train(sess):

    iter_num = int(mnist_data.train.num_examples / BATCH_SIZE)
    for e in range(EPOCH):
        epoch_loss = 0
        for i in range(iter_num):
            # x: [N, 784]
            x, y = mnist_data.train.next_batch(BATCH_SIZE)
            x = x.reshape([-1, 28, 28, 1])
            iter_loss, _ = sess.run([loss, optim], feed_dict={X: x, Y: y})
            epoch_loss += iter_loss
        print('EPOCH {}: {}'.format(e, epoch_loss))

def test(sess):
    iter_num = int(mnist_data.test.num_examples / BATCH_SIZE)
    correct = 0
    total = 0

    correct_num = tf.reduce_sum(tf.cast(
        tf.equal(
            tf.math.argmax(output, axis=-1), 
            tf.math.argmax(Y, axis=-1)
            ), tf.int8))

    for _ in range(iter_num):
        x, y = mnist_data.test.next_batch(BATCH_SIZE)
        x = x.reshape([BATCH_SIZE, 28, 28, 1])
        correct += sess.run(correct_num, feed_dict={X: x, Y: y})
        total += BATCH_SIZE
    print('ACC: ', correct / total)

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(sess)
        test(sess)
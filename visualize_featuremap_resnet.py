import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import math
import numpy as np

# read data
mnist_data = input_data.read_data_sets(r'D:\\', one_hot=True)

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


def resnet_building_block(x, kernel: list, strides: list, out_channel: int, padding: str, name):
    # conv --> batch norm --> activate
    with tf.variable_scope(name):
        output = conv_layer(x, kernel, strides, out_channel, padding, name='conv1', do_act=False)
        output = batch_norm(output, 'bn1')
        output = tf.nn.relu(output)

        output = conv_layer(output, kernel, [1, 1, 1, 1], out_channel, padding, name='conv2', do_act=False)
        output = batch_norm(output, 'bn2')

        if strides[2] != 1:
            _output = conv_layer(x, [1, 1], strides, out_channel, padding, name='conv_', do_act=False)
            _output = batch_norm(_output, '_bn')
        else:
            _output = x
        
        # print(_output.shape, output.shape)
        output += _output
        output = tf.nn.relu(output, name='relu')
        return output

def batch_norm(x, name):
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, axes=[0, 1, 2])
        output = tf.nn.batch_normalization(x, mean, var, 0, 1, name='bn', variance_epsilon=1e-3)
        return output

def build_network(x):
    # x: 28*28*1
    feature_maps = dict()
    
    output = conv_layer(x, [3, 3], [1, 1, 1, 1], 16, padding='SAME', name='conv1', do_act=False)
    feature_maps['conv1'] = output

    output = batch_norm(output, name='bn1')

    output = tf.nn.relu(output)

    output = resnet_building_block(output, [3, 3], [1, 1, 1, 1], 16, padding='SAME', name='res_block1')
    feature_maps['res1'] = output

    output = resnet_building_block(output, [3, 3], [1, 1, 1, 1], 16, padding='SAME', name='res_block2')
    feature_maps['res2'] = output
    # 28*28*16

    output = resnet_building_block(output, [3, 3], [1, 2, 2, 1], 32, padding='SAME', name='res_block3')
    feature_maps['res3'] = output
    # 14*14*32
    output = resnet_building_block(output, [3, 3], [1, 1, 1, 1], 32, padding='SAME', name='res_block4')
    feature_maps['res4'] = output
    # 14*14*32

    output = resnet_building_block(output, [3, 3], [1, 2, 2, 1], 64, padding='SAME', name='res_block5')
    feature_maps['res5'] = output

    # 7*7*64
    output = resnet_building_block(output, [3, 3], [1, 1, 1, 1], 64, padding='SAME', name='res_block6')
    feature_maps['res6'] = output
    # 7 7 64
    
    output = tf.reshape(output, [BATCH_SIZE, -1])
    output = fully_connect(output, 10, name='fc2', do_act=False)
    return output, feature_maps

BATCH_SIZE = 100
EPOCH = 200
X = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 28, 28, 1], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 10], name='Y')
output, feature_maps = build_network(X)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

saver = tf.train.Saver()

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
    saver.save('checkpoint/resnet.ckpt', sess)

def test(sess, is_training=True):
    iter_num = int(mnist_data.test.num_examples / BATCH_SIZE)
    correct = 0
    total = 0

    if is_training:
        ckpt = tf.train.latest_checkpoint('checkpoint/resnet')
        saver.restore(sess, ckpt)
    
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
    
    for k, v in feature_maps.items():
        kernel_map = sess.run(v, feed_dict={X: x, Y: y})
        # [k_h, k_w, in_ch_num, out_ch_num]
        # [k_h, k_w, 1]
        ch_num = kernel_map.shape[-1]
        rows = math.ceil(math.sqrt(ch_num))
        cols = math.ceil(ch_num / rows)
        
        figure, axes = plt.subplots(rows, cols)
        
        flag = False
        for i in range(rows):
            for j in range(cols):
                if i * rows + j >= ch_num:
                    flag = True
                    break
                
                axes[i][j].imshow(kernel_map[:, :, 0, i * rows + j], cmap='gray')
                
            if flag:
                break
        
        plt.suptitle(k)
        plt.show()

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(sess)
        test(sess, is_training=False)
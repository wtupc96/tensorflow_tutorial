import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import math
import numpy as np
import random

# read data
mnist_data = input_data.read_data_sets('../mis', one_hot=True)

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
        return output, w


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

def global_average_pooling(x, name):
    with tf.variable_scope(name):
        return tf.reduce_mean(x, axis=[1, 2])

def build_network(x):
    # x: 28*28*1
    output = conv_layer(x, [3, 3], [1, 1, 1, 1], 16, padding='SAME', name='conv1', do_act=False)

    output = batch_norm(output, name='bn1')

    output = tf.nn.relu(output)

    output = resnet_building_block(output, [3, 3], [1, 1, 1, 1], 16, padding='SAME', name='res_block1')

    output = resnet_building_block(output, [3, 3], [1, 1, 1, 1], 16, padding='SAME', name='res_block2')
    # 28*28*16

    output = resnet_building_block(output, [3, 3], [1, 2, 2, 1], 32, padding='SAME', name='res_block3')
    # 14*14*32
    output = resnet_building_block(output, [3, 3], [1, 1, 1, 1], 32, padding='SAME', name='res_block4')
    # 14*14*32

    output = resnet_building_block(output, [3, 3], [1, 2, 2, 1], 64, padding='SAME', name='res_block5')

    # 7*7*64
    last_conv_output = resnet_building_block(output, [3, 3], [1, 1, 1, 1], 64, padding='SAME', name='res_block6')
    # 7 7 64

    gap_output = global_average_pooling(last_conv_output, name='global_avg_pooling')
    output, w = fully_connect(gap_output, 10, name='fc2', do_act=False)

    return output, last_conv_output, w

BATCH_SIZE = 100
EPOCH = 10
X = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 28, 28, 1], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 10], name='Y')
output, last_conv_output, fc_w = build_network(X)

cam_weights = tf.gather(tf.transpose(fc_w), indices=tf.argmax(Y, axis=-1))

cam_weights = tf.stack([cam_weights] * 7, axis=1)
cam_weights = tf.stack([cam_weights] * 7, axis=2)

cam = tf.reduce_mean(cam_weights * last_conv_output, axis=-1)
cam = tf.expand_dims(cam, axis=-1)
print(cam.shape)
cam = tf.map_fn(lambda c: tf.image.resize(c, [28, 28]), cam)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
optim = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)

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
    saver.save(sess, 'checkpoint/resnet_cam.ckpt')

def test(sess, do_restore=True):
    iter_num = int(mnist_data.test.num_examples / BATCH_SIZE)
    correct = 0
    total = 0

    if do_restore:
        ckpt = tf.train.latest_checkpoint('checkpoint')
        print(ckpt)
        saver.restore(sess, ckpt)
    
    correct_num = tf.reduce_sum(tf.cast(
        tf.equal(
            tf.math.argmax(output, axis=-1), 
            tf.math.argmax(Y, axis=-1)
            ), tf.int8))

    ims = list()
    cams = list()

    for _ in range(iter_num):
        x, y = mnist_data.test.next_batch(BATCH_SIZE)
        x = x.reshape([BATCH_SIZE, 28, 28, 1])
        correct += sess.run(correct_num, feed_dict={X: x, Y: y})
        total += BATCH_SIZE
        cam_results = sess.run(cam, feed_dict={X: x, Y: y})
        ims.append(x)
        cams.append(cam_results)
    
    print('ACC: ', correct / total)
    
    ims = np.concatenate(ims, axis=0)
    ims = np.squeeze(ims)
    cams = np.concatenate(cams, axis=0)
    cams = np.squeeze(cams)
    row = col = 10

    random_idx = random.sample(range(ims.shape[0]), k=row * col)
    figure, axes = plt.subplots(row, col)

    for i in range(row):
        for j in range(col):
            rand_i = random_idx[i * row + j]
            axes[i][j].imshow(ims[rand_i])
            axes[i][j].imshow(cams[rand_i], alpha=0.6, cmap='rainbow')
    plt.savefig('fig.png')
    

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #train(sess)
        test(sess, do_restore=True)

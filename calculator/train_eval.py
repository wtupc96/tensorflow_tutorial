import os
import numpy as np
import tensorflow.compat.v1 as tf
import random
import cv2
from glob import glob
from tqdm import tqdm


tf.disable_eager_execution()

CKPT_PATH = r'checkpoint'

if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH)

BATCH_SIZE = 1000
BASE_LR = 0.1
CKPT_FILE = os.path.join(CKPT_PATH, r'sym.ckpt')
COEFF = 5

CLASS_ID_DICT = {str(i): i for i in range(15)}
SYM_IMGS_ROOT = r'data'
EPOCH = 500
NUM_CLASS = 15

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASS], name='y')
learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')


def get_sym_data():
    cls = CLASS_ID_DICT.keys()

    train_img_lst = []
    train_ids = []

    test_img_lst = []
    test_ids = []

    for c in cls:
        imgs = glob(os.path.join(SYM_IMGS_ROOT, c, '*.jpg'))
        print(c)
        for idx, i in tqdm(enumerate(imgs)):
            if idx < 1000:
                test_img_lst.append(np.reshape(cv2.imread(i, 0), (1, -1)))
                test_ids.append(CLASS_ID_DICT[c])
            else:
                train_img_lst.append(np.reshape(cv2.imread(i, 0), (1, -1)))
                train_ids.append(CLASS_ID_DICT[c])

    zipped_train_data = list(zip(train_img_lst, train_ids))
    random.shuffle(zipped_train_data)
    train_img_lst, train_ids = zip(*zipped_train_data)
    train_img_lst = list(train_img_lst)
    train_ids = list(train_ids)

    train_img_lst = np.concatenate(train_img_lst, axis=0)
    test_img_lst = np.concatenate(test_img_lst, axis=0)

    print(max(train_ids), max(test_ids), type(train_ids), type(test_ids))
    train_ids = np.eye(NUM_CLASS)[train_ids]
    test_ids = np.eye(NUM_CLASS)[test_ids]

    print(train_img_lst.shape, test_img_lst.shape)
    print(train_ids.shape, test_ids.shape)

    # train_mean = np.mean(train_img_lst)
    # train_std = np.sqrt(np.var(train_img_lst))

    # train_img_lst = (train_img_lst - train_mean) / train_std
    # test_img_lst = (test_img_lst - train_mean) / train_std

    train_img_lst = train_img_lst / 127.5 - 1
    test_img_lst = test_img_lst / 127.5 - 1

    return train_img_lst, train_ids, test_img_lst, test_ids


def add_layer(input_data, input_num, output_num, activation_function=None, name='fc'):
    with tf.variable_scope(name):
        w = tf.Variable(initial_value=tf.truncated_normal(shape=[
                        input_num, output_num], mean=0, stddev=np.sqrt(2 / np.prod(input_data.get_shape().as_list()[1:]))))
        b = tf.Variable(initial_value=tf.zeros(shape=[1, output_num]))

        output = tf.add(tf.matmul(input_data, w), b)

        output = tf.layers.batch_normalization(output)

        if activation_function:
            output = activation_function(output)

        return output


def build_nn(data):
    hidden_layer1 = add_layer(
        data, 784, 512, activation_function=tf.nn.leaky_relu, name='l1')
    hidden_layer2 = add_layer(hidden_layer1, 512, 256,
                              activation_function=tf.nn.leaky_relu, name='l2')
    output_layer = add_layer(hidden_layer2, 256, NUM_CLASS, name='l3')
    return output_layer


def get_lr(e):
    if e < 0.3 * EPOCH:
        return BASE_LR
    if e < 0.6 * EPOCH:
        return BASE_LR / COEFF
    if e < 0.8 * EPOCH:
        return BASE_LR / COEFF / COEFF
    return BASE_LR / COEFF / COEFF / COEFF


def train_eval_nn(imgs):
    output = build_nn(x)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)
    accuracy = tf.reduce_sum(
        tf.cast(tf.equal(tf.arg_max(y, 1), tf.arg_max(output, 1)), tf.float32))

    saver = tf.train.Saver()

    best_acc = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if not os.path.exists('checkpoint'):
            train_x, train_y, test_x, test_y = get_sym_data()

            for i in range(EPOCH):
                epoch_const = 0

                for idx in range(int(train_x.shape[0] / BATCH_SIZE)):
                    x_data, y_data = train_x[idx * BATCH_SIZE: (
                        idx + 1) * BATCH_SIZE], train_y[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]
                    
                    cost, _ = sess.run([loss, optimizer], feed_dict={
                                       x: x_data, y: y_data, learning_rate: get_lr(i)})
                    epoch_const += cost

                acc_num = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
                epoch_acc = acc_num / test_x.shape[0]

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    saver.save(sess, CKPT_FILE)

                print('Epoch {}:\t loss: {:.6}\t acc: {:.4}'.format(
                    i, epoch_const, epoch_acc))

            print('Best acc: {:.4}'.format(best_acc))
            print('Training done.')
            exit()
        else:
            saver.restore(sess, CKPT_FILE)
            return predict(sess, output, imgs)


def read_data(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed_image = cv2.resize(image, dsize=(28, 28))
    processed_image = np.resize(processed_image, new_shape=(1, 784))
    return image, processed_image


def predict(sess, output, img):
    img_ = np.reshape(img, (img.shape[0], -1))

    processed_image = img_ / 127.5 - 1

    result = sess.run(output, feed_dict={x: processed_image})
    result = np.argmax(result, 1)

    return result


if __name__ == "__main__":
    train_eval_nn(None)

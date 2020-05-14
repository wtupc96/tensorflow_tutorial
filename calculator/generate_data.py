import cv2
import numpy as np
import random
from glob import glob
import os
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image


data_root = r'data'


def reconstruct_images_0_9():
    mnist = input_data.read_data_sets(r'D:\\', one_hot=False)

    for i in range(10):
        p = os.path.join(data_root, i)
        if not os.path.exists(p):
            os.makedirs(p)

    BATCH_SIZE = 1
    cnt = 0

    for _ in range(int(mnist.train.num_examples / BATCH_SIZE)):
        x_data, y_data = mnist.train.next_batch(BATCH_SIZE)

        img = Image.fromarray(np.reshape(
            np.array(x_data[0] * 255, dtype='uint8'), newshape=(28, 28)))

        dir = np.argmax(y_data[0])

        img.save(os.path.join(data_root, dir, '{}.jpg'.format(cnt)))
        cnt += 1

    for _ in range(int(mnist.test.num_examples / BATCH_SIZE)):
        x_data, y_data = mnist.test.next_batch(BATCH_SIZE)

        img = Image.fromarray(np.reshape(
            np.array(x_data[0] * 255, dtype='uint8'), newshape=(28, 28)))
        dir = np.argmax(y_data[0])

        img.save(os.path.join(data_root, dir, '{}.jpg'.format(cnt)))
        cnt += 1

    for _ in range(int(mnist.validation.num_examples / BATCH_SIZE)):
        x_data, y_data = mnist.validation.next_batch(BATCH_SIZE)

        img = Image.fromarray(np.reshape(
            np.array(x_data[0] * 255, dtype='uint8'), newshape=(28, 28)))
        dir = np.argmax(y_data[0])

        img.save(os.path.join(data_root, dir, '{}.jpg'.format(cnt)))
        cnt += 1

    print(cnt)


def create_images_sym():
    folders = ['10', '11', '12', '13', '14']
    target_num = 7000

    for f in folders:
        chosen_images = glob(os.path.join(f, '*.jpg'))
        exist_num = len(chosen_images)

        for i in range(exist_num, target_num):
            ip = random.sample(chosen_images, k=1)[0]
            im = cv2.imread(ip, 0)

            coeff1 = 1 - (np.random.rand() * 0.4 - 0.2)
            coeff2 = 1 - (np.random.rand() * 0.4 - 0.2)

            M = np.array(
                [[coeff1, 1 - coeff2, np.random.rand()],
                 [1 - coeff1, coeff2, np.random.rand()]]
            )

            new_i = cv2.warpAffine(im, M, im.shape)
            cv2.imwrite(os.path.join(data_root, f, '{}.jpg'.format(i)), new_i)


if __name__ == "__main__":
    reconstruct_images_0_9()
    create_images_sym()

import cv2
import numpy as np
import random
from glob import glob
import os

folder = ['10', '11', '12', '13', '14']
target_num = 7000

for f in folder:
    chosen_images = glob(os.path.join(f, '*.jpg'))
    for i in range(2, target_num + 2):
        ip = random.sample(chosen_images, k=1)[0]
        im = cv2.imread(ip, 0)
        coeff1 = 1 - (np.random.rand() * 0.4 - 0.2)
        coeff2 = 1 - (np.random.rand() * 0.4 - 0.2)
        M = np.array(
            [[coeff1, 1 - coeff2, np.random.rand()], [1 - coeff1, coeff2, np.random.rand()]]
            )
        new_i = cv2.warpAffine(im, M, im.shape)

        cv2.imwrite('{}/{}.jpg'.format(f, i), new_i)

        # cv2.imshow('', new_i)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(i, coeff1, coeff2)

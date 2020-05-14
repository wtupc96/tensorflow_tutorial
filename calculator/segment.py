from skimage.measure import label
import cv2
import numpy as np


def segment(img_name):
    img = cv2.imread(img_name, 0)
    h, w = img.shape[: 2]

    _, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    img_label = label(thresh1, background=0, connectivity=2)

    num_field = np.max(img_label)
    print(num_field)

    cal_order = []

    for i in range(w):
        s = set(img_label[:, i])
        for _s in s:
            if _s > 0 and _s not in cal_order:
                cal_order.append(_s)

    all_imgs = list()

    for i in cal_order:
        new_img = np.zeros(shape=(min(w, h), min(w, h)))
        new_h, new_w = new_img.shape[: 2]

        pic_cord = np.where(img_label == i)

        out_img = thresh1[min(pic_cord[0]): max(pic_cord[0]),
                          min(pic_cord[1]): max(pic_cord[1])]
        out_h, out_w = out_img.shape[: 2]

        pad_h = (new_h - out_h) // 2
        pad_w = (new_w - out_w) // 2

        new_img[pad_h: pad_h + out_h, pad_w: pad_w + out_w] = out_img
        new_img = cv2.resize(new_img, (28, 28))
        all_imgs.append(new_img)

    all_imgs = np.array(all_imgs)

    return all_imgs


if __name__ == "__main__":
    img_name = 'test_img.jpg'
    segment(img_name)

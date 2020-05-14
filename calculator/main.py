import cv2
import numpy as np
from train_eval import train_eval_nn
from segment import segment

def main(img_name):
    segmented_images = segment(img_name)

    preds = train_eval_nn(segmented_images)
    preds = list(preds)

    for idx in range(len(preds)):
        if preds[idx] < 10:
            preds[idx] = str(preds[idx])
        if preds[idx] == 10:
            preds[idx] = '+'
        if preds[idx] == 11:
            preds[idx] = '-'
        if preds[idx] == 12:
            preds[idx] = '*'
        if preds[idx] == 13:
            preds[idx] = '/'
        if preds[idx] == 14:
            preds[idx] = '='
    
    print(preds)
    
    result = eval(''.join(preds))
    print(result)


if __name__ == "__main__":
    main(img_name)
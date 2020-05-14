import numpy as np
import cv2


CANVAS_SIZE = (128, 1280)

drawing = False


def write(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img[y: y + 10, x: x + 10] = 255
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


img = np.zeros((*CANVAS_SIZE, 1), np.uint8)
cv2.namedWindow('Write calculation')
cv2.setMouseCallback('Write calculation', write)

while True:
    cv2.imshow('Write calculation', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.imwrite('test_img.jpg', img)
        break
cv2.destroyAllWindows()

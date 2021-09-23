import cv2
import numpy as np


def preprocess(image, channels, h, w):
    if image.shape[:2] != (128, 128):
        image = cv2.resize(image, (128, 128))
        # center crop image
        a = int((128 - w) / 2)  # x start
        b = int((128 - w) / 2 + 112)  # x end
        c = int((128 - h) / 2)  # y start
        d = int((128 - h) / 2 + 112)  # y end
        cropped = image[a:b, c:d]  # center crop the image
        cropped = cropped[..., ::-1]  # BGR to RGB
        # flip image horizontally
        flipped = cv2.flip(cropped, 1)

        def to_format(image):
            image = image.swapaxes(1, 2).swapaxes(0, 1)
            image = np.reshape(image, [1, channels, w, h])
            image = np.array(image, dtype=np.float32)
            image = (image - 127.5) / 128.0
            return image

        return (to_format(cropped) + to_format(flipped)).astype(np.float32)

import numpy as np
from UNetModel import UNet
from Spacer import Spacer
import cv2


if __name__ == '__main__':
    # Initializing the model
    model = UNet()
    model.load_weights('model_weights/UNet_weights.pt')

    # Initialize BackgroundChanger object
    changer = Spacer(model)

    image = cv2.imread('images/face.jpg')
    replacer = cv2.imread('replacements/image_space_1.jpg')

    image = cv2.resize(image, (160, 160))
    replacer = cv2.resize(replacer, (160, 160))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Starting the program
    changed_image = changer(image, replacer).astype(np.uint8)
    changed_image = cv2.cvtColor(changed_image, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 512, 512)
    cv2.imshow('Image', changed_image)
    cv2.waitKey()

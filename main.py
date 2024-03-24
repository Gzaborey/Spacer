import numpy as np
from UNetModel import UNet
from Spacer import Spacer
import cv2
import time
import os


def main():
    """
    Press Q to stop the execution of a program.
    """
    # Initializing parameters
    resize_scale = 128

    # Initializing the model
    model = UNet()
    model.load_weights('model_weights/UNet_weights_128p.pt')

    # Initialize BackgroundChanger object
    changer = Spacer(model)

    # Initializing replacer
    replacements_files = os.listdir('replacements')
    replacer_path = os.path.join('replacements', np.random.choice(replacements_files))
    replacer = cv2.imread(replacer_path)
    replacer = cv2.resize(replacer, (resize_scale, resize_scale))
    replacer_rgb = cv2.cvtColor(replacer, cv2.COLOR_BGR2RGB)

    # Initialize variable for FPS tracking
    prev_frame_time = 0

    cap = cv2.VideoCapture(0)
    while True:
        success, image = cap.read()
        image = cv2.resize(image, (resize_scale, resize_scale))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Starting the program
        changed_image = changer(image_rgb, replacer_rgb).astype(np.uint8)
        changed_image = cv2.cvtColor(changed_image, cv2.COLOR_RGB2BGR)

        # Calculating the FPS
        # Getting the current frame timestamp
        new_frame_time = time.time()

        # Calculating the fps
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Converting the fps into integer
        fps = int(fps)

        # Making a frame for a video
        frame = cv2.hconcat([image, changed_image])

        # Putting the FPS count on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'FPS: {str(fps)}', (7, 15), font, 0.5,
                    (100, 255, 0), 2, cv2.LINE_AA)

        # Adjusting window size
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 1100, 500)

        cv2.imshow('Video', frame)

        # Press 'Q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
import cv2
import numpy as np
from skimage import io
from typing import Tuple
import process
import sounddevice as sd
import soundfile as sf


class Paint:
    def __init__(self, width: int = 512, height: int = 512):
        self.img = np.ones((height, width, 1), np.float32)*1e-6
        self.height = height
        self.width = width
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_circle)
        self.l_button_down = False
        self.r_button_down = False

    def run(self):
        print(f"Image Shape:\t{self.img.shape}")
        while True:
            cv2.imshow('image', self.img)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            print(key)
            if key in (27, ord('q')):
                print('Quitting')
                break
            if key == ord('h'):
                print(f"Min: {np.min(self.img)}, Max: {np.max(self.img)}")
            if key == ord('c'):
                print("Clipping Values Between 0 and 1")
                np.clip(self.img, 0, 1.0, out=self.img)
            if key == ord('r'):
                print("Clearing")
                self.img = np.ones((self.height, self.width, 1), np.float32)*1e-6
            if key == ord('n'):
                print("Normalizing")
                im_max = np.max(self.img)
                if im_max > 1e-5:
                    self.img /= im_max
            if key == ord('s'):
                print("Saving Magnitude Spectrogram Image")
                io.imsave("mag_spec.tif", self.img)
                np.save("mag_spec.npy", self.img)
            if key == ord('p'):
                print("Playing Audio")
                self.play_audio()
        cv2.destroyAllWindows()

    def draw_circle(self, event, x: int, y: int, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.l_button_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.l_button_down = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.r_button_down = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.r_button_down = False

        if self.l_button_down:
            self._modify_image(x, y, increase=True)
        elif self.r_button_down:
            self._modify_image(x, y, increase=False)

    def _modify_image(self, x: int, y: int, increase: bool):
        height, width = self.img.shape[:2]
        for i in range(-8, 8):
            for j in range(-8, 8):
                xi = (x + i) % width
                yi = (y + j) % height
                if increase:
                    self.img[yi, xi] += 0.01
                    self.img[yi, xi] *= 1.05
                else:
                    self.img[yi, xi] = 1e-6

    def play_audio(self):
        print("Inverting Spectrogram")
        waveform = process.waveform_from_spectrogram(np.flip(self.img[...,0], axis=0).copy())
        print("Inversion Done - Playing Audio")
        waveform = waveform/np.max(np.abs(waveform))
        sd.play(waveform, process.SAMPLE_RATE)
        sd.wait()
        print("Audio Finished Playing")

if __name__ == '__main__':
    paint = Paint()
    paint.run()
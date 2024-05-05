import re

import numpy as np
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
from pytesseract import pytesseract
from rapid_orientation import RapidOrientation


class OrientationCorrector:
    orientation_engine = RapidOrientation()

    @classmethod
    def adjust_rotation_angle(cls, img, img_bytes, page):
        orientation_res, _ = cls.orientation_engine(img_bytes)
        rotated_angle = int(orientation_res)
        img_rotated = img
        # rotate
        if rotated_angle in [90, 270]:
            img_rotated = img.rotate(rotated_angle, expand=True)
            new_rotation = (360 - (page.rotation - rotated_angle)) % 360
            page.set_rotation(new_rotation)
        return img_rotated, rotated_angle

    @classmethod
    def rotate_through_tesseract(cls, img, page):
        osd_data = pytesseract.image_to_osd(img)
        # use re to extract the orientation in degrees and the confidence
        degrees = int(re.search(r'Orientation in degrees: (\d+)', osd_data).group(1))
        confidence = float(re.search(r'Orientation confidence: (\d+\.\d+)', osd_data).group(1))
        img_rotated = img
        if degrees in [90, 270]:
            img_rotated = img.rotate(degrees, expand=True)
            new_rotation = (360 - (page.rotation - degrees)) % 360
            page.set_rotation(new_rotation)
        return img_rotated, degrees

    @classmethod
    def deskew_image(cls, img):
        image_array = np.array(img)
        angle = get_angle(image_array, angle_max=10)
        img_deskew = img
        if abs(angle) >= 1:
            img_deskew = rotate(img, angle)
        print(angle)
        return img_deskew, angle

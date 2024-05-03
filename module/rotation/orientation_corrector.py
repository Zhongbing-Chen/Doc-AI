import numpy as np
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
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
    def deskew_image(cls, img):
        image_array = np.array(img)
        angle = get_angle(image_array, angle_max=10)
        img_deskew = img
        if abs(angle) >= 1:
            img_deskew = rotate(img, angle)
        print(angle)
        return img_deskew, angle

import re

import PIL
import cv2
import numpy as np
from PIL.Image import Image
from jdeskew.estimator import get_angle
from jdeskew.utility import rotate
from pytesseract import pytesseract
from rapid_orientation import RapidOrientation


class OrientationCorrector:
    """
    The orientation corrector class

    Methods:
        adjust_rotation_angle: adjust the rotation angle of the image
    """
    orientation_engine = RapidOrientation()

    @classmethod
    def adjust_rotation_angle(cls, img, img_bytes, page) -> tuple:
        """
        Adjust the rotation angle of the image
        :param img: the image object
        :param img_bytes: the bytes of the image
        :param page: the page object
        :return: the rotated image and the rotated angle
        """
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
    def rotate_through_tesseract(cls, img, page) -> tuple:
        """
        Rotate the image through tesseract
        :param img: the image object
        :param page: the page object
        :return: the rotated image and the rotated angle
        """
        osd_data = pytesseract.image_to_osd(img)
        # use re to extract the orientation in degrees and the confidence
        degrees = int(re.search(r'Orientation in degrees: (\d+)', osd_data).group(1))

        # orientation confidence
        confidence = float(re.search(r'Orientation confidence: (\d+\.\d+)', osd_data).group(1))
        img_rotated = img
        if degrees in [90, 270]:
            # rotate the image based on the degrees
            img_rotated = img.rotate(degrees, expand=True)

            # calculate the new rotation
            new_rotation = (360 - (page.rotation - degrees)) % 360
            page.set_rotation(new_rotation)
        return img_rotated, degrees

    @classmethod
    def deskew_image(cls, img) -> tuple:
        """
        Deskew the image
        :param img: the image object
        :return: the deskewed image and the angle
        """
        image_array = np.array(img)
        angle = get_angle(image_array, angle_max=10)
        print("Deskew angle: ", angle)

        img_deskew = img
        if abs(angle) >= 0.3:
            cv2_image = cls.pil_to_cv2(img)
            img_deskew = rotate(cv2_image, angle)
            img_deskew = cls.cv2_to_pil(img_deskew)
        return img_deskew, angle

    @staticmethod
    def pil_to_cv2(pil_image):
        """
        将PIL图像转换为OpenCV格式

        Args:
            pil_image (PIL.Image.Image): PIL格式的图像

        Returns:
            numpy.ndarray: OpenCV格式的图像(BGR)
        """
        # 首先确保图像在RGB模式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # 转换为numpy数组
        image_array = np.array(pil_image)

        # 转换RGB为BGR
        opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        return opencv_image

    @staticmethod
    def cv2_to_pil(cv2_image):
        """
        将OpenCV格式图像转换为PIL格式

        Args:
            cv2_image (numpy.ndarray): OpenCV格式的图像(BGR)

        Returns:
            PIL.Image.Image: PIL格式的图像
        """
        # 转换BGR为RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # 转换为PIL Image
        pil_image = PIL.Image.fromarray(rgb_image.astype('uint8'))

        return pil_image

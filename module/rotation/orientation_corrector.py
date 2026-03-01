"""
Orientation corrector for standalone document parser
"""
import re

import PIL
import cv2
import numpy as np
from PIL.Image import Image


class OrientationCorrector:
    """
    The orientation corrector class for standalone document parser.
    """

    @classmethod
    def adjust_rotation_angle(cls, img, img_bytes, page) -> tuple:
        """
        Adjust the rotation angle using rapid_orientation
        :param img: the image object
        :param img_bytes: the bytes of the image
        :param page: the page object
        :return: the rotated image and the rotated angle
        """
        try:
            from rapid_orientation import RapidOrientation
            orientation_engine = RapidOrientation()
            orientation_res, _ = orientation_engine(img_bytes)
            rotated_angle = int(orientation_res)
            img_rotated = img
            if rotated_angle in [90, 270]:
                img_rotated = img.rotate(rotated_angle, expand=True)
                new_rotation = (360 - (page.rotation - rotated_angle)) % 360
                page.set_rotation(new_rotation)
            return img_rotated, rotated_angle
        except ImportError:
            # rapid_orientation not available, return original image
            return img, 0

    @classmethod
    def rotate_through_tesseract(cls, img, page) -> tuple:
        """
        Rotate the image through tesseract
        :param img: the image object
        :param page: the page object
        :return: the rotated image and the rotated angle
        """
        try:
            from pytesseract import pytesseract
            osd_data = pytesseract.image_to_osd(img)
            degrees = int(re.search(r'Orientation in degrees: (\d+)', osd_data).group(1))
            img_rotated = img
            if degrees in [90, 270]:
                img_rotated = img.rotate(degrees, expand=True)
                new_rotation = (360 - (page.rotation - degrees)) % 360
                page.set_rotation(new_rotation)
            return img_rotated, degrees
        except ImportError:
            # Tesseract not available, return original image
            return img, 0

    @classmethod
    def deskew_image(cls, img) -> tuple:
        """
        Deskew the image
        :param img: the image object
        :return: the deskewed image and the angle
        """
        try:
            from jdeskew.estimator import get_angle
            from jdeskew.utility import rotate
            image_array = np.array(img)
            angle = get_angle(image_array, angle_max=10)
            print("Deskew angle: ", angle)

            img_deskew = img
            if abs(angle) >= 0.3:
                cv2_image = cls.pil_to_cv2(img)
                img_deskew = rotate(cv2_image, angle)
                img_deskew = cls.cv2_to_pil(img_deskew)
            return img_deskew, angle
        except ImportError:
            # jdeskew not available, return original image
            return img, 0.0

    @staticmethod
    def pil_to_cv2(pil_image):
        """
        Convert PIL image to OpenCV format

        Args:
            pil_image (PIL.Image.Image): PIL image

        Returns:
            numpy.ndarray: OpenCV image (BGR)
        """
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image_array = np.array(pil_image)
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(cv2_image):
        """
        Convert OpenCV image to PIL format

        Args:
            cv2_image (numpy.ndarray): OpenCV image (BGR)

        Returns:
            PIL.Image.Image: PIL image
        """
        image_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_array)

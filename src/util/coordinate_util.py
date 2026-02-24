import fitz
from fitz import Rect


class CoordinateUtil:
    """
    The coordinate utility class

    Methods:
        adjust_coordinate: adjust the coordinate based on the zoom factor
        iob: compute the intersection area over box area
    """

    @staticmethod
    def adjust_coordinate(bbox, zoom_factor) -> list:
        """
        Adjust the coordinate based on the zoom factor
        :param bbox: the bounding box
        :param zoom_factor: the zoom factor
        :return: the adjusted bounding box
        """
        x_1, y_1, x_2, y_2 = bbox
        return [x_1.item() / zoom_factor, y_1.item() / zoom_factor, x_2.item() / zoom_factor, y_2.item() / zoom_factor]

    @staticmethod
    def iob(bbox1, bbox2) -> tuple:
        """
        Compute the intersection area over box area, for bbox1.
        :param bbox1: the bounding box 1
        :param bbox2: the bounding box 2
        :return: the intersection area over box area for bbox1 and bbox2
        """

        # Compute the intersection area over box area for bbox1 and bbox2
        intersection = Rect(bbox1).intersect(bbox2)

        # Compute the area of bbox1, and the intersection area over box area for bbox1
        bbox1_area = Rect(bbox1).get_area()
        bbox1_iob_bbox2 = 0
        if bbox1_area > 0:
            bbox1_iob_bbox2 = intersection.get_area() / bbox1_area

        # Compute the area of bbox2, and the intersection area over box area for bbox2
        bbox2_area = Rect(bbox2).get_area()
        bbox2_iob_bbox1 = 0
        if bbox2_area > 0:
            bbox2_iob_bbox1 = intersection.get_area() / bbox2_area

        return bbox1_iob_bbox2, bbox2_iob_bbox1

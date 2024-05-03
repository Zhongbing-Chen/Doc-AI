import fitz
from fitz import Rect


class CoordinateUtil:
    @staticmethod
    def adjust_coordinate(bbox, zoom_factor):
        x_1, y_1, x_2, y_2 = bbox
        return [x_1.item() / zoom_factor, y_1.item() / zoom_factor, x_2.item() / zoom_factor, y_2.item() / zoom_factor]

    @staticmethod
    def iob(bbox1, bbox2):
        """
        Compute the intersection area over box area, for bbox1.
        """
        intersection = Rect(bbox1).intersect(bbox2)

        bbox1_area = Rect(bbox1).get_area()
        bbox1_iob_bbox2 = 0
        if bbox1_area > 0:
            bbox1_iob_bbox2 = intersection.get_area() / bbox1_area

        bbox2_area = Rect(bbox2).get_area()
        bbox2_iob_bbox1 = 0
        if bbox2_area > 0:
            bbox2_iob_bbox1 = intersection.get_area() / bbox2_area

        return bbox1_iob_bbox2, bbox2_iob_bbox1

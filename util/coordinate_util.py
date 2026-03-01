"""Coordinate utility for document parser"""


class CoordinateUtil:
    """Utility class for coordinate operations"""

    @staticmethod
    def iob(box_a, box_b):
        """
        Calculate Intersection over Box (IOB) for two boxes.

        Returns:
            tuple: (ratio_a, ratio_b) where:
                - ratio_a = overlap_area / box_a_area
                - ratio_b = overlap_area / box_b_area
        """
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        if x_right > x_left and y_bottom > y_top:
            overlap_area = (x_right - x_left) * (y_bottom - y_top)
        else:
            overlap_area = 0

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        ratio_a = overlap_area / box_a_area if box_a_area > 0 else 0
        ratio_b = overlap_area / box_b_area if box_b_area > 0 else 0

        return ratio_a, ratio_b

    @staticmethod
    def iou(box_a, box_b):
        """
        Calculate Intersection over Union (IOU) for two boxes.

        Returns:
            float: IOU value between 0 and 1
        """
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        if x_right > x_left and y_bottom > y_top:
            overlap_area = (x_right - x_left) * (y_bottom - y_top)
        else:
            overlap_area = 0

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        union_area = box_a_area + box_b_area - overlap_area

        return overlap_area / union_area if union_area > 0 else 0

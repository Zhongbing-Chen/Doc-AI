class CoordinateUtil:
    @staticmethod
    def adjust_coordinate(bbox, zoom_factor):
        x_1, y_1, x_2, y_2 = bbox
        return [x_1.item() / zoom_factor, y_1.item() / zoom_factor, x_2.item() / zoom_factor, y_2.item() / zoom_factor]

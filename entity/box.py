class Box:
    x_1: float
    y_1: float
    x_2: float
    y_2: float
    content: str


class OcrBlock(Box):
    """
    The OcrBlock class represents a block of text recognized by an OCR (Optical Character Recognition) system.

    Attributes:
        x_1 (float): The x-coordinate of the top-left corner of the bounding box.
        y_1 (float): The y-coordinate of the top-left corner of the bounding box.
        x_2 (float): The x-coordinate of the bottom-right corner of the bounding box.
        y_2 (float): The y-coordinate of the bottom-right corner of the bounding box.
        content (str): The recognized text within the bounding box.
        confidence (float): The confidence score of the recognition result.
    """

    def __init__(self, x1, y1, x2, y2, text, confidence, raw_ocr_result=None):
        """
        Initializes an OcrBlock instance.

        Args:
            x1 (float): The x-coordinate of the top-left corner of the bounding box.
            y1 (float): The y-coordinate of the top-left corner of the bounding box.
            x2 (float): The x-coordinate of the bottom-right corner of the bounding box.
            y2 (float): The y-coordinate of the bottom-right corner of the bounding box.
            text (str): The recognized text within the bounding box.
            confidence (float): The confidence score of the recognition result.
        """
        self.x_1 = x1
        self.y_1 = y1
        self.x_2 = x2
        self.y_2 = y2
        self.content = text
        self.confidence = confidence
        self.raw_ocr_result = raw_ocr_result

    def __repr__(self):
        return f"OcrBlock(x1={self.x_1}, y1={self.y_1}, x2={self.x_2}, y2={self.y_2}, text='{self.content}', confidence={self.confidence})"

    @classmethod
    def from_rapid_ocr(cls, ocr_result):
        """Extract OCR text blocks with their coordinates and content"""
        ocr_blocks = []
        for block in ocr_result:
            coords = block[0]
            text = block[1]
            confidence = block[2]

            x1 = min(coord[0] for coord in coords)
            y1 = min(coord[1] for coord in coords)
            x2 = max(coord[0] for coord in coords)
            y2 = max(coord[1] for coord in coords)

            ocr_blocks.append(
                OcrBlock(x1=x1, x2=x2, y1=y1, y2=y2, text=text, confidence=confidence, raw_ocr_result=block))

        return ocr_blocks

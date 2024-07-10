import numpy as np
from cv2 import (
    ADAPTIVE_THRESH_MEAN_C,
    COLOR_RGB2GRAY,
    THRESH_BINARY_INV,
    adaptiveThreshold,
    threshold,
    THRESH_BINARY,
    THRESH_OTSU,
    Canny,
    GaussianBlur,
    cvtColor,
    dilate,
    erode,
    bilateralFilter,
    medianBlur,
    HoughLinesP,
    line
)

# from craft import craft_mask

from .contours import get_contours
from .debug_utils import debug_show
from .options import cfg

__all__ = ["box", "Mask"]


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


class Mask:
    def __init__(self, name, small, pagemask, text=True):
        self.name = name
        self.small = small
        self.pagemask = pagemask
        self.text = text
        self.calculate()

    def calculate(self):


        # mask_2 = craft_mask(self.small)
        # mask_2 = np.uint8(mask_2)

        self.text = False
        sgray = cvtColor(self.small, COLOR_RGB2GRAY)
        sgray = GaussianBlur(src=sgray, ksize=(5, 5), sigmaX=0.5)
        mask = Canny(sgray, 30, 150)

        # lines = HoughLinesP(
        #     mask, # Input edge image
        #     1, # Distance resolution in pixels
        #     np.pi/180, # Angle resolution in radians
        #     threshold=100, # Min number of votes for valid line
        #     minLineLength=5, # Min allowed length of line
        #     maxLineGap=10 # Max allowed gap between line for joining them
        # )

        # # Iterate over points
        
        # line_img = np.zeros(self.small.shape, dtype=np.uint8)
        # for points in lines:
        #     # Extracted points nested in the list
        #     x1,y1,x2,y2=points[0]
        #     # Draw the lines joing the points
        #     # On the original image
        #     line(line_img,(x1,y1),(x2,y2),(255,255,255),2)

        #     # mask = adaptiveThreshold(
        #     #     src=sgray,
        #     #     maxValue=255,
        #     #     adaptiveMethod=ADAPTIVE_THRESH_MEAN_C,
        #     #     thresholdType=THRESH_BINARY_INV,
        #     #     blockSize=cfg.mask_opts.ADAPTIVE_WINSZ,
        #     #     C=25 if self.text else 7,
        
        # line_img = np.uint8(line_img)
        # line_img = cvtColor(line_img, COLOR_RGB2GRAY)
        # mask = line_img
        # """
        self.log(0.1, "thresholded", mask)
        mask = (
            dilate(mask, box(9, 1))
            if self.text
            else erode(mask, box(3, 1), iterations=3)
        )
        self.log(0.2, "dilated" if self.text else "eroded", mask)
        mask = erode(mask, box(1, 3)) if self.text else dilate(mask, box(8, 2))
        self.log(0.3, "eroded" if self.text else "dilated", mask)
        # """

        # mask = medianBlur(mask, 3)
        # mask = (mask + mask_2)
        # mask = np.uint8(mask)
        # mask = mask_2
        # mask = dilate(mask, box(8, 2))

        self.log(0.4, "final before contour", mask)
    

        self.value = np.minimum(mask, self.pagemask)

    def log(self, step, text, display):
        if cfg.debug_lvl_opt.DEBUG_LEVEL >= 3:
            if not self.text:
                step += 0.3  # text images from 0.1 to 0.3, table images from 0.4 to 0.6
            debug_show(self.name, step, text, display)

    def contours(self):
        return get_contours(self.name, self.small, self.value)

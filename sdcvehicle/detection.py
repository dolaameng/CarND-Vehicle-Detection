from . import config
from . import classification

class VehicleDetector(object):
    def __init__(self):
        # self.vehicle_model = classification.build_model()
        pass
    def slide_window(self, image, rows, cols, window_size, stride):
        """Slide a window through an image and return all the patches.
        `image`: image to slide through
        `rows`: tuple of (start_row, end_row)
        `cols`: tuple of (start_col, end_col)
        `window_size`: (nrow, ncol)
        `stride`: (row_stride, col_stride)
        """
        start_row, end_row = rows
        start_col, end_col = cols
        stride_row, stride_col = stride
        wdow_nr, wdow_nc = window_size
        img_nr, img_nc = image.shape[:2] 
        for r in range(start_row, end_row+1, stride_row):
            for c in range(start_col, end_col+1, stride_col):
                # print(r, c)
                if r+wdow_nr > img_nr:
                    # print(r, c)
                    break
                if c+wdow_nc > img_nc: continue
                patch = image[r:r+wdow_nr, c:c+wdow_nc]
                bbox = [(r, c), (r+wdow_nr, c+wdow_nc)]
                yield (patch, bbox)
    def get_image_pyramid(self, image, scale=1.5, min_rc=(64, 64)):
        min_r, min_c = min_rc
        factor = 1
        yield image, factor
        while True:
            factor *= scale
            img = imresize(image, 1/factor, )
            if img.shape[0] < min_r or img.shape[1] < min_c:
                break
            yield img, factor
    def detect_in_image(image):
        pass
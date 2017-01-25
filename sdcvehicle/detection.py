from . import config
from . import classification

from scipy.misc import imresize
import numpy as np

class VehicleDetector(object):
    def __init__(self):
        """Main control parameters for balance between recall vs precision:
        - stride
        - scale
        """
        # vehicle classification model
        print("building vehicle classification model")
        self.vehicle_model = classification.build_model()
        # parameters for sliding window
        self.window_size = (64, 64)
        self.rows = (0.5, 1)
        self.cols = (0.15, 1)
        self.stride = (32, 32)
        # parameters for layer pyramid
        self.scale = 1.1

    def slide_window(self, image, 
        rows=(0, 1), cols=(0, 1), 
        window_size=(64, 64), stride=(8, 8), scale=1):
        """Slide a window through an image and return all the patches.
        `image`: image to slide through
        `rows`: tuple of (start_row, end_row)
        `cols`: tuple of (start_col, end_col)
        `window_size`: (nrow, ncol)
        `stride`: (row_stride, col_stride)
        `scale`: scale of pyramid layer from original. used to scale stride
        accordingly.
        """
        wdow_nr, wdow_nc = window_size
        img_nr, img_nc = image.shape[:2] 

        start_row, end_row = int(img_nr*rows[0]), int(img_nr*rows[1])
        start_col, end_col = int(img_nc*cols[0]), int(img_nc*cols[1])
        stride_row, stride_col = stride
        stride_row = max(1, int(stride_row / scale))
        stride_col = max(1, int(stride_col / scale))
        # print (stride_row, stride_col, scale)
        
        for r in range(start_row, end_row+1, stride_row):
            for c in range(start_col, end_col+1, stride_col):
                if r+wdow_nr > img_nr:
                    break
                if c+wdow_nc > img_nc: continue
                patch = image[r:r+wdow_nr, c:c+wdow_nc]
                bbox = [(c, r), (c+wdow_nc, r+wdow_nr)]
                yield (patch, bbox)
    def get_image_pyramid(self, image, scale=1.25, min_rc=(256, 256)):
        """Get a pyramid of layers from original images by downscaling at each step 
        at a fixed scale factor. 
        It serves as a generator that returns layer_image and scale_factor at each step.
        `scale`: downscale rate for each layer
        `min_rc`: minimum # of rows and columns at the very top of pyramid.
        """
        min_r, min_c = min_rc
        multiple = 1
        yield image, multiple
        while True:
            multiple *= scale
            img = imresize(image, 1/multiple, )
            if img.shape[0] < min_r or img.shape[1] < min_c:
                break
            yield img, multiple
    def get_pyramid_slide_window(self, image, pyramid_params=None, window_params=None):
        """Combine the generator of pyramid of layers and sliding window on each layer, 
        It returns a new generator that yields an image patch, its bounding box in original 
        image and its scaling factor from the original.
        """
        pyramid_params = pyramid_params or {}
        window_params = window_params or {}
        for layer, multiple in self.get_image_pyramid(image, **pyramid_params):
            window_params.update({"scale": multiple})
            for patch, [(w0, h0), (w1, h1)] in self.slide_window(layer, **window_params):
                bbox_in_original = [(int(w0*multiple), int(h0*multiple)), 
                            (int(w1*multiple), int(h1*multiple))]
                yield (patch, bbox_in_original, multiple)
    def preprocess(self, image):
        """Preprocess Image, e.g., resizing
        """
        return image
    def merge_boxes(self, image, bboxes):
        """Merge boxes by creating a heatmap.
        """
        heatmap = np.zeros(image.shape[:2])
        for bbox in bboxes:
            (c1, r1), (c2, r2) = bbox
            heatmap[r1:r2,c1:c2] += 1
        return heatmap

    def detect_in_image(self, image):
        """Detect vehicles in an image.
        It returns a list of bboxes for each vehicle in the original image.
        """
        image = self.preprocess(image)
        pyramid_params = {"scale": self.scale}
        window_params = {"rows": self.rows, "cols": self.cols, "window_size": self.window_size, "stride": self.stride}
        patches, bboxes, scales = zip(*list(self.get_pyramid_slide_window(image, pyramid_params, window_params)))
        is_vehicle = (self.vehicle_model.predict(np.array(patches)) == "vehicle")
        vehicle_bboxes = np.array(bboxes)[is_vehicle]
        return self.merge_boxes(image, vehicle_bboxes)
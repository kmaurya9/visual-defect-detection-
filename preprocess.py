import cv2
import numpy as np
from PIL import Image

def preprocess(pil_image):
    img = np.array(pil_image)

    # 1. Gaussian blur — reduce noise before edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. Canny edge detection — highlights cracks, scratches, boundaries
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # 3. Adaptive thresholding — isolates regions of interest
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 4. Stack: original RGB + edges + threshold as extra signal
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    thresh_3ch = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    combined = cv2.addWeighted(img, 0.7, edges_3ch, 0.2, 0)
    combined = cv2.addWeighted(combined, 1.0, thresh_3ch, 0.1, 0)

    return Image.fromarray(combined)

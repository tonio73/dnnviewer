from io import BytesIO
from PIL import Image
import base64
import numpy as np


def array_to_img_src(array):
    """ Convert image in a ndarray to inline png string to embed into an HTML IMG src"""

    mem_file = BytesIO()

    pil_img = Image.fromarray(array)
    if array.ndim == 2:
        # Grayscale handling
        pil_img = pil_img.convert("L")

    pil_img.save(mem_file, 'png')
    mem_file.seek(0)
    encoded_image = base64.b64encode(mem_file.getbuffer())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


def to_8bit_img(img):
    """ Normalize an image to span the unsigned 8bit range (0..255)"""
    imax = img.max()
    imin = img.min()
    delta = imax - imin

    if delta > 1e-8:
        img = (img - imin) / delta
    else:
        img = img - imin
    return (img * 255).astype(np.uint8)

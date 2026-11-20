import os
import tempfile

import cv2
import numpy as np

from ..generic import imwrite_unicode
from .kumikolib import Kumiko


def _as_decoded_color_image(img):
    """Return what ``cv2.imdecode(png, IMREAD_COLOR)`` would yield for ``img``.

    Only the lossless uint8 cases are handled in memory; anything else returns
    ``None`` so the caller keeps the original PNG round trip.
    """
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.size == 0:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim != 3:
        return None
    channels = img.shape[2]
    if channels == 3:
        return np.ascontiguousarray(img)
    if channels == 4:
        # IMREAD_COLOR strips the alpha channel without compositing.
        return np.ascontiguousarray(img[:, :, :3])
    if channels == 1:
        return cv2.cvtColor(np.ascontiguousarray(img[:, :, 0]), cv2.COLOR_GRAY2BGR)
    return None


def get_panels_from_array(img_rgb, rtl=True, logger=None):
    k = Kumiko({'rtl': rtl})

    in_memory = _as_decoded_color_image(img_rgb)
    if in_memory is not None:
        # Same pixels the PNG round trip below would produce, without
        # encoding/writing/decoding a temp file for every page.
        k.parse_array(in_memory)
        return k.get_infos()[0]['panels']

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    path = tmp.name
    tmp.close()

    # Always use the unicode-safe writer.
    # If no logger is passed, create a default one.
    if not logger:
        import logging
        logger = logging.getLogger(__name__)
    try:
        imwrite_unicode(path, img_rgb, logger)
        k.parse_image(path)
        infos = k.get_infos()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    return infos[0]['panels']

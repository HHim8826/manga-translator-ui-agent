import cv2
import numpy as np
import pytest

from manga_translator.utils.panel import _as_decoded_color_image, get_panels_from_array
from manga_translator.utils.panel.lib.segment import Segment


def _png_round_trip(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _synthetic_page():
    img = np.full((600, 420, 3), 255, np.uint8)
    cv2.rectangle(img, (10, 10), (200, 280), (0, 0, 0), 4)
    cv2.rectangle(img, (220, 10), (410, 280), (0, 0, 0), 4)
    cv2.rectangle(img, (10, 300), (410, 590), (0, 0, 0), 4)
    cv2.putText(img, "abc", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2)
    return img


@pytest.mark.parametrize(
    "convert",
    [
        lambda im: im,
        lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
        lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2BGRA),
        lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, :, None],
    ],
)
def test_in_memory_image_matches_png_round_trip(convert):
    img = convert(_synthetic_page())
    assert np.array_equal(_as_decoded_color_image(img), _png_round_trip(img))


def test_non_uint8_images_fall_back_to_file_path():
    assert _as_decoded_color_image(np.zeros((8, 8, 3), np.float32)) is None


@pytest.mark.parametrize("rtl", [True, False])
def test_panels_detected_from_memory(rtl):
    panels = get_panels_from_array(_synthetic_page(), rtl=rtl)
    assert len(panels) == 3


def test_segment_key_matches_equality():
    s1 = Segment((1, 2), (5, 9))
    s2 = Segment((5, 9), (1, 2))
    s3 = Segment((1, 2), (5, 8))
    assert s1 == s2 and s1._key() == s2._key()
    assert s1 != s3 and s1._key() != s3._key()


def test_union_all_merges_collinear_segments():
    merged = Segment.union_all([Segment((0, 0), (100, 0)), Segment((80, 0), (200, 0))])
    assert [(s.a, s.b) for s in merged] == [((0, 0), (200, 0))]

"""Check actual source pixels, not just crop sizes, against labelled targets."""

import os

import numpy as np
import pytest

from crop_test_data import BATCH_CASES, CROP_CASES, SAMPLING, coordinate_image
from turbojpeg import (
    TurboJPEG, TJCS_RGB, TJCS_YCbCr, TJPF_RGB, TJPF_GRAY,
    TJSAMP_444, TJSAMP_420, TJFLAG_FASTUPSAMPLE,
)


@pytest.fixture(scope='module')
def jpeg():
    return TurboJPEG(os.environ.get('TURBOJPEG_LIB_PATH'))


@pytest.fixture(scope='module', params=[8, 12], ids=['8bit', '12bit'])
def codec(jpeg, request):
    bits = request.param
    encode = jpeg.encode if bits == 8 else jpeg.encode_12bit
    decode = jpeg.decode if bits == 8 else jpeg.decode_12bit
    return bits, encode, decode


@pytest.fixture(scope='module', params=SAMPLING, ids=lambda item: item[0])
def source(codec, request):
    bits, encode, decode = codec
    _, subsample, mcu = request.param
    encoded = encode(coordinate_image(bits=bits), pixel_format=TJPF_RGB,
                     quality=100, jpeg_subsample=subsample)
    # Nearest-neighbor chroma upsampling removes the legitimate dependency on
    # pixels outside a crop at its edges. Exact equality then tests every RGB
    # channel even with subsampled JPEGs. Grayscale independently tests luma.
    pixels = {
        gray: decode(encoded, pixel_format=TJPF_GRAY if gray else TJPF_RGB,
                     flags=TJFLAG_FASTUPSAMPLE)
        for gray in (False, True)
    }
    return encoded, mcu, pixels


def assert_rectangle(jpeg, decode, encoded, source_pixels, rectangle, gray=False):
    x, y, width, height = rectangle
    assert jpeg.decode_header(encoded)[:2] == (width, height)
    actual = decode(encoded, pixel_format=TJPF_GRAY if gray else TJPF_RGB,
                    flags=TJFLAG_FASTUPSAMPLE)
    np.testing.assert_array_equal(
        actual, source_pixels[y:y + height, x:x + width])


@pytest.mark.parametrize('case', CROP_CASES, ids=lambda case: case.name)
def test_crop_matches_explicit_source_rectangle(jpeg, codec, source, case):
    _, _, decode = codec
    encoded, mcu, pixels = source
    expected = case.expected[mcu]
    for gray in (False, True):
        if expected is None:
            with pytest.raises(ValueError, match='after iMCU alignment'):
                jpeg.crop(encoded, *case.requested, preserve=case.preserve, gray=gray)
        else:
            result = jpeg.crop(encoded, *case.requested,
                               preserve=case.preserve, gray=gray)
            assert_rectangle(jpeg, decode, result, pixels[gray], expected, gray)


def test_batch_crops_match_source_rectangles_in_order(jpeg, codec, source):
    _, _, decode = codec
    encoded, _, pixels = source
    for gray in (False, True):
        results = jpeg.crop_multiple(encoded, [item[0] for item in BATCH_CASES], gray=gray)
        assert len(results) == len(BATCH_CASES)
        for result, (_, expected) in zip(results, BATCH_CASES):
            assert_rectangle(jpeg, decode, result, pixels[gray], expected, gray)


def test_issue_88_left_bar_crop_preserves_right_and_bottom_edges(jpeg, codec):
    """Remove 16 left pixels from a 505x731 JPEG without trimming other edges."""
    bits, encode, decode = codec
    image = coordinate_image(width=505, height=731, bits=bits)
    image[:, :16] = 0
    encoded = encode(image, quality=100, pixel_format=TJPF_RGB,
                     jpeg_subsample=TJSAMP_420)
    assert jpeg.decode_header(encoded) == (505, 731, TJSAMP_420, TJCS_YCbCr)

    cropped = jpeg.crop(encoded, 16, 0, 489, 731, preserve=True)
    assert jpeg.crop_multiple(encoded, [(16, 0, 489, 731)]) == [cropped]
    assert jpeg.crop_multiple(encoded, [(16, 0, 0, 0)]) == [cropped]

    # Both dimensions end in a partial iMCU. Issue #88 incorrectly produced
    # 480x720 instead of 489x731; verify the full source rectangle pixel by pixel.
    pixels = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    assert_rectangle(jpeg, decode, cropped, pixels, (16, 0, 489, 731))


def test_batch_does_not_silently_align_origins(jpeg, source):
    encoded, _, _ = source
    for x, y in ((1, 0), (0, 1), (1, 1)):
        with pytest.raises(ValueError, match='aligned'):
            jpeg.crop_multiple(encoded, [(x, y, 2, 2)])


@pytest.mark.parametrize('case', [case for case in CROP_CASES if case.name in (
    'tiny_expand', 'interior_preserve', 'clamp_edges_preserve', 'whole_image',
)], ids=lambda case: case.name)
def test_rgb_storage_crop_matches_all_source_channels(jpeg, codec, case):
    bits, encode, decode = codec
    encoded = encode(coordinate_image(bits=bits), pixel_format=TJPF_RGB,
                     jpeg_colorspace=TJCS_RGB, jpeg_subsample=TJSAMP_444, quality=100)
    pixels = decode(encoded, pixel_format=TJPF_RGB)
    result = jpeg.crop(encoded, *case.requested, preserve=case.preserve)
    assert_rectangle(jpeg, decode, result, pixels, case.expected[(8, 8)])


@pytest.mark.parametrize(('colorspace', 'subsample'), [
    (TJCS_RGB, TJSAMP_444), (TJCS_YCbCr, TJSAMP_420),
], ids=['rgb', 'ycbcr420'])
@pytest.mark.parametrize('luminance', [0.0, 1.0], ids=['black', 'white'])
@pytest.mark.parametrize(('region', 'source_rect', 'placement'), [
    ((32, 32, 320, 256), (0, 0, 256, 192), (32, 32)),
    ((32, 32, 320, 128), (0, 32, 256, 128), (32, 0)),
    ((32, 32, 192, 256), (32, 0, 192, 192), (0, 32)),
], ids=['extend_both', 'extend_x_crop_y', 'crop_x_extend_y'])
def test_extension_places_entire_coordinate_grid(
        jpeg, codec, colorspace, subsample, luminance, region, source_rect, placement):
    bits, encode, decode = codec
    encoded = encode(coordinate_image(256, 192, bits), pixel_format=TJPF_RGB,
                     jpeg_colorspace=colorspace, jpeg_subsample=subsample, quality=100)
    source = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    result = jpeg.crop_multiple(encoded, [region], background_luminance=luminance)[0]
    actual = decode(result, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    _, _, width, height = region
    sx, sy, sw, sh = source_rect
    px, py = placement
    assert jpeg.decode_header(result)[:2] == (width, height)
    np.testing.assert_array_equal(
        actual[py:py + sh, px:px + sw], source[sy:sy + sh, sx:sx + sw])
    background = np.ones((height, width), dtype=bool)
    background[py:py + sh, px:px + sw] = False
    np.testing.assert_allclose(actual[background], luminance * ((1 << bits) - 1), atol=1)

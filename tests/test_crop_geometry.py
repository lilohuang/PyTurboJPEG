"""Check actual source pixels, not just crop sizes, against labelled targets."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from crop_test_data import BATCH_CASES, CROP_CASES, SAMPLING, coordinate_image
from turbojpeg import (
    TurboJPEG, TJCS_RGB, TJCS_YCbCr, TJCS_GRAY, TJCS_CMYK, TJCS_YCCK,
    TJPF_RGB, TJPF_GRAY, TJPF_CMYK, TJSAMP_444, TJSAMP_422, TJSAMP_420,
    TJSAMP_GRAY, TJSAMP_441, TJFLAG_FASTUPSAMPLE,
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
@pytest.mark.parametrize('sampling', [
    item for item in SAMPLING if item[1] != TJSAMP_GRAY
], ids=lambda item: item[0])
def test_rgb_storage_crop_matches_all_source_channels(jpeg, codec, case, sampling):
    bits, encode, decode = codec
    _, subsample, mcu = sampling
    encoded = encode(coordinate_image(bits=bits), pixel_format=TJPF_RGB,
                     jpeg_colorspace=TJCS_RGB, jpeg_subsample=subsample, quality=100)
    pixels = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    result = jpeg.crop(encoded, *case.requested, preserve=case.preserve)
    assert_rectangle(jpeg, decode, result, pixels, case.expected[mcu])
    assert jpeg.decode_header(result)[2:] == (subsample, TJCS_RGB)
    assert jpeg.crop_multiple(encoded, [case.requested],
                              align='shrink' if case.preserve else 'expand') == [result]


@pytest.mark.parametrize(('colorspace', 'subsample'), [
    (TJCS_RGB, subsample) for _, subsample, _ in SAMPLING if subsample != TJSAMP_GRAY
] + [(TJCS_YCbCr, TJSAMP_420)], ids=[
    'rgb' + name for name, subsample, _ in SAMPLING if subsample != TJSAMP_GRAY
] + ['ycbcr420'])
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


@pytest.mark.parametrize('case', CROP_CASES, ids=lambda case: case.name)
def test_batch_auto_alignment_matches_crop_and_explicit_pixels(jpeg, codec, source, case):
    _, _, decode = codec
    encoded, mcu, pixels = source
    align = 'shrink' if case.preserve else 'expand'
    expected = case.expected[mcu]
    for gray in (False, True):
        if expected is None:
            with pytest.raises(ValueError, match=r'crop_parameters\[0\].*after iMCU alignment'):
                jpeg.crop_multiple(encoded, [case.requested], align=align, gray=gray)
        else:
            results = jpeg.crop_multiple(encoded, [case.requested], align=align, gray=gray)
            assert results == [jpeg.crop(encoded, *case.requested,
                                         preserve=case.preserve, gray=gray)]
            assert_rectangle(jpeg, decode, results[0], pixels[gray], expected, gray)


@pytest.mark.parametrize('align', ['expand', 'shrink'])
def test_auto_alignment_zero_dimensions_and_batch_order(jpeg, codec, source, align):
    _, _, decode = codec
    encoded, mcu, pixels = source
    # Explicit origins, independent of the production alignment helper.
    origins = {
        'expand': {(8, 8): (8, 8), (16, 8): (0, 8), (16, 16): (0, 0),
                   (8, 16): (8, 0), (32, 8): (0, 8), (8, 32): (8, 0)},
        'shrink': {(8, 8): (16, 16), (16, 8): (16, 16), (16, 16): (16, 16),
                   (8, 16): (16, 16), (32, 8): (32, 16), (8, 32): (16, 32)},
    }
    ox, oy = origins[align][mcu]
    regions = [(10, 10, 0, 0), (10, 32, 0, 17), (32, 10, 17, 0),
               (32, 32, 2, 2), (0, 0, 0, 0)]
    expected = [(ox, oy, 259 - ox, 197 - oy), (ox, 32, 259 - ox, 17),
                (32, oy, 17, 197 - oy), (32, 32, 2, 2), (0, 0, 259, 197)]
    for gray in (False, True):
        results = jpeg.crop_multiple(encoded, regions, align=align, gray=gray)
        assert len(results) == len(expected)
        for result, rectangle in zip(results, expected):
            assert_rectangle(jpeg, decode, result, pixels[gray], rectangle, gray)


@pytest.mark.parametrize('copynone', [False, True])
def test_strict_default_preserves_positional_arguments_and_metadata(jpeg, codec, copynone):
    bits, encode, _ = codec
    encoded = encode(coordinate_image(64, 64, bits), pixel_format=TJPF_RGB,
                     jpeg_subsample=TJSAMP_420)
    comment = b'crop_multiple metadata regression'
    marker = b'\xff\xfe' + (len(comment) + 2).to_bytes(2, 'big') + comment
    encoded = encoded[:2] + marker + encoded[2:]
    for gray in (False, True):
        default = jpeg.crop_multiple(encoded, [(16, 16, 0, 0)], 1.0, gray, copynone)
        explicit = jpeg.crop_multiple(encoded, [(16, 16, 0, 0)], 1.0, gray,
                                      copynone, align='strict')
        assert default == explicit == [jpeg.crop(encoded, 16, 16, 48, 48,
                                               gray=gray, copynone=copynone)]
        assert (comment in default[0]) is not copynone


def assert_canvas_pixels(actual, source, offset, bits, luminance):
    x, y = offset
    height, width = source.shape[:2]
    np.testing.assert_array_equal(actual[y:y + height, x:x + width], source)
    background = np.ones(actual.shape[:2], dtype=bool)
    background[y:y + height, x:x + width] = False
    np.testing.assert_allclose(actual[background], luminance * ((1 << bits) - 1), atol=1)


@pytest.mark.parametrize(('subsample', 'colorspace'), [
    (subsample, TJCS_GRAY if subsample == TJSAMP_GRAY else TJCS_YCbCr)
    for _, subsample, _ in SAMPLING
] + [
    (subsample, TJCS_RGB) for _, subsample, _ in SAMPLING if subsample != TJSAMP_GRAY
], ids=[item[0] for item in SAMPLING] + [
    'rgb' + name for name, subsample, _ in SAMPLING if subsample != TJSAMP_GRAY
])
@pytest.mark.parametrize('luminance', [0.0, 0.5, 1.0], ids=['black', 'gray', 'white'])
def test_large_canvas_placements_preserve_pixels_and_background(
        jpeg, codec, subsample, colorspace, luminance):
    bits, encode, decode = codec
    encoded = encode(coordinate_image(64, 64, bits), quality=100, pixel_format=TJPF_RGB,
                     jpeg_subsample=subsample, jpeg_colorspace=colorspace)
    source = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    regions = [(128, 128, 320, 320), (64, 0, 128, 64), (0, 64, 64, 128),
               (256, 256, 320, 320), (96, 128, 193, 225)]
    results = jpeg.crop_multiple(encoded, regions, background_luminance=luminance)
    assert len(results) == len(regions)
    for result, (x, y, width, height) in zip(results, regions):
        assert jpeg.decode_header(result)[:2] == (width, height)
        actual = decode(result, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
        assert_canvas_pixels(actual, source, (x, y), bits, luminance)


@pytest.mark.parametrize('subsample', [TJSAMP_422, TJSAMP_420, TJSAMP_441])
@pytest.mark.parametrize('luminance', [0.0, 1.0])
def test_grayscale_large_placement_uses_output_block_size(jpeg, codec, subsample, luminance):
    bits, encode, decode = codec
    # 72x56 has complete 8x8 blocks, but partial source iMCUs.
    encoded = encode(coordinate_image(72, 56, bits), quality=100, pixel_format=TJPF_RGB,
                     jpeg_subsample=subsample)
    source = decode(encoded, pixel_format=TJPF_GRAY)
    result = jpeg.crop_multiple(encoded, [(96, 96, 208, 192)],
                                gray=True, background_luminance=luminance)[0]
    assert jpeg.decode_header(result) == (208, 192, TJSAMP_GRAY, TJCS_GRAY)
    actual = decode(result, pixel_format=TJPF_GRAY)
    assert_canvas_pixels(actual, source, (96, 96), bits, luminance)


@pytest.mark.parametrize(('colorspace', 'subsample'), [
    (TJCS_RGB, TJSAMP_444), (TJCS_YCbCr, TJSAMP_420), (TJCS_GRAY, TJSAMP_GRAY),
], ids=['rgb', 'ycbcr420', 'gray'])
@pytest.mark.parametrize(('size', 'region', 'rectangle', 'offset', 'output_size'), [
    ((64, 67), (128, 16, 256, 0), (0, 16, 64, 51), (128, 0), (256, 51)),
    ((67, 64), (16, 128, 0, 256), (16, 0, 51, 64), (0, 128), (51, 256)),
    ((64, 64), (128, 16, 256, 32), (0, 16, 64, 32), (128, 0), (256, 32)),
    ((64, 64), (16, 128, 32, 256), (16, 0, 32, 64), (0, 128), (32, 256)),
], ids=['extend_x_partial_y', 'partial_x_extend_y', 'extend_x_crop_y', 'crop_x_extend_y'])
def test_large_placement_with_crop_on_other_axis(
        jpeg, codec, colorspace, subsample, size, region, rectangle, offset, output_size):
    bits, encode, decode = codec
    encoded = encode(coordinate_image(*size, bits=bits), quality=100, pixel_format=TJPF_RGB,
                     jpeg_colorspace=colorspace, jpeg_subsample=subsample)
    source = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    result = jpeg.crop_multiple(encoded, [region])[0]
    assert jpeg.decode_header(result)[:2] == output_size
    x, y, width, height = rectangle
    actual = decode(result, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    assert_canvas_pixels(actual, source[y:y + height, x:x + width], offset, bits, 1.0)


@pytest.mark.parametrize('sampling', [
    item for item in SAMPLING if item[1] != TJSAMP_GRAY
], ids=lambda item: item[0])
@pytest.mark.parametrize('luminance', [0.0, 0.5, 1.0], ids=['black', 'gray', 'white'])
@pytest.mark.parametrize(('size', 'region', 'rectangle', 'offset', 'output_size'), [
    ((64, 67), (128, 32, 257, 0), (0, 32, 64, 35), (128, 0), (257, 35)),
    ((67, 64), (32, 128, 0, 257), (32, 0, 35, 64), (0, 128), (35, 257)),
], ids=['extend_x_partial_y', 'partial_x_extend_y'])
def test_rgb_extension_preserves_partial_edges_on_other_axis(
        jpeg, codec, sampling, luminance, size, region, rectangle, offset, output_size):
    bits, encode, decode = codec
    _, subsample, _ = sampling
    encoded = encode(coordinate_image(*size, bits=bits), quality=100,
                     pixel_format=TJPF_RGB, jpeg_colorspace=TJCS_RGB,
                     jpeg_subsample=subsample)
    source = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    result = jpeg.crop_multiple(encoded, [region], background_luminance=luminance)[0]
    assert jpeg.decode_header(result) == (*output_size, subsample, TJCS_RGB)
    x, y, width, height = rectangle
    actual = decode(result, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    assert_canvas_pixels(actual, source[y:y + height, x:x + width], offset, bits, luminance)


@pytest.mark.parametrize('align', ['expand', 'shrink'])
def test_mixed_batch_auto_crops_and_extensions_uses_one_native_transform(jpeg, codec, align):
    bits, encode, decode = codec
    encoded = encode(coordinate_image(64, 64, bits), quality=100,
                     pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_420)
    source = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    regions = [(10, 10, 20, 20), (128, 128, 320, 320), (32, 32, 0, 0)]
    with patch.object(jpeg, '_TurboJPEG__transform', wraps=jpeg._TurboJPEG__transform) as transform:
        results = jpeg.crop_multiple(encoded, regions, align=align)
        assert transform.call_count == 1
        assert transform.call_args.args[3] == 3
    assert len(results) == 3
    expected = (0, 0, 30, 30) if align == 'expand' else (16, 16, 14, 14)
    assert_rectangle(jpeg, decode, results[0], source, expected)
    actual = decode(results[1], pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    assert actual.shape[:2] == (320, 320)
    assert_canvas_pixels(actual, source, (128, 128), bits, 1.0)
    assert_rectangle(jpeg, decode, results[2], source, (32, 32, 32, 32))


@pytest.mark.parametrize('align', ['strict', 'expand', 'shrink'])
@pytest.mark.parametrize('container', ['list', 'tuple', 'generator', 'numpy'])
def test_region_containers_and_numpy_integers_are_not_mutated(jpeg, codec, align, container):
    bits, encode, _ = codec
    encoded = encode(coordinate_image(64, 64, bits), pixel_format=TJPF_RGB,
                     jpeg_subsample=TJSAMP_420)
    origin = 16 if align == 'strict' else 10
    raw = [[origin, origin, 20, 20], [32, 32, 0, 0]]
    snapshot = np.array(raw)
    if container == 'list':
        parameters = raw
    elif container == 'tuple':
        parameters = tuple(tuple(row) for row in raw)
    elif container == 'generator':
        parameters = (tuple(np.int64(value) for value in row) for row in raw)
    else:
        parameters = np.array(raw, dtype=np.int64)
    results = jpeg.crop_multiple(encoded, parameters, align=align)
    assert results == [jpeg.crop(encoded, origin, origin, 20, 20, preserve=align == 'shrink'),
                       jpeg.crop(encoded, 32, 32, 32, 32)]
    np.testing.assert_array_equal(raw, snapshot)
    if container == 'numpy':
        np.testing.assert_array_equal(parameters, snapshot)


@pytest.fixture(scope='module')
def diagnostic_source(jpeg):
    return jpeg.encode(coordinate_image(64, 64), pixel_format=TJPF_RGB,
                       jpeg_subsample=TJSAMP_420)


def assert_region_error(jpeg, encoded, region, parts, error_type=ValueError,
                        align='strict', gray=False):
    # A later invalid entry must be identified before any batch output is made.
    parameters = [(0, 0, 16, 16), (0, 0, 32, 32), region]
    with patch.object(jpeg, '_TurboJPEG__transform',
                      side_effect=AssertionError('invalid batch reached native transform')) as transform:
        with patch.object(jpeg, '_TurboJPEG__destroy', wraps=jpeg._TurboJPEG__destroy) as destroy:
            with pytest.raises(error_type) as caught:
                jpeg.crop_multiple(encoded, parameters, align=align, gray=gray)
            destroy.assert_called_once()
        transform.assert_not_called()
    message = str(caught.value)
    assert message.startswith('crop_parameters[2]: ')
    assert 'source=' in message and 'subsampling=' in message and 'iMCU=' in message
    for part in parts:
        assert part in message, message


@pytest.mark.parametrize(('region', 'align', 'parts'), [
    ((10, 0, 16, 16), 'strict', [
        'x=10', 'x must be a multiple of 16', "align='expand'", 'may include extra area',
        "align='shrink'", 'discards unaligned edges', 'may be empty']),
    ((0, 10, 16, 16), 'strict', [
        'y=10', 'y must be a multiple of 16', "align='expand'", 'may include extra area',
        "align='shrink'", 'discards unaligned edges', 'may be empty']),
    ((64, 0, 0, 16), 'strict', [
        'x=64', 'inside the source width', 'x < 64', 'use x=48, w=0', 'different source region']),
    ((0, 64, 16, 0), 'strict', [
        'y=64', 'inside the source height', 'y < 64', 'use y=48, h=0', 'different source region']),
    ((65, 0, 1, 16), 'expand', ['x=65', 'inside the source width']),
    ((0, 65, 16, 1), 'shrink', ['y=65', 'inside the source height']),
    ((48, 0, 32, 16), 'strict', [
        'w=32', 'x=48', 'source width=64', 'w <= 16', 'w=0', 'reduces output width']),
    ((0, 48, 16, 32), 'strict', [
        'h=32', 'y=48', 'source height=64', 'h <= 16', 'h=0', 'reduces output height']),
    ((272, 128, 320, 320), 'strict', ['x=272', 'output w=320', 'x must be at most 256']),
    ((128, 272, 320, 320), 'strict', ['y=272', 'output h=320', 'y must be at most 256']),
    ((10, 0, 2, 16), 'shrink', ['x=10', 'w=2', 'no pixels', "align='shrink'"]),
    ((0, 10, 16, 2), 'shrink', ['y=10', 'h=2', 'no pixels', "align='shrink'"]),
], ids=['unaligned_x', 'unaligned_y', 'x_at_edge', 'y_at_edge',
        'x_outside_expand', 'y_outside_shrink', 'width_overflow', 'height_overflow',
        'source_exceeds_canvas_x', 'source_exceeds_canvas_y', 'empty_x', 'empty_y'])
def test_geometry_errors_identify_entry_parameter_source_and_grid(
        jpeg, diagnostic_source, region, align, parts):
    assert_region_error(jpeg, diagnostic_source, region,
                        parts + ['source=64x64', 'subsampling=4:2:0', 'iMCU=16x16'], align=align)


@pytest.mark.parametrize('align', ['strict', 'expand', 'shrink'])
@pytest.mark.parametrize(('region', 'parameter'), [
    ((129, 128, 320, 320), 'x=129'),
    ((128, 129, 320, 320), 'y=129'),
    ((128, 10, 320, 16), 'y=10'),
    ((10, 128, 16, 320), 'x=10'),
], ids=['placement_x', 'placement_y', 'extend_x_crop_y', 'crop_x_extend_y'])
def test_extension_operations_never_silently_align_coordinates(
        jpeg, diagnostic_source, align, region, parameter):
    parts = [parameter, parameter[0] + ' must be a multiple of 16', 'source=64x64']
    if align != 'strict':
        parts.append('automatic alignment applies only to pure crops')
    assert_region_error(jpeg, diagnostic_source, region, parts, align=align)


@pytest.mark.parametrize('align', ['strict', 'expand', 'shrink'])
@pytest.mark.parametrize('axis', ['width', 'height'])
@pytest.mark.parametrize(('name', 'subsample', 'mcu'), SAMPLING, ids=[item[0] for item in SAMPLING])
def test_partial_extension_errors_name_source_dimension_and_required_multiple(
        jpeg, align, axis, name, subsample, mcu):
    width, height = (63, 64) if axis == 'width' else (64, 63)
    encoded = jpeg.encode(coordinate_image(width, height), pixel_format=TJPF_RGB,
                          jpeg_subsample=subsample)
    region = (0, 0, 128, height) if axis == 'width' else (0, 0, width, 128)
    argument = 'w' if axis == 'width' else 'h'
    multiple = mcu[0] if axis == 'width' else mcu[1]
    sampling = 'grayscale' if name == 'gray' else ':'.join(name)
    assert_region_error(jpeg, encoded, region, [
        argument + '=128', 'source ' + axis + '=63', 'partial iMCU',
        'source {} must be a multiple of {}'.format(axis, multiple),
        'source={}x{}'.format(width, height), 'iMCU={}x{}'.format(*mcu),
        'subsampling=' + sampling,
    ], align=align)


@pytest.mark.parametrize('align', ['strict', 'expand', 'shrink'])
@pytest.mark.parametrize('gray', [False, True])
def test_partial_extension_error_explains_grayscale_block_requirement(jpeg, codec, align, gray):
    bits, encode, _ = codec
    encoded = encode(coordinate_image(68, 64, bits), pixel_format=TJPF_RGB,
                     jpeg_subsample=TJSAMP_420)
    parts = ['w=144', 'source width=68', 'partial iMCU', 'source=68x64', 'iMCU=16x16',
             'source width must be a multiple of {}'.format(8 if gray else 16)]
    if gray:
        parts.append('grayscale output iMCU=8x8')
    assert_region_error(jpeg, encoded, (64, 0, 144, 64), parts, align=align, gray=gray)


@pytest.mark.parametrize('argument', ['x', 'y', 'w', 'h'])
@pytest.mark.parametrize(('value', 'error_type'), [
    (True, TypeError), (1.5, TypeError), ('16', TypeError), (None, TypeError),
    (np.bool_(False), TypeError), (-1, ValueError), (2**31, ValueError),
], ids=['bool', 'float', 'string', 'none', 'numpy_bool', 'negative', 'overflow'])
def test_invalid_numeric_parameters_include_batch_and_image_context(
        jpeg, diagnostic_source, argument, value, error_type):
    region = [0, 0, 16, 16]
    region[('x', 'y', 'w', 'h').index(argument)] = value
    assert_region_error(jpeg, diagnostic_source, region,
                        [argument + '=', 'source=64x64', 'iMCU=16x16'], error_type)


@pytest.mark.parametrize('region', [None, (), (0,), (0, 0, 16), (0, 0, 16, 16, 0), 42],
                         ids=['none', 'empty', 'one_value', 'three_values', 'five_values', 'scalar'])
def test_malformed_regions_report_their_batch_index(jpeg, diagnostic_source, region):
    assert_region_error(jpeg, diagnostic_source, region,
                        ['expected a region containing (x, y, w, h)', 'source=64x64'])


@pytest.mark.parametrize(('parameters', 'error_type', 'message'), [
    (None, TypeError, 'iterable'), (42, TypeError, 'iterable'),
    ([], ValueError, 'must not be empty'), ((), ValueError, 'must not be empty'),
])
def test_invalid_batch_container_has_clear_context(jpeg, diagnostic_source, parameters, error_type, message):
    with patch.object(jpeg, '_TurboJPEG__transform', side_effect=AssertionError('invalid batch')):
        with pytest.raises(error_type) as caught:
            jpeg.crop_multiple(diagnostic_source, parameters)
    assert 'crop_parameters' in str(caught.value)
    assert message in str(caught.value)
    assert 'source=64x64' in str(caught.value)


@pytest.mark.parametrize(('align', 'error_type'), [
    (None, TypeError), (False, TypeError), (1, TypeError), ([], TypeError),
    ({}, TypeError), (np.array(['strict']), TypeError),
    ('', ValueError), ('auto', ValueError), ('STRICT', ValueError), (' strict ', ValueError),
], ids=['none', 'bool', 'int', 'list', 'dict', 'numpy', 'empty', 'unknown', 'uppercase', 'whitespace'])
def test_invalid_align_is_rejected_before_native_allocation(jpeg, align, error_type):
    with patch.object(jpeg, '_TurboJPEG__init', side_effect=AssertionError('native allocation')):
        with pytest.raises(error_type, match="'align'.*'strict'.*'expand'.*'shrink'"):
            jpeg.crop_multiple(b'invalid JPEG', [(0, 0, 16, 16)], align=align)


@pytest.mark.parametrize('align', ['strict', 'expand', 'shrink'])
@pytest.mark.parametrize('colorspace', [TJCS_CMYK, TJCS_YCCK], ids=['cmyk', 'ycck'])
def test_unsupported_extension_colorspace_names_batch_entry_and_format(jpeg, colorspace, align):
    image = coordinate_image(64, 64)
    image = np.concatenate((image, image[:, :, :1]), axis=2)
    encoded = jpeg.encode(image, pixel_format=TJPF_CMYK, jpeg_colorspace=colorspace,
                          jpeg_subsample=TJSAMP_444)
    name = 'CMYK' if colorspace == TJCS_CMYK else 'YCCK'
    assert_region_error(jpeg, encoded, (128, 128, 320, 320), [
        'w=320', 'h=320', name, 'supports only grayscale, YCbCr, and RGB', 'source=64x64',
        'workflow that supports this colorspace', 'additional lossy compression',
        'preserve color profiles explicitly',
    ], align=align)


@pytest.mark.parametrize('axis', [0, 1], ids=['width', 'height'])
@pytest.mark.parametrize(('name', 'subsample', 'mcu'), SAMPLING, ids=[item[0] for item in SAMPLING])
def test_partial_source_trim_remedy_preserves_exact_remaining_pixels(
        jpeg, codec, axis, name, subsample, mcu):
    bits, encode, decode = codec
    dimensions = [64, 64]
    dimensions[axis] = 63
    encoded = encode(coordinate_image(*dimensions, bits=bits), quality=100,
                     pixel_format=TJPF_RGB, jpeg_subsample=subsample)
    pixels = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    output_size = [64, 64]
    output_size[axis] = 128
    region = (0, 0, *output_size)
    with pytest.raises(ValueError) as caught:
        jpeg.crop_multiple(encoded, [region])
    message = str(caught.value)
    # Explicit expected boundaries, independent of the production helper.
    retained = {8: 56, 16: 48, 32: 32}[mcu[axis]]
    dimension, argument, edge = (
        ('width', 'w', 'right-edge columns') if axis == 0
        else ('height', 'h', 'bottom-edge rows'))
    assert 'crop source {} to {} first'.format(dimension, retained) in message
    assert 'discards {} {}'.format(63 - retained, edge) in message
    assert 'Changing output {} or align does not fix'.format(argument) in message
    assert 'decode, pad, and re-encode (additional lossy compression)' in message

    dimensions[axis] = retained
    width, height = dimensions
    trimmed = jpeg.crop(encoded, 0, 0, width, height)
    result = jpeg.crop_multiple(trimmed, [region])[0]
    assert jpeg.decode_header(result)[:2] == tuple(output_size)
    actual = decode(result, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    assert_canvas_pixels(actual, pixels[:height, :width], (0, 0), bits, 1.0)


@pytest.mark.parametrize('axis', [0, 1], ids=['width', 'height'])
def test_grayscale_source_trim_remedy_uses_output_blocks(jpeg, codec, axis):
    bits, encode, decode = codec
    dimensions = [64, 64]
    dimensions[axis] = 63
    encoded = encode(coordinate_image(*dimensions, bits=bits), quality=100,
                     pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_420)
    pixels = decode(encoded, pixel_format=TJPF_GRAY)
    output_size = [64, 64]
    output_size[axis] = 128
    with pytest.raises(ValueError) as caught:
        jpeg.crop_multiple(encoded, [(0, 0, *output_size)], gray=True)
    dimension = 'width' if axis == 0 else 'height'
    assert 'source {} must be a multiple of 8'.format(dimension) in str(caught.value)
    assert 'crop source {} to 56 first (discards 7'.format(dimension) in str(caught.value)
    dimensions[axis] = 56
    width, height = dimensions
    trimmed = jpeg.crop(encoded, 0, 0, width, height)
    result = jpeg.crop_multiple(trimmed, [(0, 0, *output_size)], gray=True)[0]
    assert jpeg.decode_header(result) == (*output_size, TJSAMP_GRAY, TJCS_GRAY)
    assert_canvas_pixels(decode(result, pixel_format=TJPF_GRAY),
                         pixels[:height, :width], (0, 0), bits, 1.0)


@pytest.mark.parametrize('axis', [0, 1], ids=['width', 'height'])
@pytest.mark.parametrize('subsample', [TJSAMP_GRAY, TJSAMP_420, TJSAMP_441])
def test_source_smaller_than_one_imcu_does_not_suggest_zero_size_crop(
        jpeg, codec, axis, subsample):
    bits, encode, _ = codec
    dimensions = [64, 64]
    dimensions[axis] = 7
    encoded = encode(coordinate_image(*dimensions, bits=bits),
                     pixel_format=TJPF_RGB, jpeg_subsample=subsample)
    output_size = [64, 64]
    output_size[axis] = 128
    with pytest.raises(ValueError) as caught:
        jpeg.crop_multiple(encoded, [(0, 0, *output_size)])
    message = str(caught.value)
    assert 'no non-empty source' in message and 'can be obtained by cropping' in message
    assert 'crop source' not in message
    assert 'decode, pad, and re-encode (additional lossy compression)' in message


@pytest.mark.parametrize('axis', [0, 1], ids=['width', 'height'])
@pytest.mark.parametrize(('name', 'subsample', 'mcu'), SAMPLING, ids=[item[0] for item in SAMPLING])
def test_empty_shrink_remedy_gives_minimum_extent_with_a_real_pixel(
        jpeg, codec, axis, name, subsample, mcu):
    bits, encode, decode = codec
    encoded = encode(coordinate_image(64, 64, bits), quality=100,
                     pixel_format=TJPF_RGB, jpeg_subsample=subsample)
    pixels = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    requested = [0, 0, 16, 16]
    requested[axis], requested[axis + 2] = 10, 2
    with pytest.raises(ValueError) as caught:
        jpeg.crop_multiple(encoded, [requested], align='shrink')
    next_origin, minimum_extent = {8: (16, 7), 16: (16, 7), 32: (32, 23)}[mcu[axis]]
    message = str(caught.value)
    assert 'increase {} to at least {}'.format('w' if axis == 0 else 'h', minimum_extent) in message
    assert 'selects a larger requested area' in message
    assert "align='expand' (may include extra source area)" in message
    assert 'additional lossy compression' in message

    # One pixel short still fails; the suggested minimum produces exactly one.
    requested[axis + 2] = minimum_extent - 1
    with pytest.raises(ValueError, match='contains no pixels'):
        jpeg.crop_multiple(encoded, [requested], align='shrink')
    requested[axis + 2] = minimum_extent
    result = jpeg.crop_multiple(encoded, [requested], align='shrink')[0]
    rectangle = [0, 0, 16, 16]
    rectangle[axis], rectangle[axis + 2] = next_origin, 1
    assert_rectangle(jpeg, decode, result, pixels, rectangle)


@pytest.mark.parametrize(('requested', 'corrected', 'rectangle', 'hint'), [
    ((258, 0, 1, 16), (256, 0, 0, 16), (256, 0, 3, 16), 'x=256, w=0'),
    ((0, 196, 16, 1), (0, 192, 16, 0), (0, 192, 16, 5), 'y=192, h=0'),
], ids=['right_partial_edge', 'bottom_partial_edge'])
def test_empty_shrink_at_source_edge_suggests_earlier_pixels(
        jpeg, codec, source, requested, corrected, rectangle, hint):
    _, _, decode = codec
    encoded, _, pixels = source
    with pytest.raises(ValueError) as caught:
        jpeg.crop_multiple(encoded, [requested], align='shrink')
    message = str(caught.value)
    assert 'no aligned' in message and 'inside the source at or after' in message
    assert 'use ' + hint + ' to include earlier source pixels' in message
    assert 'increase' not in message
    result = jpeg.crop_multiple(encoded, [corrected], align='shrink')[0]
    assert_rectangle(jpeg, decode, result, pixels[False], rectangle)
    assert jpeg.crop_multiple(encoded, [requested], align='expand') == [result]


@pytest.mark.parametrize(('requested', 'corrected', 'rectangle', 'offset', 'hint', 'cost'), [
    ((129, 128, 320, 320), (128, 128, 320, 320), (0, 0, 64, 64), (128, 128),
     'use x=128', 'moves the source placement'),
    ((128, 129, 320, 320), (128, 128, 320, 320), (0, 0, 64, 64), (128, 128),
     'use y=128', 'moves the source placement'),
    ((128, 10, 320, 16), (128, 0, 320, 26), (0, 0, 64, 26), (128, 0),
     'use y=0, h=26', 'includes extra source pixels and changes output height'),
    ((10, 128, 16, 320), (0, 128, 26, 320), (0, 0, 26, 64), (0, 128),
     'use x=0, w=26', 'includes extra source pixels and changes output width'),
], ids=['place_x', 'place_y', 'extend_x_crop_y', 'crop_x_extend_y'])
def test_extension_alignment_remedy_explains_actual_geometry_change(
        jpeg, codec, requested, corrected, rectangle, offset, hint, cost):
    bits, encode, decode = codec
    encoded = encode(coordinate_image(64, 64, bits), quality=100,
                     pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_420)
    pixels = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    with pytest.raises(ValueError) as caught:
        jpeg.crop_multiple(encoded, [requested], align='expand')
    assert hint in str(caught.value) and cost in str(caught.value)
    assert 'automatic alignment applies only to pure crops' in str(caught.value)
    result = jpeg.crop_multiple(encoded, [corrected], align='expand')[0]
    assert jpeg.decode_header(result)[:2] == corrected[2:]
    x, y, width, height = rectangle
    assert_canvas_pixels(decode(result, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE),
                         pixels[y:y + height, x:x + width], offset, bits, 1.0)


@pytest.mark.parametrize('axis', [0, 1], ids=['width', 'height'])
def test_placement_overflow_remedies_move_or_enlarge_with_odd_canvas(jpeg, codec, axis):
    bits, encode, decode = codec
    encoded = encode(coordinate_image(64, 64, bits), quality=100,
                     pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_420)
    pixels = decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    requested = [128, 128, 320, 320]
    requested[axis], requested[axis + 2] = 288, 321
    origin_name, size_name = ('x', 'w') if axis == 0 else ('y', 'h')
    with pytest.raises(ValueError) as caught:
        jpeg.crop_multiple(encoded, [requested])
    message = str(caught.value)
    assert '{} must be at most 257'.format(origin_name) in message
    assert 'use {}=256 (moves the source within the canvas)'.format(origin_name) in message
    assert 'increase {} to at least 352 (larger output canvas)'.format(size_name) in message
    moved, enlarged = requested.copy(), requested.copy()
    moved[axis] = 256
    enlarged[axis + 2] = 352
    for corrected in (moved, enlarged):
        result = jpeg.crop_multiple(encoded, [corrected])[0]
        assert jpeg.decode_header(result)[:2] == tuple(corrected[2:])
        actual = decode(result, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
        assert_canvas_pixels(actual, pixels, corrected[:2], bits, 1.0)

    # Increasing the canvas alone cannot fix an unaligned offset as well.
    requested[axis] = 289
    with pytest.raises(ValueError) as caught:
        jpeg.crop_multiple(encoded, [requested])
    assert 'choose an aligned {} first'.format(origin_name) in str(caught.value)
    assert 'increase {} to at least'.format(size_name) not in str(caught.value)

"""Deterministic, labelled crop targets and hand-specified expected rectangles."""

from collections import namedtuple

import numpy as np

from turbojpeg import (
    TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY, TJSAMP_440,
    TJSAMP_411, TJSAMP_441,
)


# This table is independent of the wrapper's MCU constants/alignment helper.
SAMPLING = [
    ('444', TJSAMP_444, (8, 8)),
    ('422', TJSAMP_422, (16, 8)),
    ('420', TJSAMP_420, (16, 16)),
    ('gray', TJSAMP_GRAY, (8, 8)),
    ('440', TJSAMP_440, (8, 16)),
    ('411', TJSAMP_411, (32, 8)),
    ('441', TJSAMP_441, (8, 32)),
]
MCU_SHAPES = [(8, 8), (16, 8), (16, 16), (8, 16), (32, 8), (8, 32)]
CropCase = namedtuple('CropCase', 'name requested preserve expected')


def _case(name, requested, preserve, *expected):
    if len(expected) == 1:
        expected = expected * len(MCU_SHAPES)
    assert len(expected) == len(MCU_SHAPES)
    return CropCase(name, requested, preserve, dict(zip(MCU_SHAPES, expected)))


# Rectangles are (x, y, width, height) in a 259 x 197 source. None means
# inward alignment leaves an empty crop. These are explicit test oracles,
# not values calculated using the production alignment implementation.
CROP_CASES = [
    _case('origin_pixel', (0, 0, 1, 1), False, (0, 0, 1, 1)),
    _case('aligned_tiny', (32, 32, 2, 2), False, (32, 32, 2, 2)),
    _case('tiny_expand', (10, 10, 2, 2), False,
          (8, 8, 4, 4), (0, 8, 12, 4), (0, 0, 12, 12),
          (8, 0, 4, 12), (0, 8, 12, 4), (8, 0, 4, 12)),
    _case('tiny_preserve_empty', (10, 10, 2, 2), True, None),
    _case('cross_mcu_expand', (31, 31, 2, 2), False,
          (24, 24, 9, 9), (16, 24, 17, 9), (16, 16, 17, 17),
          (24, 16, 9, 17), (0, 24, 33, 9), (24, 0, 9, 33)),
    _case('cross_mcu_preserve', (31, 31, 2, 2), True, (32, 32, 1, 1)),
    _case('interior_expand', (45, 37, 84, 71), False,
          (40, 32, 89, 76), (32, 32, 97, 76), (32, 32, 97, 76),
          (40, 32, 89, 76), (32, 32, 97, 76), (40, 32, 89, 76)),
    _case('interior_preserve', (45, 37, 84, 71), True,
          (48, 40, 81, 68), (48, 40, 81, 68), (48, 48, 81, 60),
          (48, 48, 81, 60), (64, 40, 65, 68), (48, 64, 81, 44)),
    _case('clamp_edges_expand', (253, 189, 50, 50), False,
          (248, 184, 11, 13), (240, 184, 19, 13), (240, 176, 19, 21),
          (248, 176, 11, 21), (224, 184, 35, 13), (248, 160, 11, 37)),
    _case('clamp_edges_preserve', (253, 189, 50, 50), True, (256, 192, 3, 5)),
    _case('last_pixel_expand', (258, 196, 8, 8), False, (256, 192, 3, 5)),
    _case('last_pixel_preserve_empty', (258, 196, 8, 8), True, None),
    _case('whole_image', (0, 0, 259, 197), False, (0, 0, 259, 197)),
    _case('horizontal_only', (10, 32, 2, 2), False,
          (8, 32, 4, 2), (0, 32, 12, 2), (0, 32, 12, 2),
          (8, 32, 4, 2), (0, 32, 12, 2), (8, 32, 4, 2)),
    _case('vertical_only', (32, 10, 2, 2), False,
          (32, 8, 2, 4), (32, 8, 2, 4), (32, 0, 2, 12),
          (32, 0, 2, 12), (32, 8, 2, 4), (32, 0, 2, 12)),
]

BATCH_CASES = [
    ((32, 32, 2, 2), (32, 32, 2, 2)),
    ((32, 32, 0, 0), (32, 32, 227, 165)),
    ((0, 0, 0, 0), (0, 0, 259, 197)),
    ((256, 192, 3, 5), (256, 192, 3, 5)),
    ((256, 192, 0, 0), (256, 192, 3, 5)),
    ((32, 32, 0, 17), (32, 32, 227, 17)),
    ((32, 32, 17, 0), (32, 32, 17, 165)),
]

_FONT = {
    '0': ('111', '101', '101', '101', '111'),
    '1': ('010', '110', '010', '010', '111'),
    '2': ('111', '001', '111', '100', '111'),
    '3': ('111', '001', '111', '001', '111'),
    '4': ('101', '101', '111', '001', '001'),
    '5': ('111', '100', '111', '001', '111'),
    '6': ('111', '100', '111', '101', '111'),
    '7': ('111', '001', '010', '010', '010'),
    '8': ('111', '101', '111', '101', '111'),
    '9': ('111', '101', '111', '001', '111'),
    ',': ('0', '0', '0', '1', '1'),
}


def coordinate_image(width=259, height=197, bits=8):
    """RGB grid with absolute x,y labels, gradients, and visible partial edges.

    NumPy alone constructs the fixture, so CI needs no imaging/font package.
    Labels give each 32-pixel cell's top-left source coordinates. The varying
    texture also detects misplaced pixels within a cell, including tiny crops.
    """
    y, x = np.indices((height, width))
    image = np.stack((32 + (3*x + 5*y) % 192,
                      32 + (7*x + 2*y) % 192,
                      32 + (2*x + 9*y) % 192), axis=-1).astype(np.uint8)
    image[(x % 16 == 0) | (y % 16 == 0)] = (150, 150, 150)
    image[(x % 32 == 0) | (y % 32 == 0)] = (255, 255, 255)
    for cy in range(0, height - 8, 32):
        for cx in range(0, width - 30, 32):
            label = '{},{}'.format(cx, cy)
            image[cy + 2:cy + 9, cx + 2:cx + 30] = 8
            cursor = cx + 3
            for char in label:
                glyph = np.array([[bit == '1' for bit in row]
                                  for row in _FONT[char]])
                patch = image[cy + 3:cy + 8, cursor:cursor + glyph.shape[1]]
                patch[glyph] = 250
                cursor += glyph.shape[1] + 1
    # Deliberately conspicuous strips in the incomplete right/bottom MCU.
    if width % 32:
        image[:, width - width % 32:] = (255, 32, 192)
    if height % 32:
        image[height - height % 32:, :] = (32, 255, 80)
    image[-1, -1] = (255, 255, 0)
    return image if bits == 8 else image.astype(np.uint16) * 16

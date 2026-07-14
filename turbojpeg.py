# -*- coding: UTF-8 -*-
#
# PyTurboJPEG - A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image.
#
# Copyright (c) 2018-2026, Lilo Huang. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__author__ = 'Lilo Huang <kuso.cc@gmail.com>'
__version__ = '2.5.0'

from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_int,
    c_short,
    c_size_t,
    c_ubyte,
    c_ushort,
    c_void_p,
    cast,
    cdll,
    pointer,
    string_at,
)
from ctypes.util import find_library
from typing import NamedTuple
import platform
import numpy as np
import warnings
import os
from struct import unpack

# default libTurboJPEG library path
DEFAULT_LIB_PATHS = {
    'Darwin': [
        '/usr/local/lib/libturbojpeg.dylib',
        '/usr/local/opt/jpeg-turbo/lib/libturbojpeg.dylib',
        '/opt/libjpeg-turbo/lib64/libturbojpeg.dylib',
        '/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib'
    ],
    'Linux': [
        '/usr/local/lib/libturbojpeg.so.0',
        '/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0',
        '/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0',
        '/usr/lib/libturbojpeg.so.0',
        '/usr/lib64/libturbojpeg.so.0',
        '/opt/libjpeg-turbo/lib64/libturbojpeg.so'
    ],
    'FreeBSD': [
        '/usr/local/lib/libturbojpeg.so.0',
        '/usr/local/lib/libturbojpeg.so'
    ],
    'NetBSD': [
        '/usr/pkg/lib/libturbojpeg.so.0',
        '/usr/pkg/lib/libturbojpeg.so'
    ],
    'Windows': ['C:/libjpeg-turbo64/bin/turbojpeg.dll']
}

# error codes
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
TJERR_WARNING = 0
TJERR_FATAL = 1

# color spaces
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
TJCS_RGB = 0
TJCS_YCbCr = 1
TJCS_GRAY = 2
TJCS_CMYK = 3
TJCS_YCCK = 4

# pixel formats
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
TJPF_RGB = 0
TJPF_BGR = 1
TJPF_RGBX = 2
TJPF_BGRX = 3
TJPF_XBGR = 4
TJPF_XRGB = 5
TJPF_GRAY = 6
TJPF_RGBA = 7
TJPF_BGRA = 8
TJPF_ABGR = 9
TJPF_ARGB = 10
TJPF_CMYK = 11

# chrominance subsampling options
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
TJSAMP_444 = 0
TJSAMP_422 = 1
TJSAMP_420 = 2
TJSAMP_GRAY = 3
TJSAMP_440 = 4
TJSAMP_411 = 5
TJSAMP_441 = 6

# Precision constants for TurboJPEG 3.0+
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
TJPRECISION_8 = 8
TJPRECISION_12 = 12
TJPRECISION_16 = 16

# transform operations
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
TJXOP_NONE = 0
TJXOP_HFLIP = 1
TJXOP_VFLIP = 2
TJXOP_TRANSPOSE = 3
TJXOP_TRANSVERSE = 4
TJXOP_ROT90 = 5
TJXOP_ROT180 = 6
TJXOP_ROT270 = 7

# transform options
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
TJXOPT_PERFECT = 1
TJXOPT_TRIM = 2
TJXOPT_CROP = 4
TJXOPT_GRAY = 8
TJXOPT_NOOUTPUT = 16
TJXOPT_PROGRESSIVE = 32
TJXOPT_COPYNONE = 64
TJXOPT_OPTIMIZE = 256  # Huffman table optimization

# pixel size
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
tjPixelSize = [3, 3, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4]

# MCU block width (in pixels) for a given level of chrominance subsampling.
# MCU block sizes:
#  - 8x8 for no subsampling or grayscale
#  - 16x8 for 4:2:2
#  - 8x16 for 4:4:0
#  - 16x16 for 4:2:0
#  - 32x8 for 4:1:1
#  - 8x32 for 4:4:1
tjMCUWidth = [8, 16, 16, 8, 8, 32, 8]

# MCU block height (in pixels) for a given level of chrominance subsampling.
# MCU block sizes:
#  - 8x8 for no subsampling or grayscale
#  - 16x8 for 4:2:2
#  - 8x16 for 4:4:0
#  - 16x16 for 4:2:0
#  - 32x8 for 4:1:1
#  - 8x32 for 4:4:1
tjMCUHeight = [8, 8, 16, 8, 16, 8, 32]

# miscellaneous flags
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
# note: TJFLAG_NOREALLOC cannot be supported due to reallocation is needed by PyTurboJPEG.
TJFLAG_BOTTOMUP = 2
TJFLAG_FASTUPSAMPLE = 256
TJFLAG_FASTDCT = 2048
TJFLAG_ACCURATEDCT = 4096
TJFLAG_STOPONWARNING = 8192
TJFLAG_PROGRESSIVE = 16384
TJFLAG_LIMITSCANS = 32768

# tj3Init types
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
TJINIT_COMPRESS = 0
TJINIT_DECOMPRESS = 1
TJINIT_TRANSFORM = 2

# tj3Set/tj3Get parameters
# see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/turbojpeg.h
TJPARAM_STOPONWARNING = 0
TJPARAM_BOTTOMUP = 1
TJPARAM_NOREALLOC = 2
TJPARAM_QUALITY = 3
TJPARAM_SUBSAMP = 4
TJPARAM_JPEGWIDTH = 5
TJPARAM_JPEGHEIGHT = 6
TJPARAM_PRECISION = 7
TJPARAM_COLORSPACE = 8
TJPARAM_FASTUPSAMPLE = 9
TJPARAM_FASTDCT = 10
TJPARAM_OPTIMIZE = 11
TJPARAM_PROGRESSIVE = 12
TJPARAM_SCANLIMIT = 13
TJPARAM_ARITHMETIC = 14
TJPARAM_LOSSLESS = 15
TJPARAM_LOSSLESSPSV = 16
TJPARAM_LOSSLESSPT = 17
TJPARAM_RESTARTBLOCKS = 18
TJPARAM_RESTARTROWS = 19
TJPARAM_XDENSITY = 20
TJPARAM_YDENSITY = 21
TJPARAM_DENSITYUNITS = 22
TJPARAM_MAXMEMORY = 23
TJPARAM_MAXPIXELS = 24
TJPARAM_SAVEMARKERS = 25

# Resource limits are opt-in so that very large trusted-image workflows retain
# libjpeg-turbo's full supported range.  Applications that process untrusted
# JPEG sources should configure finite limits for their deployment.
DEFAULT_MAX_PIXELS = 0
DEFAULT_MAX_MEMORY = 0
DEFAULT_SCAN_LIMIT = 0

_MAX_TJPARAM_VALUE = (1 << 31) - 1
_LEGACY_SCAN_LIMIT = 500

_DCT_FLAGS = TJFLAG_FASTDCT | TJFLAG_ACCURATEDCT
_PACKED_DECOMPRESS_FLAGS = (
    TJFLAG_BOTTOMUP | TJFLAG_FASTUPSAMPLE | _DCT_FLAGS |
    TJFLAG_STOPONWARNING | TJFLAG_LIMITSCANS
)
_YUV_DECOMPRESS_FLAGS = (
    _DCT_FLAGS | TJFLAG_STOPONWARNING | TJFLAG_LIMITSCANS
)
_PACKED_COMPRESS_FLAGS = (
    TJFLAG_BOTTOMUP | _DCT_FLAGS | TJFLAG_STOPONWARNING |
    TJFLAG_PROGRESSIVE
)
_YUV_COMPRESS_FLAGS = (
    _DCT_FLAGS | TJFLAG_STOPONWARNING | TJFLAG_PROGRESSIVE
)
_SCALE_FLAGS = (
    _DCT_FLAGS | TJFLAG_STOPONWARNING | TJFLAG_PROGRESSIVE |
    TJFLAG_LIMITSCANS
)

class CroppingRegion(Structure):
    _fields_ = [("x", c_int), ("y", c_int), ("w", c_int), ("h", c_int)]

class ScalingFactor(Structure):
    _fields_ = ('num', c_int), ('denom', c_int)


class YUVPlaneInfo(NamedTuple):
    """Location and dimensions of one plane in a unified YUV buffer."""

    offset: int
    stride: int
    width: int
    height: int

CUSTOMFILTER = CFUNCTYPE(
    c_int,
    POINTER(c_short),
    CroppingRegion,
    CroppingRegion,
    c_int,
    c_int,
    c_void_p
)

class BackgroundStruct(Structure):
    """Struct to send data to fill_background callback function.

    Parameters
    ----------
    w: c_int
        Width of the input image.
    h: c_int
        Height of the input image.
    lum: c_int
        Luminance value to use as background when extending the image.
    """
    _fields_ = [
        ("w", c_int),
        ("h", c_int),
        ("lum", c_int)
    ]

class TransformStruct(Structure):
    _fields_ = [
        ("r", CroppingRegion),
        ("op", c_int),
        ("options", c_int),
        ("data", POINTER(BackgroundStruct)),
        ("customFilter", CUSTOMFILTER)
    ]

# MCU for luminance is always 8
MCU_WIDTH = 8
MCU_HEIGHT = 8
MCU_SIZE = 64

def _fill_background(
        coeffs_ptr, arrayRegion, planeRegion, componentID, transformID,
        transform_ptr):
    """Callback function for filling extended crop images with background
    color. The callback can be called multiple times for each component, each
    call providing a region (defined by arrayRegion) of the image.

    Parameters
    ----------
    coeffs_ptr: POINTER(c_short)
        Pointer to the coefficient array for the callback.
    arrayRegion: CroppingRegion
        The width and height coefficient array and its offset relative to
        the component plane.
    planeRegion: CroppingRegion
        The width and height of the component plane of the coefficient array.
    componentID: c_int
        The component number (i.e. 0, 1, or 2)
    transformID: c_int
        The index of the transformation in the array of transformation given to
        the transform function.
    transform_ptr: c_voipd_p
        Pointer to the transform structure used for the transformation.

    Returns
    ----------
    c_int
        CFUNCTYPE function must return an int.
    """

    # Only modify luminance data, so we dont need to worry about subsampling
    if componentID == 0:
        coeff_array_size = arrayRegion.w * arrayRegion.h
        # Read the coefficients in the pointer as a np array (no copy)
        ArrayType = c_short*coeff_array_size
        array_pointer = cast(coeffs_ptr, POINTER(ArrayType))
        coeffs = np.frombuffer(array_pointer.contents, dtype=np.int16)
        coeffs.shape = (
            arrayRegion.h//MCU_WIDTH,
            arrayRegion.w//MCU_HEIGHT,
            MCU_SIZE
        )

        # Cast the content of the transform pointer into a transform structure
        transform = cast(transform_ptr, POINTER(TransformStruct)).contents
        # Cast the content of the callback data pointer in the transform
        # structure to a background structure
        background_data = cast(
            transform.data, POINTER(BackgroundStruct)
        ).contents

        # The coeff array is typically just one MCU heigh, but it is up to the
        # libjpeg implementation how to do it. The part of the coeff array that
        # is 'left' of 'non-background' data should thus be handled separately
        # from the part 'under'. (Most of the time, the coeff array will be
        # either 'left' or 'under', but both could happen). Note that start
        # and end rows defined below can be outside the arrayRegion, but that
        # the range they then define is of 0 length.

        # fill mcus left of image
        left_start_row = min(arrayRegion.y, background_data.h) - arrayRegion.y
        left_end_row = (
            min(arrayRegion.y+arrayRegion.h, background_data.h)
            - arrayRegion.y
        )
        y_start = left_start_row // MCU_HEIGHT
        y_end = left_end_row // MCU_HEIGHT
        x_start = background_data.w // MCU_WIDTH
        x_end = planeRegion.w // MCU_WIDTH
        if y_end > y_start and x_end > x_start:
            coeffs[y_start:y_end, x_start:x_end, 0] = background_data.lum

        # fill mcus under image
        bottom_start_row = (
            max(arrayRegion.y, background_data.h) - arrayRegion.y
        )
        bottom_end_row = (
            max(arrayRegion.y+arrayRegion.h, background_data.h)
            - arrayRegion.y
        )
        y_start = bottom_start_row // MCU_HEIGHT
        y_end = bottom_end_row // MCU_HEIGHT
        x_end = planeRegion.w // MCU_WIDTH
        if y_end > y_start and x_end > 0:
            coeffs[y_start:y_end, 0:x_end, 0] = background_data.lum

    return None


def fill_background(
        coeffs_ptr, arrayRegion, planeRegion, componentID, transformID,
        transform_ptr):
    """TurboJPEG custom filter that converts Python failures to native errors."""
    try:
        _fill_background(
            coeffs_ptr, arrayRegion, planeRegion, componentID, transformID,
            transform_ptr)
    except Exception:
        return -1
    return 0

def split_byte_into_nibbles(value):
    """Split byte int into 2 nibbles (4 bits)."""
    first = value >> 4
    second = value & 0x0F
    return first, second

class TurboJPEG(object):
    """A Python wrapper of libjpeg-turbo for JPEG encoding and decoding.

    Parameters
    ----------
    lib_path : str or None
        Explicit path to libturbojpeg, or None to locate it automatically.
    max_pixels : int
        Maximum source image size in pixels for decompression and lossless
        transforms.  The default 0 disables the limit.
    max_memory : int
        Maximum intermediate-buffer memory in megabytes for decompression and
        lossless transforms.  The default 0 disables the limit.
    scan_limit : int
        Maximum progressive JPEG scan count for decompression and lossless
        transforms.  The default 0 disables the limit.
    """
    __resource_limit_warning_issued = False

    def __init__(
            self, lib_path=None, max_pixels=DEFAULT_MAX_PIXELS,
            max_memory=DEFAULT_MAX_MEMORY, scan_limit=DEFAULT_SCAN_LIMIT):
        self.__max_pixels = self.__validate_resource_limit(
            'max_pixels', max_pixels)
        self.__max_memory = self.__validate_resource_limit(
            'max_memory', max_memory)
        self.__scan_limit = self.__validate_resource_limit(
            'scan_limit', scan_limit)
        self.__source_limits_enabled = bool(
            self.__max_pixels or self.__max_memory or self.__scan_limit)

        turbo_jpeg = cdll.LoadLibrary(
            self.__find_turbojpeg() if lib_path is None else lib_path)
        
        # Check for TurboJPEG 3.x API compatibility
        # tj3Init is the key function that indicates TurboJPEG 3.0+
        if not hasattr(turbo_jpeg, 'tj3Init'):
            raise RuntimeError(
                'PyTurboJPEG 2.0 requires libjpeg-turbo 3.0 or later.\n'
                'The loaded library appears to be libjpeg-turbo 2.x or older.\n'
                '\n'
                'Please upgrade your libjpeg-turbo installation to version 3.0 or later.\n'
                'Download the appropriate binary for your system from:\n'
                'https://github.com/libjpeg-turbo/libjpeg-turbo/releases\n'
                '\n'
                'Alternatively, use PyTurboJPEG 1.x for libjpeg-turbo 2.x compatibility.')
        
        # tj3Init - unified initialization for compress/decompress/transform
        self.__init = turbo_jpeg.tj3Init
        self.__init.argtypes = [c_int]
        self.__init.restype = c_void_p
        
        # tj3Destroy - cleanup
        self.__destroy = turbo_jpeg.tj3Destroy
        self.__destroy.argtypes = [c_void_p]
        self.__destroy.restype = None
        
        # tj3Set - set compression/decompression parameters
        self.__set = turbo_jpeg.tj3Set
        self.__set.argtypes = [c_void_p, c_int, c_int]
        self.__set.restype = c_int
        
        # tj3Get - get parameters from handle
        self.__get = turbo_jpeg.tj3Get
        self.__get.argtypes = [c_void_p, c_int]
        self.__get.restype = c_int
        
        # tj3SetScalingFactor - set scaling factor for decompression
        self.__set_scaling_factor = turbo_jpeg.tj3SetScalingFactor
        self.__set_scaling_factor.argtypes = [c_void_p, ScalingFactor]
        self.__set_scaling_factor.restype = c_int
        
        # tj3JPEGBufSize - calculate buffer size for JPEG compression
        self.__buffer_size = turbo_jpeg.tj3JPEGBufSize
        self.__buffer_size.argtypes = [c_int, c_int, c_int]
        self.__buffer_size.restype = c_size_t
        
        # tj3YUVBufSize - calculate buffer size for YUV
        self.__buffer_size_YUV = turbo_jpeg.tj3YUVBufSize
        self.__buffer_size_YUV.argtypes = [c_int, c_int, c_int, c_int]
        self.__buffer_size_YUV.restype = c_size_t
        
        # tj3YUVPlaneWidth - get YUV plane width
        self.__plane_width = turbo_jpeg.tj3YUVPlaneWidth
        self.__plane_width.argtypes = [c_int, c_int, c_int]
        self.__plane_width.restype = c_int
        
        # tj3YUVPlaneHeight - get YUV plane height
        self.__plane_height = turbo_jpeg.tj3YUVPlaneHeight
        self.__plane_height.argtypes = [c_int, c_int, c_int]
        self.__plane_height.restype = c_int
        
        # tj3DecompressHeader - decompress JPEG header
        self.__decompress_header = turbo_jpeg.tj3DecompressHeader
        self.__decompress_header.argtypes = [c_void_p, POINTER(c_ubyte), c_size_t]
        self.__decompress_header.restype = c_int
        
        # tj3Decompress8 - decompress JPEG to 8-bit image
        self.__decompress = turbo_jpeg.tj3Decompress8
        self.__decompress.argtypes = [
            c_void_p, POINTER(c_ubyte), c_size_t, POINTER(c_ubyte), c_int, c_int]
        self.__decompress.restype = c_int
        
        # tj3DecompressToYUV8 - decompress JPEG to YUV
        self.__decompressToYUV = turbo_jpeg.tj3DecompressToYUV8
        self.__decompressToYUV.argtypes = [
            c_void_p, POINTER(c_ubyte), c_size_t, POINTER(c_ubyte), c_int]
        self.__decompressToYUV.restype = c_int
        
        # tj3DecompressToYUVPlanes8 - decompress JPEG to YUV planes
        self.__decompressToYUVPlanes = turbo_jpeg.tj3DecompressToYUVPlanes8
        self.__decompressToYUVPlanes.argtypes = [
            c_void_p, POINTER(c_ubyte), c_size_t, POINTER(POINTER(c_ubyte)), POINTER(c_int)]
        self.__decompressToYUVPlanes.restype = c_int
        
        # tj3Compress8 - compress 8-bit image to JPEG
        self.__compress = turbo_jpeg.tj3Compress8
        self.__compress.argtypes = [
            c_void_p, POINTER(c_ubyte), c_int, c_int, c_int, c_int,
            POINTER(c_void_p), POINTER(c_size_t)]
        self.__compress.restype = c_int
        
        # tj3CompressFromYUV8 - compress YUV to JPEG
        self.__compressFromYUV = turbo_jpeg.tj3CompressFromYUV8
        self.__compressFromYUV.argtypes = [
            c_void_p, POINTER(c_ubyte), c_int, c_int, c_int,
            POINTER(c_void_p), POINTER(c_size_t)]
        self.__compressFromYUV.restype = c_int
        
        # tj3Transform - lossless JPEG transformation
        self.__transform = turbo_jpeg.tj3Transform
        self.__transform.argtypes = [
            c_void_p, POINTER(c_ubyte), c_size_t, c_int, POINTER(c_void_p),
            POINTER(c_size_t), POINTER(TransformStruct)]
        self.__transform.restype = c_int
        
        # tj3Free - free memory allocated by TurboJPEG
        self.__free = turbo_jpeg.tj3Free
        self.__free.argtypes = [c_void_p]
        self.__free.restype = None
        
        # tj3Alloc - allocate memory using TurboJPEG allocator
        self.__alloc = turbo_jpeg.tj3Alloc
        self.__alloc.argtypes = [c_size_t]
        self.__alloc.restype = c_void_p
        
        # tj3GetErrorStr - get error string
        self.__get_error_str = turbo_jpeg.tj3GetErrorStr
        self.__get_error_str.argtypes = [c_void_p]
        self.__get_error_str.restype = c_char_p
        
        # tj3GetErrorCode - get error code
        self.__get_error_code = turbo_jpeg.tj3GetErrorCode
        self.__get_error_code.argtypes = [c_void_p]
        self.__get_error_code.restype = c_int
        
        # tj3Compress12 - compress 12-bit image to JPEG
        self.__compress12 = turbo_jpeg.tj3Compress12
        self.__compress12.argtypes = [
            c_void_p, POINTER(c_ushort), c_int, c_int, c_int, c_int,
            POINTER(c_void_p), POINTER(c_size_t)]
        self.__compress12.restype = c_int
        
        # tj3Compress16 - compress 16-bit image to JPEG
        self.__compress16 = turbo_jpeg.tj3Compress16
        self.__compress16.argtypes = [
            c_void_p, POINTER(c_ushort), c_int, c_int, c_int, c_int,
            POINTER(c_void_p), POINTER(c_size_t)]
        self.__compress16.restype = c_int
        
        # tj3Decompress12 - decompress JPEG to 12-bit image
        self.__decompress12 = turbo_jpeg.tj3Decompress12
        self.__decompress12.argtypes = [
            c_void_p, POINTER(c_ubyte), c_size_t, POINTER(c_ushort), c_int, c_int]
        self.__decompress12.restype = c_int
        
        # tj3Decompress16 - decompress JPEG to 16-bit image
        self.__decompress16 = turbo_jpeg.tj3Decompress16
        self.__decompress16.argtypes = [
            c_void_p, POINTER(c_ubyte), c_size_t, POINTER(c_ushort), c_int, c_int]
        self.__decompress16.restype = c_int

        # tjGetScalingFactors
        get_scaling_factors = turbo_jpeg.tjGetScalingFactors
        get_scaling_factors.argtypes = [POINTER(c_int)]
        get_scaling_factors.restype = POINTER(ScalingFactor)
        num_scaling_factors = c_int()
        scaling_factors = get_scaling_factors(byref(num_scaling_factors))
        self.__scaling_factors = frozenset(
            (scaling_factors[i].num, scaling_factors[i].denom)
            for i in range(num_scaling_factors.value)
        )
        
        # tj3CompressFromYUV16 - compress 16-bit YUV to JPEG (TurboJPEG 3.1+)
        # These functions may not be available in all TurboJPEG 3.x versions
        try:
            self.__compressFromYUV16 = turbo_jpeg.tj3CompressFromYUV16
            self.__compressFromYUV16.argtypes = [
                c_void_p, POINTER(c_ushort), c_int, c_int, c_int,
                POINTER(c_void_p), POINTER(c_size_t)]
            self.__compressFromYUV16.restype = c_int
        except AttributeError:
            self.__compressFromYUV16 = None
        
        # tj3DecompressToYUV16 - decompress JPEG to 16-bit YUV (TurboJPEG 3.1+)
        try:
            self.__decompressToYUV16 = turbo_jpeg.tj3DecompressToYUV16
            self.__decompressToYUV16.argtypes = [
                c_void_p, POINTER(c_ubyte), c_size_t, POINTER(c_ushort), c_int]
            self.__decompressToYUV16.restype = c_int
        except AttributeError:
            self.__decompressToYUV16 = None
        
        # tj3DecompressToYUVPlanes16 - decompress JPEG to 16-bit YUV planes (TurboJPEG 3.1+)
        try:
            self.__decompressToYUVPlanes16 = turbo_jpeg.tj3DecompressToYUVPlanes16
            self.__decompressToYUVPlanes16.argtypes = [
                c_void_p, POINTER(c_ubyte), c_size_t, POINTER(POINTER(c_ushort)), POINTER(c_int)]
            self.__decompressToYUVPlanes16.restype = c_int
        except AttributeError:
            self.__decompressToYUVPlanes16 = None

        # tj3GetICCProfile - retrieve ICC profile from decompressor after header parsing (TurboJPEG 3.1+)
        try:
            self.__get_icc_profile = turbo_jpeg.tj3GetICCProfile
            self.__get_icc_profile.argtypes = [c_void_p, POINTER(c_void_p), POINTER(c_size_t)]
            self.__get_icc_profile.restype = c_int
        except AttributeError:
            self.__get_icc_profile = None

        # tj3SetICCProfile - attach ICC profile to compressor before compression (TurboJPEG 3.1+)
        try:
            self.__set_icc_profile = turbo_jpeg.tj3SetICCProfile
            self.__set_icc_profile.argtypes = [c_void_p, c_void_p, c_size_t]
            self.__set_icc_profile.restype = c_int
        except AttributeError:
            self.__set_icc_profile = None

        # MAXMEMORY and MAXPIXELS were added in libjpeg-turbo 3.0.2.  Keep
        # compatibility with 3.0.0/3.0.1, for which max_pixels can still be
        # enforced safely in Python before output allocation.
        self.__supports_native_resource_limits = \
            self.__detect_native_resource_limits()
        if self.__max_memory and not self.__supports_native_resource_limits \
                and not TurboJPEG.__resource_limit_warning_issued:
            warnings.warn(
                'libjpeg-turbo 3.0.2 or later is required to enforce '
                'max_memory; max_pixels and scan_limit remain enforced',
                RuntimeWarning,
                stacklevel=2,
            )
            TurboJPEG.__resource_limit_warning_issued = True

    def decode_header(self, jpeg_buf, return_precision=False):
        """decodes JPEG header and returns image properties as a tuple.
        
        Parameters
        ----------
        jpeg_buf : bytes
            JPEG image data buffer
        return_precision : bool, optional
            If True, returns precision as 5th element in tuple (default: False)
        
        Returns
        -------
        tuple
            By default: (width, height, jpeg_subsample, jpeg_colorspace)
            With return_precision=True: (width, height, jpeg_subsample, jpeg_colorspace, precision)
            
            - width: image width in pixels
            - height: image height in pixels
            - jpeg_subsample: chroma subsampling (TJSAMP_*)
            - jpeg_colorspace: colorspace (TJCS_*)
            - precision: bit precision (8, 12, or 16) - only when return_precision=True
        
        Examples
        --------
        >>> # Standard usage (backward compatible)
        >>> width, height, subsample, colorspace = jpeg.decode_header(jpeg_data)
        >>> 
        >>> # Get precision to select decode function
        >>> width, height, subsample, colorspace, precision = jpeg.decode_header(jpeg_data, return_precision=True)
        >>> if precision == 8:
        ...     img = jpeg.decode(jpeg_data)
        ... elif precision == 12:
        ...     img = jpeg.decode_12bit(jpeg_data)
        ... elif precision == 16:
        ...     img = jpeg.decode_16bit(jpeg_data)
        """
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            if self.__source_limits_enabled:
                self.__apply_source_limits(handle)
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            status = self.__decompress_header(handle, src_addr, jpeg_array.size)
            if status != 0:
                self.__report_error(handle)
            # Use tj3Get to retrieve header information
            width = self.__get(handle, TJPARAM_JPEGWIDTH)
            height = self.__get(handle, TJPARAM_JPEGHEIGHT)
            jpeg_subsample = self.__get(handle, TJPARAM_SUBSAMP)
            jpeg_colorspace = self.__get(handle, TJPARAM_COLORSPACE)
            # Check for errors (tj3Get returns -1 on error)
            if width < 0 or height < 0 or jpeg_subsample < 0 or jpeg_colorspace < 0:
                self.__report_error(handle)
            if self.__max_pixels:
                self.__enforce_max_pixels(handle, width, height)
            
            if return_precision:
                precision = self.__get(handle, TJPARAM_PRECISION)
                if precision < 0:
                    self.__report_error(handle)
                return (width, height, jpeg_subsample, jpeg_colorspace, precision)
            else:
                return (width, height, jpeg_subsample, jpeg_colorspace)
        finally:
            self.__destroy(handle)

    def get_icc_profile(self, jpeg_buf):
        """Extracts the embedded ICC color profile from a JPEG image.

        Requires TurboJPEG 3.1 or later with tj3GetICCProfile support.

        Parameters
        ----------
        jpeg_buf : bytes
            JPEG image data buffer containing an embedded ICC profile.

        Returns
        -------
        bytes or None
            Raw ICC profile data as a bytes object, or None if no ICC profile
            is present in the JPEG stream.

        Raises
        ------
        OSError
            If the JPEG header cannot be parsed or a fatal error occurs.
        NotImplementedError
            If the loaded libturbojpeg does not export tj3GetICCProfile.

        Examples
        --------
        >>> jpeg = TurboJPEG()
        >>> with open('photo_with_icc.jpg', 'rb') as f:
        ...     data = f.read()
        >>> icc = jpeg.get_icc_profile(data)
        >>> if icc:
        ...     print(f'ICC profile size: {len(icc)} bytes')
        """
        if self.__get_icc_profile is None:
            raise NotImplementedError(
                'tj3GetICCProfile is not available in the loaded libturbojpeg. '
                'Please upgrade to libjpeg-turbo 3.1 or later.')
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            if self.__source_limits_enabled:
                self.__apply_source_limits(handle)
            # Set TJPARAM_SAVEMARKERS to 2 (APP2) so the decompressor
            # retains ICC profile markers during header parsing.
            if self.__set(handle, TJPARAM_SAVEMARKERS, 2) != 0:
                self.__report_error(handle)
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            status = self.__decompress_header(handle, src_addr, jpeg_array.size)
            if status != 0:
                self.__report_error(handle)
            if self.__max_pixels:
                self.__enforce_max_pixels(handle)
            icc_buf = c_void_p()
            icc_size = c_size_t()
            status = self.__get_icc_profile(handle, byref(icc_buf), byref(icc_size))
            if status != 0:
                # A non-fatal return (e.g. no profile present) should return None
                err_code = self.__get_error_code(handle)
                if err_code == TJERR_WARNING:
                    return None
                self.__report_error(handle)
            if icc_buf.value is None or icc_size.value == 0:
                return None
            result = self.__copy_from_buffer(icc_buf.value, icc_size.value)
            self.__free(icc_buf)
            return result
        finally:
            self.__destroy(handle)

    def set_icc_profile(self, handle, icc_buf):
        """Attaches an ICC color profile to an active compressor handle.

        This is a low-level helper intended for use when building custom
        compression pipelines. In most cases, use encode() with the
        icc_profile parameter instead.

        Parameters
        ----------
        handle : ctypes void pointer
            An active TurboJPEG compressor handle (TJINIT_COMPRESS).
        icc_buf : bytes
            Raw ICC profile data to embed.

        Raises
        ------
        OSError
            If tj3SetICCProfile returns a non-zero status.
        NotImplementedError
            If the loaded libturbojpeg does not export tj3SetICCProfile.
        """
        if self.__set_icc_profile is None:
            raise NotImplementedError(
                'tj3SetICCProfile is not available in the loaded libturbojpeg. '
                'Please upgrade to libjpeg-turbo 3.1 or later.')
        icc_view = self.__get_buffer_view(
            icc_buf, "'icc_buf' argument", writable=False)
        icc_array = np.frombuffer(icc_view, dtype=np.uint8)
        icc_addr = self.__getaddr(icc_array)
        status = self.__set_icc_profile(handle, icc_addr, icc_view.nbytes)
        if status != 0:
            self.__report_error(handle)

    def decode(self, jpeg_buf, pixel_format=TJPF_BGR, scaling_factor=None, flags=0, dst=None):
        """decodes JPEG memory buffer to numpy array.
        
        Parameters
        ----------
        jpeg_buf : bytes
            JPEG image data to decode
        pixel_format : int
            Pixel format (TJPF_RGB, TJPF_BGR, etc.)
        scaling_factor : tuple or None
            Scaling factor as (num, denom) tuple
        flags : int
            Decompression flags
        dst : ndarray or None
            Destination array (optional). It must have the exact output shape,
            dtype uint8, C-contiguous storage, and writable memory.
            
        Returns
        -------
        ndarray
            Decoded image as numpy array (uint8)
        """
        pixel_format = self.__validate_pixel_format(pixel_format)
        flags = self.__validate_flags(
            flags, _PACKED_DECOMPRESS_FLAGS, 'decode()')
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            if flags or self.__source_limits_enabled:
                self.__apply_decompress_flags(handle, flags)
            
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, _, _ = \
                self.__get_header_and_dimensions(handle, jpeg_array.size, src_addr, scaling_factor)
            
            dtype = np.uint8
            expected_shape = (
                scaled_height, scaled_width, tjPixelSize[pixel_format])
            if dst is None:
                img_array = np.empty(
                    expected_shape, dtype=dtype)
            else:
                if not isinstance(dst, np.ndarray):
                    raise TypeError("'dst' argument must be a numpy array")
                if dst.shape != expected_shape:
                    raise ValueError(
                        "'dst' array must have shape {}".format(
                            expected_shape))
                if dst.dtype != dtype:
                    raise ValueError("'dst' array must have dtype uint8")
                if not dst.flags.c_contiguous:
                    raise ValueError("'dst' array must be C-contiguous")
                if not dst.flags.writeable:
                    raise ValueError("'dst' array must be writable")
                img_array = dst
            dest_addr = self.__getaddr(img_array)
            pitch = img_array.strides[0]
            status = self.__decompress(
                handle, src_addr, jpeg_array.size, dest_addr, pitch, pixel_format)
            
            if status != 0:
                self.__report_error(handle)
            return img_array
        finally:
            self.__destroy(handle)

    def decode_to_yuv(
            self, jpeg_buf, scaling_factor=None, pad=4, flags=0,
            return_metadata=False):
        """Decode JPEG data to a zero-initialized unified planar YUV buffer.

        By default the second return value is the backward-compatible list of
        ``(height, width)`` plane sizes.  If ``return_metadata`` is true, it is
        instead a list of :class:`YUVPlaneInfo` entries with explicit offsets,
        strides, valid widths, and heights.
        """
        if type(return_metadata) is not bool and not isinstance(
                return_metadata, np.bool_):
            raise TypeError("'return_metadata' argument must be a boolean")
        pad = self.__validate_alignment(pad, 'pad')
        if type(flags) is not int or flags != 0:
            flags = self.__validate_flags(
                flags, _YUV_DECOMPRESS_FLAGS, 'decode_to_yuv()')
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            if flags or self.__source_limits_enabled:
                self.__apply_decompress_flags(handle, flags)
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, jpeg_subsample, _ = \
                self.__get_header_and_dimensions(handle, jpeg_array.size, src_addr, scaling_factor)
            buffer_size = self.__buffer_size_YUV(scaled_width, pad, scaled_height, jpeg_subsample)
            if buffer_size == 0:
                self.__report_error(handle)
            buffer_array = np.zeros(buffer_size, dtype=np.uint8)
            dest_addr = self.__getaddr(buffer_array)
            status = self.__decompressToYUV(
                handle, src_addr, jpeg_array.size, dest_addr, pad)
            if status != 0:
                self.__report_error(handle)
            plane_sizes = self.__get_yuv_plane_sizes(
                scaled_width, scaled_height, jpeg_subsample)
            if return_metadata:
                offset = 0
                plane_metadata = []
                for height, width in plane_sizes:
                    stride = ((width + pad - 1) // pad) * pad
                    plane_metadata.append(YUVPlaneInfo(
                        offset=offset,
                        stride=stride,
                        width=width,
                        height=height,
                    ))
                    offset += stride * height
                if offset != buffer_array.size:
                    raise RuntimeError(
                        'Unexpected unified YUV buffer layout')
                return buffer_array, plane_metadata
            return buffer_array, plane_sizes
        finally:
            self.__destroy(handle)

    def decode_to_yuv_planes(self, jpeg_buf, scaling_factor=None, strides=(0, 0, 0), flags=0):
        """Decode JPEG data to zero-initialized planar YUV arrays."""
        if type(flags) is not int or flags != 0:
            flags = self.__validate_flags(
                flags, _YUV_DECOMPRESS_FLAGS, 'decode_to_yuv_planes()')
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            if flags or self.__source_limits_enabled:
                self.__apply_decompress_flags(handle, flags)
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, jpeg_subsample, _ = \
                self.__get_header_and_dimensions(handle, jpeg_array.size, src_addr, scaling_factor)
            num_planes = 1 if jpeg_subsample == TJSAMP_GRAY else 3
            try:
                strides = tuple(strides)
            except TypeError as exc:
                raise TypeError("'strides' argument must be a sequence") from exc
            if len(strides) < num_planes:
                raise ValueError(
                    "'strides' argument must provide {} value(s)".format(
                        num_planes))
            strides_addr = (c_int * num_planes)()
            dest_addr = (POINTER(c_ubyte) * num_planes)()
            planes = list()
            for i in range(num_planes):
                plane_width = self.__plane_width(
                    i, scaled_width, jpeg_subsample)
                plane_height = self.__plane_height(
                    i, scaled_height, jpeg_subsample)
                stride = self.__validate_resource_limit(
                    'strides[{}]'.format(i), strides[i])
                if stride == 0:
                    stride = plane_width
                if stride < plane_width:
                    raise ValueError(
                        "'strides[{}]' must be at least {}".format(
                            i, plane_width))
                strides_addr[i] = stride
                planes.append(np.zeros(
                    (plane_height, stride), dtype=np.uint8))
                dest_addr[i] = self.__getaddr(planes[i])
            status = self.__decompressToYUVPlanes(
                handle, src_addr, jpeg_array.size, dest_addr, strides_addr)
            if status != 0:
                self.__report_error(handle)
            return planes
        finally:
            self.__destroy(handle)

    def encode(self, img_array, quality=85, pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_422, flags=0, dst=None, lossless=False, icc_profile=None):
        """encodes numpy array to JPEG memory buffer.
        
        Parameters
        ----------
        img_array : ndarray
            Image data to encode (uint8)
        quality : int
            JPEG quality (1-100) - ignored if lossless=True
        pixel_format : int
            Pixel format (TJPF_RGB, TJPF_BGR, etc.)
        jpeg_subsample : int
            Chroma subsampling (TJSAMP_444, TJSAMP_422, etc.) - ignored if lossless=True
        flags : int
            Compression flags
        dst : writable contiguous buffer or None
            Destination buffer (optional). The buffer must be large enough for
            the worst-case JPEG size. If provided, TurboJPEG is not allowed to
            reallocate it.
        lossless : bool
            Enable lossless JPEG compression (default: False)
            When True, provides perfect reconstruction with larger file sizes.
            Note: quality and jpeg_subsample parameters are ignored in lossless mode;
            subsampling is automatically set to 4:4:4 by the library.
        icc_profile : bytes or None
            Raw ICC profile data to embed in the JPEG (optional).
            Requires TurboJPEG 3.1 or later with tj3SetICCProfile support.
            
        Returns
        -------
        bytes
            JPEG image data (lossy or lossless depending on lossless parameter)
        """
        # Avoid generic validator call overhead for the overwhelmingly common
        # tiny-image path while retaining the same dtype/shape/size checks.
        if (type(img_array) is np.ndarray and
                type(quality) is int and quality == 85 and
                type(pixel_format) is int and pixel_format == TJPF_BGR and
                type(jpeg_subsample) is int and
                jpeg_subsample == TJSAMP_422 and
                type(flags) is int and flags == 0 and lossless is False):
            if img_array.dtype != np.uint8:
                raise ValueError(
                    'encode() requires uint8 array with values in range 0-255')
            if img_array.ndim != 3 or img_array.shape[2] != 3:
                raise ValueError(
                    'Invalid shape for encode() with pixel_format={}: '
                    'expected 3 channel(s)'.format(pixel_format))
            height, width = img_array.shape[:2]
            if not 1 <= height <= _MAX_TJPARAM_VALUE:
                height = self.__validate_positive_integer('height', height)
            if not 1 <= width <= _MAX_TJPARAM_VALUE:
                width = self.__validate_positive_integer('width', width)
            img_array = np.ascontiguousarray(img_array)
        else:
            if not lossless:
                if not (type(quality) is int and 1 <= quality <= 100):
                    quality = self.__validate_quality(quality)
                if not (type(jpeg_subsample) is int and
                        TJSAMP_444 <= jpeg_subsample <= TJSAMP_441):
                    jpeg_subsample = self.__validate_subsampling(
                        jpeg_subsample)
            if type(flags) is not int or flags != 0:
                flags = self.__validate_flags(
                    flags, _PACKED_COMPRESS_FLAGS, 'encode()')
            if lossless and flags & (TJFLAG_PROGRESSIVE | _DCT_FLAGS):
                raise ValueError(
                    'Progressive and DCT flags are not supported for lossless '
                    'JPEG')
            img_array, height, width, pixel_format = self.__prepare_image(
                img_array, np.uint8, pixel_format, 'encode()')

        handle = self.__init(TJINIT_COMPRESS)
        try:
            if flags:
                self.__apply_compress_flags(handle, flags)
            if lossless:
                self.__set_parameter(handle, TJPARAM_LOSSLESS, 1)
                # In lossless mode, subsampling is automatically set to 4:4:4
                # and quality parameter is ignored
            else:
                self.__set_parameter(
                    handle, TJPARAM_SUBSAMP, jpeg_subsample)
                self.__set_parameter(handle, TJPARAM_QUALITY, quality)

            icc_size = 0
            if icc_profile is not None:
                icc_view = self.__get_buffer_view(
                    icc_profile, "'icc_profile' argument", writable=False)
                icc_size = icc_view.nbytes
                self.set_icc_profile(handle, icc_view)

            if dst is not None:
                dst_view = self.__get_buffer_view(
                    dst, "'dst' argument", writable=True)
                buffer_subsample = TJSAMP_444 if lossless else jpeg_subsample
                required_size = self.__buffer_size(
                    width, height, buffer_subsample) + icc_size
                if dst_view.nbytes < required_size:
                    raise ValueError(
                        "'dst' buffer is too small: requires at least {} bytes, "
                        "got {}".format(required_size, dst_view.nbytes))
                if self.__set(handle, TJPARAM_NOREALLOC, 1) != 0:
                    self.__report_error(handle)
                dst_array = np.frombuffer(dst_view, dtype=np.uint8)
                jpeg_buf = dst_array.ctypes.data_as(c_void_p)
                jpeg_size = c_size_t(dst_view.nbytes)
            else:
                dst_array = None
                jpeg_buf = c_void_p()
                jpeg_size = c_size_t()
            
            src_addr = self.__getaddr(img_array)
            try:
                status = self.__compress(
                    handle, src_addr, width, img_array.strides[0], height,
                    pixel_format, byref(jpeg_buf), byref(jpeg_size))

                if status != 0:
                    self.__report_error(handle)
                if dst_array is None:
                    return self.__copy_from_buffer(
                        jpeg_buf.value, jpeg_size.value)
                if jpeg_buf.value != dst_array.ctypes.data:
                    raise RuntimeError(
                        'TurboJPEG unexpectedly reallocated the destination '
                        'buffer')
                return dst, jpeg_size.value
            finally:
                # TurboJPEG owns buffers that it allocated or reallocated.
                # Never free the caller-provided destination buffer.
                if jpeg_buf.value is not None and (
                        dst_array is None or
                        jpeg_buf.value != dst_array.ctypes.data):
                    self.__free(jpeg_buf)
        finally:
            self.__destroy(handle)

    def encode_from_yuv(
            self, img_array, height, width, quality=85,
            jpeg_subsample=TJSAMP_420, flags=0, align=4):
        """Encode a unified planar uint8 YUV buffer to JPEG.

        Parameters
        ----------
        img_array : buffer
            C-contiguous uint8 buffer containing sequential Y, U, and V
            planes in TurboJPEG's unified planar layout.
        height, width : int
            Source image dimensions in pixels.
        quality : int
            JPEG quality from 1 to 100.
        jpeg_subsample : int
            Chroma subsampling used by the YUV layout.
        flags : int
            Compression flags.
        align : int
            Power-of-two row alignment used by the YUV layout.

        Returns
        -------
        bytes
            Encoded JPEG image.
        """
        height = self.__validate_positive_integer('height', height)
        width = self.__validate_positive_integer('width', width)
        align = self.__validate_alignment(align)
        quality = self.__validate_quality(quality)
        jpeg_subsample = self.__validate_subsampling(jpeg_subsample)
        flags = self.__validate_flags(
            flags, _YUV_COMPRESS_FLAGS, 'encode_from_yuv()')

        source_view = self.__get_buffer_view(
            img_array, "'img_array' argument", writable=False)
        source_array = np.asarray(source_view)
        if source_array.dtype != np.uint8:
            raise ValueError("'img_array' argument must have dtype uint8")

        # Reject obviously short buffers without calling tj3YUVBufSize().
        # This also avoids old libjpeg-turbo releases evaluating pathological
        # alignments that cannot possibly fit in the supplied source buffer.
        minimum_luma_stride = (
            (width + align - 1) // align) * align
        minimum_luma_size = minimum_luma_stride * height
        if source_view.nbytes < minimum_luma_size:
            raise ValueError(
                "'img_array' buffer is too small: requires at least {} bytes, "
                'got {}'.format(minimum_luma_size, source_view.nbytes))

        required_size = self.__buffer_size_YUV(
            width, align, height, jpeg_subsample)
        if required_size == 0:
            raise ValueError(
                'Invalid YUV dimensions, alignment, or subsampling')
        if source_view.nbytes < required_size:
            raise ValueError(
                "'img_array' buffer is too small: requires at least {} bytes, "
                'got {}'.format(required_size, source_view.nbytes))

        # Flatten the validated byte buffer without copying it.
        source_array = np.frombuffer(source_view, dtype=np.uint8)
        handle = self.__init(TJINIT_COMPRESS)
        try:
            if flags:
                self.__apply_compress_flags(handle, flags)
            self.__set_parameter(handle, TJPARAM_SUBSAMP, jpeg_subsample)
            self.__set_parameter(handle, TJPARAM_QUALITY, quality)
            
            jpeg_buf = c_void_p()
            jpeg_size = c_size_t()
            try:
                src_addr = self.__getaddr(source_array)
                status = self.__compressFromYUV(
                    handle, src_addr, width, align, height,
                    byref(jpeg_buf), byref(jpeg_size))
                if status != 0:
                    self.__report_error(handle)
                return self.__copy_from_buffer(
                    jpeg_buf.value, jpeg_size.value)
            finally:
                if jpeg_buf.value is not None:
                    self.__free(jpeg_buf)
        finally:
            self.__destroy(handle)

    def scale_with_quality(self, jpeg_buf, scaling_factor=None, quality=85, flags=0):
        """decompresstoYUV with scale factor, recompresstoYUV with quality factor"""
        quality = self.__validate_quality(quality)
        flags = self.__validate_flags(
            flags, _SCALE_FLAGS, 'scale_with_quality()')

        decompress_handle = self.__init(TJINIT_DECOMPRESS)
        try:
            decompress_flags = flags & _YUV_DECOMPRESS_FLAGS
            if decompress_flags or self.__source_limits_enabled:
                self.__apply_decompress_flags(
                    decompress_handle, decompress_flags)
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, jpeg_subsample, _ = self.__get_header_and_dimensions(
                decompress_handle, jpeg_array.size, src_addr, scaling_factor)
            buffer_YUV_size = self.__buffer_size_YUV(
                scaled_width, 4, scaled_height, jpeg_subsample)
            if buffer_YUV_size == 0:
                self.__report_error(decompress_handle)
            img_array = np.zeros(buffer_YUV_size, dtype=np.uint8)
            dest_addr = self.__getaddr(img_array)
            status = self.__decompressToYUV(
                decompress_handle, src_addr, jpeg_array.size, dest_addr, 4)
            if status != 0:
                self.__report_error(decompress_handle)
        finally:
            self.__destroy(decompress_handle)

        compress_handle = self.__init(TJINIT_COMPRESS)
        try:
            compress_flags = flags & _YUV_COMPRESS_FLAGS
            if compress_flags:
                self.__apply_compress_flags(compress_handle, compress_flags)
            self.__set_parameter(
                compress_handle, TJPARAM_SUBSAMP, jpeg_subsample)
            self.__set_parameter(compress_handle, TJPARAM_QUALITY, quality)
            jpeg_buf = c_void_p()
            jpeg_size = c_size_t()
            try:
                status = self.__compressFromYUV(
                    compress_handle, dest_addr, scaled_width, 4,
                    scaled_height, byref(jpeg_buf), byref(jpeg_size))
                if status != 0:
                    self.__report_error(compress_handle)
                return self.__copy_from_buffer(
                    jpeg_buf.value, jpeg_size.value)
            finally:
                if jpeg_buf.value is not None:
                    self.__free(jpeg_buf)
        finally:
            self.__destroy(compress_handle)
    
    def encode_12bit(self, img_array, quality=85, pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_422, flags=0, lossless=False):
        """Encodes 12-bit numpy array (uint16) to JPEG memory buffer.
        
        Parameters
        ----------
        img_array : ndarray
            12-bit image data (uint16, values 0-4095)
        quality : int
            JPEG quality (1-100) - ignored if lossless=True
        pixel_format : int
            Pixel format (TJPF_RGB, TJPF_BGR, etc.)
        jpeg_subsample : int
            Chroma subsampling (TJSAMP_444, TJSAMP_422, etc.) - ignored if lossless=True
        flags : int
            Compression flags
        lossless : bool
            Enable lossless JPEG compression (default: False)
            When True, provides perfect reconstruction with larger file sizes
            
        Returns
        -------
        bytes
            JPEG image data (lossy or lossless depending on lossless parameter)
        """
        if not lossless:
            quality = self.__validate_quality(quality)
            jpeg_subsample = self.__validate_subsampling(jpeg_subsample)
        flags = self.__validate_flags(
            flags, _PACKED_COMPRESS_FLAGS, 'encode_12bit()')
        if lossless and flags & (TJFLAG_PROGRESSIVE | _DCT_FLAGS):
            raise ValueError(
                'Progressive and DCT flags are not supported for lossless '
                'JPEG')
        img_array, height, width, pixel_format = self.__prepare_image(
            img_array, np.uint16, pixel_format, 'encode_12bit()',
            maximum_value=4095)

        handle = self.__init(TJINIT_COMPRESS)
        try:
            if flags:
                self.__apply_compress_flags(handle, flags)
            if lossless:
                self.__set_parameter(handle, TJPARAM_LOSSLESS, 1)
                # In lossless mode, subsampling is automatically set to 4:4:4
                # and quality parameter is ignored
            else:
                self.__set_parameter(
                    handle, TJPARAM_SUBSAMP, jpeg_subsample)
                self.__set_parameter(handle, TJPARAM_QUALITY, quality)
            
            jpeg_buf = c_void_p()
            jpeg_size = c_size_t()
            try:
                src_addr = self.__getaddr_uint16(img_array)
                # High-precision TurboJPEG strides are measured in samples.
                stride_samples = img_array.strides[0] // 2
                status = self.__compress12(
                    handle, src_addr, width, stride_samples, height,
                    pixel_format, byref(jpeg_buf), byref(jpeg_size))
                if status != 0:
                    self.__report_error(handle)
                return self.__copy_from_buffer(
                    jpeg_buf.value, jpeg_size.value)
            finally:
                if jpeg_buf.value is not None:
                    self.__free(jpeg_buf)
        finally:
            self.__destroy(handle)
    
    def encode_16bit(self, img_array, pixel_format=TJPF_BGR, flags=0):
        """Encodes 16-bit numpy array (uint16) to lossless JPEG memory buffer.
        
        **Note:** 16-bit precision requires lossless JPEG compression per the JPEG standard.
        This method automatically enables lossless mode (4:4:4 subsampling, no lossy compression).
        
        Parameters
        ----------
        img_array : ndarray
            16-bit image data (uint16, values 0-65535)
        pixel_format : int
            Pixel format (TJPF_RGB, TJPF_BGR, etc.)
        flags : int
            Compression flags
            
        Returns
        -------
        bytes
            Lossless JPEG image data
        """
        flags = self.__validate_flags(
            flags, _PACKED_COMPRESS_FLAGS, 'encode_16bit()')
        if flags & (TJFLAG_PROGRESSIVE | _DCT_FLAGS):
            raise ValueError(
                'Progressive and DCT flags are not supported for lossless '
                'JPEG')
        img_array, height, width, pixel_format = self.__prepare_image(
            img_array, np.uint16, pixel_format, 'encode_16bit()')

        handle = self.__init(TJINIT_COMPRESS)
        try:
            if flags:
                self.__apply_compress_flags(handle, flags)
            self.__set_parameter(handle, TJPARAM_LOSSLESS, 1)
            
            jpeg_buf = c_void_p()
            jpeg_size = c_size_t()
            try:
                src_addr = self.__getaddr_uint16(img_array)
                stride_samples = img_array.strides[0] // 2
                status = self.__compress16(
                    handle, src_addr, width, stride_samples, height,
                    pixel_format, byref(jpeg_buf), byref(jpeg_size))
                if status != 0:
                    self.__report_error(handle)
                return self.__copy_from_buffer(
                    jpeg_buf.value, jpeg_size.value)
            finally:
                if jpeg_buf.value is not None:
                    self.__free(jpeg_buf)
        finally:
            self.__destroy(handle)
    
    def decode_12bit(self, jpeg_buf, pixel_format=TJPF_BGR, scaling_factor=None, flags=0):
        """Decodes JPEG memory buffer to 12-bit numpy array (uint16).
        
        Parameters
        ----------
        jpeg_buf : bytes
            JPEG image data to decode
        pixel_format : int
            Pixel format (TJPF_RGB, TJPF_BGR, etc.)
        scaling_factor : tuple or None
            Scaling factor as (num, denom) tuple
        flags : int
            Decompression flags
            
        Returns
        -------
        ndarray
            12-bit image as uint16 numpy array (values 0-4095)
        """
        pixel_format = self.__validate_pixel_format(pixel_format)
        flags = self.__validate_flags(
            flags, _PACKED_DECOMPRESS_FLAGS, 'decode_12bit()')
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            if flags or self.__source_limits_enabled:
                self.__apply_decompress_flags(handle, flags)
            
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, _, _ = \
                self.__get_header_and_dimensions(handle, jpeg_array.size, src_addr, scaling_factor)
            
            # 12-bit precision
            dtype = np.uint16
            img_array = np.empty(
                [scaled_height, scaled_width, tjPixelSize[pixel_format]],
                dtype=dtype)
            dest_addr = self.__getaddr_uint16(img_array)
            # pitch should be width * samples_per_pixel (not bytes)
            pitch = scaled_width * tjPixelSize[pixel_format]
            
            status = self.__decompress12(
                handle, src_addr, jpeg_array.size, dest_addr, pitch, pixel_format)
            
            if status != 0:
                self.__report_error(handle)
            return img_array
        finally:
            self.__destroy(handle)
    
    def decode_16bit(self, jpeg_buf, pixel_format=TJPF_BGR, scaling_factor=None, flags=0):
        """Decodes lossless 16-bit JPEG memory buffer to 16-bit numpy array (uint16).
        
        **Note:** This method decodes lossless 16-bit JPEG images created with encode_16bit().
        The JPEG standard only supports 16-bit precision for lossless compression.
        
        Parameters
        ----------
        jpeg_buf : bytes
            JPEG image data to decode (must be a lossless 16-bit JPEG)
        pixel_format : int
            Pixel format (TJPF_RGB, TJPF_BGR, etc.)
        scaling_factor : tuple or None
            Scaling factor as (num, denom) tuple (may not be supported for lossless)
        flags : int
            Decompression flags
            
        Returns
        -------
        ndarray
            16-bit image as uint16 numpy array (values 0-65535)
            
        Raises
        ------
        IOError or OSError
            If the JPEG is not a 16-bit lossless JPEG image
        """
        pixel_format = self.__validate_pixel_format(pixel_format)
        flags = self.__validate_flags(
            flags, _PACKED_DECOMPRESS_FLAGS, 'decode_16bit()')
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            if flags or self.__source_limits_enabled:
                self.__apply_decompress_flags(handle, flags)
            
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, _, _ = \
                self.__get_header_and_dimensions(handle, jpeg_array.size, src_addr, scaling_factor)
            
            # 16-bit precision
            dtype = np.uint16
            img_array = np.empty(
                [scaled_height, scaled_width, tjPixelSize[pixel_format]],
                dtype=dtype)
            dest_addr = self.__getaddr_uint16(img_array)
            # pitch should be width * samples_per_pixel (not bytes)
            pitch = scaled_width * tjPixelSize[pixel_format]
            
            status = self.__decompress16(
                handle, src_addr, jpeg_array.size, dest_addr, pitch, pixel_format)
            
            if status != 0:
                self.__report_error(handle)
            return img_array
        finally:
            self.__destroy(handle)

    def crop(self, jpeg_buf, x, y, w, h, preserve=False, gray=False, copynone=False):
        """Losslessly crop a JPEG image with optional grayscale conversion.

        The native transform requires an iMCU-aligned origin. With
        ``preserve=False``, the origin is rounded down and the output expands
        to retain the complete requested region. With ``preserve=True``, the
        origin is rounded up and the output remains inside that region.
        Partial iMCUs at the source's right and bottom edges are retained.
        """
        x = self.__validate_resource_limit('x', x)
        y = self.__validate_resource_limit('y', y)
        w = self.__validate_positive_integer('w', w)
        h = self.__validate_positive_integer('h', h)
        handle = self.__init(TJINIT_TRANSFORM)
        try:
            if self.__source_limits_enabled:
                self.__apply_source_limits(handle)
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            # Get header information using tj3DecompressHeader
            status = self.__decompress_header(handle, src_addr, jpeg_array.size)
            if status != 0:
                self.__report_error(handle)
            width = self.__get(handle, TJPARAM_JPEGWIDTH)
            height = self.__get(handle, TJPARAM_JPEGHEIGHT)
            jpeg_subsample = self.__get(handle, TJPARAM_SUBSAMP)
            if self.__max_pixels:
                self.__enforce_max_pixels(handle, width, height)
            jpeg_subsample = self.__validate_subsampling(jpeg_subsample)
            
            x, w = self.__axis_to_image_boundaries(
                x, w, width, preserve, tjMCUWidth[jpeg_subsample])
            y, h = self.__axis_to_image_boundaries(
                y, h, height, preserve, tjMCUHeight[jpeg_subsample])
            region = CroppingRegion(x, y, w, h)
            # Use array initialization to ensure all fields are properly zero-initialized
            crop_transforms = (TransformStruct * 1)()
            crop_transforms[0].r = region
            crop_transforms[0].op = TJXOP_NONE
            crop_transforms[0].options = TJXOPT_CROP | (gray and TJXOPT_GRAY) | (copynone and TJXOPT_COPYNONE)
            return self.__do_transform(handle, src_addr, jpeg_array.size, 1, crop_transforms)[0]

        finally:
            self.__destroy(handle)

    def crop_multiple(self, jpeg_buf, crop_parameters, background_luminance=1.0, gray=False, copynone=False):
        """Lossless crop and/or extension operations on jpeg image.
        Crop origin(s) needs be divisable by the MCU block size and inside
        the input image, or OSError: Invalid crop request is raised.

        Parameters
        ----------
        jpeg_buf: bytes
            Input jpeg image.
        crop_parameters: List[Tuple[int, int, int, int]]
            List of crop parameters defining start x and y origin and width
            and height of each crop operation.
        background_luminance: float
            Luminance level (0 -1 ) to fill background when extending image.
            Default to 1, resulting in white background.
        gray: bool
            Produce greyscale output
        copynone: bool
            True = do not copy EXIF data (False by default)

        Returns
        ----------
        List[bytes]
            Cropped and/or extended jpeg images.
        """
        handle = self.__init(TJINIT_TRANSFORM)
        try:
            if self.__source_limits_enabled:
                self.__apply_source_limits(handle)
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)

            # Decompress header to get input image size and subsample value
            decompress_header_status = self.__decompress_header(
                handle,
                src_addr,
                jpeg_array.size
            )

            if decompress_header_status != 0:
                self.__report_error(handle)
            
            image_width = self.__get(handle, TJPARAM_JPEGWIDTH)
            image_height = self.__get(handle, TJPARAM_JPEGHEIGHT)
            jpeg_subsample = self.__get(handle, TJPARAM_SUBSAMP)
            if self.__max_pixels:
                self.__enforce_max_pixels(
                    handle, image_width, image_height)
            jpeg_subsample = self.__validate_subsampling(jpeg_subsample)

            if isinstance(background_luminance, (bool, np.bool_)) or \
                    not isinstance(
                        background_luminance,
                        (int, float, np.integer, np.floating)):
                raise TypeError(
                    "'background_luminance' argument must be a number")
            background_luminance = float(background_luminance)
            if not np.isfinite(background_luminance) or not \
                    0.0 <= background_luminance <= 1.0:
                raise ValueError(
                    "'background_luminance' argument must be between 0 and 1")

            # Define cropping regions from input parameters and image size
            crop_regions = self.__define_cropping_regions(crop_parameters)
            number_of_operations = len(crop_regions)
            if number_of_operations == 0:
                raise ValueError("'crop_parameters' argument must not be empty")

            # Define crop transforms from cropping_regions
            crop_transforms = (TransformStruct * number_of_operations)()
            # Pre-compute luminance coefficient once for all crops
            lum_coefficient = None
            callback_data_refs = []
            callback_refs = []
            for i, crop_region in enumerate(crop_regions):
                if crop_region.x >= image_width or \
                        crop_region.y >= image_height:
                    raise ValueError(
                        'Crop origin must be inside the source image')
                if crop_region.x % tjMCUWidth[jpeg_subsample] or \
                        crop_region.y % tjMCUHeight[jpeg_subsample]:
                    raise ValueError(
                        'Crop origin must be aligned to the JPEG iMCU size')
                # The fill_background callback is slow, only use it if needed
                if self.__need_fill_background(
                    crop_region,
                    (image_width, image_height),
                    background_luminance
                ):
                    if lum_coefficient is None:
                        lum_coefficient = self.__map_luminance_to_dc_dct_coefficient(
                            bytearray(jpeg_buf),
                            background_luminance
                        )
                    # Use callback to fill in background post-transform
                    callback_data = BackgroundStruct(
                        image_width,
                        image_height,
                        lum_coefficient
                    )
                    callback = CUSTOMFILTER(fill_background)
                    callback_data_refs.append(callback_data)
                    callback_refs.append(callback)
                    crop_transforms[i] = TransformStruct(
                        crop_region,
                        TJXOP_NONE,
                        TJXOPT_PERFECT | TJXOPT_CROP | (gray and TJXOPT_GRAY) | (copynone and TJXOPT_COPYNONE),
                        pointer(callback_data),
                        callback
                    )
                else:
                    crop_transforms[i] = TransformStruct(
                        crop_region,
                        TJXOP_NONE,
                        TJXOPT_PERFECT | TJXOPT_CROP | (gray and TJXOPT_GRAY) | (copynone and TJXOPT_COPYNONE)
                    )
            results = self.__do_transform(handle, src_addr, jpeg_array.size, number_of_operations, crop_transforms)

            return results

        finally:
            self.__destroy(handle)

    def optimize(self, jpeg_buf, copynone=False):
        """Losslessly optimize the Huffman tables of a jpeg image.

        Re-encodes the entropy-coded data with optimal Huffman tables
        (equivalent to ``jpegtran -optimize``) without any loss in image
        quality. Typically reduces file size, unless the input is already
        optimized.

        Parameters
        ----------
        jpeg_buf: bytes
            Input jpeg image.
        copynone: bool
            True = do not copy EXIF data (False by default)

        Returns
        ----------
        bytes
            Huffman-optimized jpeg image.
        """
        handle = self.__init(TJINIT_TRANSFORM)
        try:
            if self.__source_limits_enabled:
                self.__apply_source_limits(handle)
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            status = self.__decompress_header(handle, src_addr, jpeg_array.size)
            if status != 0:
                self.__report_error(handle)
            if self.__max_pixels:
                self.__enforce_max_pixels(handle)
            # Without TJXOPT_CROP the cropping region is ignored and the whole
            # image is transformed, so the (zeroed) region needs no setup.
            transforms = (TransformStruct * 1)()
            transforms[0].op = TJXOP_NONE
            transforms[0].options = TJXOPT_OPTIMIZE | (copynone and TJXOPT_COPYNONE)
            return self.__do_transform(handle, src_addr, jpeg_array.size, 1, transforms)[0]

        finally:
            self.__destroy(handle)

    def buffer_size(self, img_array, jpeg_subsample=TJSAMP_422):
        """Get maximum number of bytes of compressed jpeg data"""
        jpeg_subsample = self.__validate_subsampling(jpeg_subsample)
        image = np.asarray(img_array)
        if image.ndim < 2:
            raise ValueError('Invalid shape for image data')
        height = self.__validate_positive_integer('height', image.shape[0])
        width = self.__validate_positive_integer('width', image.shape[1])
        return self.__buffer_size(width, height, jpeg_subsample)

    def __do_transform(self, handle, src_buf, src_size, number_of_transforms, transforms):
        """Do transform.

        Parameters
        ----------
        handle: int
            Initiated transform handle.
        src_buf: LP_c_ubyte
            Pointer to source buffer for transform
        src_size: int
            Size of source buffer.
        number_of_transforms: int
            Number of transforms to perform.
        transforms: CArgObject
            C-array of transforms to perform.

        Returns
        ----------
        List[bytes]
            Cropped and/or extended jpeg images.
        """
        # Pointers to output image buffers
        dest_array = (c_void_p * number_of_transforms)()
        try:
            dest_size = (c_size_t * number_of_transforms)()
            transform_status = self.__transform(
                handle,
                src_buf,
                src_size,
                number_of_transforms,
                dest_array,
                dest_size,
                transforms,
            )

            if transform_status != 0:
                self.__report_error(handle)
             # Copy the transform results into python bytes
            return [
                self.__copy_from_buffer(dest_array[i], dest_size[i])
                for i in range(number_of_transforms)
            ]
        finally:
            # Free the output image buffers
            for dest in dest_array:
                self.__free(dest)

    @staticmethod
    def __copy_from_buffer(buffer, size):
        """Copy bytes from buffer to python bytes."""
        return string_at(buffer, size)

    @staticmethod
    def __get_yuv_plane_sizes(width, height, jpeg_subsample):
        """Calculate native YUV plane dimensions from sampling factors."""
        horizontal = tjMCUWidth[jpeg_subsample] // MCU_WIDTH
        vertical = tjMCUHeight[jpeg_subsample] // MCU_HEIGHT
        chroma_width = (width + horizontal - 1) // horizontal
        chroma_height = (height + vertical - 1) // vertical
        plane_sizes = [(
            chroma_height * vertical,
            chroma_width * horizontal,
        )]
        if jpeg_subsample != TJSAMP_GRAY:
            plane_sizes.extend((
                (chroma_height, chroma_width),
                (chroma_height, chroma_width),
            ))
        return plane_sizes

    def __set_parameter(self, handle, parameter, value):
        """Set a TurboJPEG parameter and surface every native failure."""
        if self.__set(handle, parameter, value) != 0:
            self.__report_error(handle)

    @staticmethod
    def __validate_flags(flags, supported, operation):
        """Validate a legacy flag bitmask for a specific public operation."""
        if type(flags) is int and flags == 0:
            return 0
        if isinstance(flags, (bool, np.bool_)) or not isinstance(
                flags, (int, np.integer)):
            raise TypeError("'flags' argument must be a non-negative integer")
        flags = int(flags)
        if flags < 0 or flags > _MAX_TJPARAM_VALUE:
            raise ValueError(
                "'flags' argument must be between 0 and {}".format(
                    _MAX_TJPARAM_VALUE))
        unsupported = flags & ~supported
        if unsupported:
            raise ValueError(
                "Unsupported flags for {}: 0x{:x}".format(
                    operation, unsupported))
        if flags & TJFLAG_FASTDCT and flags & TJFLAG_ACCURATEDCT:
            raise ValueError(
                'TJFLAG_FASTDCT and TJFLAG_ACCURATEDCT are mutually '
                'exclusive')
        return flags

    def __apply_decompress_flags(self, handle, flags):
        """Map supported decompression flags to tj3 parameters."""
        if self.__source_limits_enabled or flags & TJFLAG_LIMITSCANS:
            self.__apply_source_limits(handle, flags)
        if flags == 0:
            return flags
        parameters = (
            (TJPARAM_BOTTOMUP, TJFLAG_BOTTOMUP),
            (TJPARAM_FASTUPSAMPLE, TJFLAG_FASTUPSAMPLE),
            (TJPARAM_STOPONWARNING, TJFLAG_STOPONWARNING),
        )
        for parameter, flag in parameters:
            if flags & flag:
                self.__set_parameter(handle, parameter, 1)
        if flags & TJFLAG_FASTDCT:
            self.__set_parameter(handle, TJPARAM_FASTDCT, 1)
        elif flags & TJFLAG_ACCURATEDCT:
            self.__set_parameter(handle, TJPARAM_FASTDCT, 0)
        return flags

    def __apply_compress_flags(self, handle, flags):
        """Map supported compression flags to tj3 parameters."""
        if flags == 0:
            return flags
        parameters = (
            (TJPARAM_BOTTOMUP, TJFLAG_BOTTOMUP),
            (TJPARAM_STOPONWARNING, TJFLAG_STOPONWARNING),
            (TJPARAM_PROGRESSIVE, TJFLAG_PROGRESSIVE),
        )
        for parameter, flag in parameters:
            if flags & flag:
                self.__set_parameter(handle, parameter, 1)
        if flags & TJFLAG_FASTDCT:
            self.__set_parameter(handle, TJPARAM_FASTDCT, 1)
        elif flags & TJFLAG_ACCURATEDCT:
            self.__set_parameter(handle, TJPARAM_FASTDCT, 0)
        return flags

    def __apply_source_limits(self, handle, flags=0):
        """Apply decompression/transform resource limits to a native handle."""
        if not (self.__source_limits_enabled or flags & TJFLAG_LIMITSCANS):
            return

        scan_limit = self.__scan_limit
        if flags & TJFLAG_LIMITSCANS and (
                scan_limit == 0 or scan_limit > _LEGACY_SCAN_LIMIT):
            scan_limit = _LEGACY_SCAN_LIMIT

        parameters = []
        if self.__supports_native_resource_limits:
            if self.__max_memory:
                parameters.append((TJPARAM_MAXMEMORY, self.__max_memory))
            if self.__max_pixels:
                parameters.append((TJPARAM_MAXPIXELS, self.__max_pixels))
        if scan_limit:
            parameters.append((TJPARAM_SCANLIMIT, scan_limit))
        for parameter, value in parameters:
            if self.__set(handle, parameter, value) != 0:
                self.__report_error(handle)

    def __detect_native_resource_limits(self):
        """Return whether MAXMEMORY/MAXPIXELS are supported by the library."""
        handle = self.__init(TJINIT_DECOMPRESS)
        if not handle:
            raise RuntimeError('Unable to initialize TurboJPEG decompressor')
        try:
            if self.__set(handle, TJPARAM_MAXMEMORY, 0) != 0:
                return False
            return self.__set(handle, TJPARAM_MAXPIXELS, 0) == 0
        finally:
            self.__destroy(handle)

    def __enforce_max_pixels(self, handle, width=None, height=None):
        """Reject oversized headers before allocating Python output memory."""
        if self.__max_pixels == 0:
            return
        if width is None:
            width = self.__get(handle, TJPARAM_JPEGWIDTH)
        if height is None:
            height = self.__get(handle, TJPARAM_JPEGHEIGHT)
        if width < 0 or height < 0:
            self.__report_error(handle)

        pixel_count = width * height
        if pixel_count > self.__max_pixels:
            raise ValueError(
                'JPEG image contains {} pixels, exceeding max_pixels={}'
                .format(pixel_count, self.__max_pixels))

    def __get_header_and_dimensions(self, handle, jpeg_array_size, src_addr, scaling_factor):
        """returns scaled image dimensions and header data"""
        if scaling_factor is not None and \
            scaling_factor not in self.__scaling_factors:
            raise ValueError('supported scaling factors are ' +
                str(self.__scaling_factors))
        
        # Decompress header first to get dimensions
        status = self.__decompress_header(handle, src_addr, jpeg_array_size)
        if status != 0:
            self.__report_error(handle)
        
        # Get unscaled header information using tj3Get
        width = self.__get(handle, TJPARAM_JPEGWIDTH)
        height = self.__get(handle, TJPARAM_JPEGHEIGHT)
        jpeg_subsample = self.__get(handle, TJPARAM_SUBSAMP)
        jpeg_colorspace = self.__get(handle, TJPARAM_COLORSPACE)
        
        # Check for errors (tj3Get returns -1 on error)
        if width < 0 or height < 0 or jpeg_subsample < 0 or jpeg_colorspace < 0:
            self.__report_error(handle)
        if self.__max_pixels:
            self.__enforce_max_pixels(handle, width, height)
        
        # Set scaling factor if provided - must be done AFTER reading header.
        # TurboJPEG ignores scaling for lossless JPEG images, so calculating a
        # smaller destination in that case would allow the native decoder to
        # write beyond the allocated numpy array.
        scaled_width = width
        scaled_height = height
        if scaling_factor is not None:
            lossless = self.__get(handle, TJPARAM_LOSSLESS)
            if lossless < 0:
                self.__report_error(handle)
            if lossless:
                if scaling_factor != (1, 1):
                    raise ValueError(
                        'Decompression scaling is not supported for lossless '
                        'JPEG images')
                return (scaled_width, scaled_height, jpeg_subsample,
                        jpeg_colorspace)
            num, denom = scaling_factor[0], scaling_factor[1]
            sf = ScalingFactor()
            sf.num = num
            sf.denom = denom
            status = self.__set_scaling_factor(handle, sf)
            if status != 0:
                self.__report_error(handle)
            # Calculate scaled dimensions manually
            def get_scaled_value(dim, n, d):
                return (dim * n + d - 1) // d
            scaled_width = get_scaled_value(width, num, denom)
            scaled_height = get_scaled_value(height, num, denom)
        
        return scaled_width, scaled_height, jpeg_subsample, jpeg_colorspace

    def __axis_to_image_boundaries(
            self, a, b, img_boundary, preserve, mcuBlock):
        """Align a crop origin while retaining the source's partial edge."""
        if a >= img_boundary:
            raise ValueError('Crop origin must be inside the source image')
        requested_end = min(a + b, img_boundary)
        if preserve:
            a = ((a + mcuBlock - 1) // mcuBlock) * mcuBlock
        else:
            a = (a // mcuBlock) * mcuBlock
        b = requested_end - a
        if b <= 0:
            raise ValueError(
                'Crop region contains no pixels after iMCU alignment')
        return a, b

    @classmethod
    def __define_cropping_regions(cls, crop_parameters):
        """Return list of crop regions from crop parameters

        Parameters
        ----------
        crop_parameters: List[Tuple[int, int, int, int]]
            List of crop parameters defining start x and y origin and width
            and height of each crop operation.

        Returns
        ----------
        List[CroppingRegion]
            List of crop operations, size is equal to the product of number of
            crop operations to perform in x and y direction.
        """
        try:
            crop_parameters = list(crop_parameters)
        except TypeError as exc:
            raise TypeError(
                "'crop_parameters' argument must be an iterable") from exc
        regions = []
        for index, crop in enumerate(crop_parameters):
            try:
                if len(crop) != 4:
                    raise ValueError
                x, y, w, h = crop
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    'Crop parameter {} must contain (x, y, w, h)'.format(
                        index)) from exc
            regions.append(CroppingRegion(
                x=cls.__validate_resource_limit(
                    'crop_parameters[{}].x'.format(index), x),
                y=cls.__validate_resource_limit(
                    'crop_parameters[{}].y'.format(index), y),
                w=cls.__validate_resource_limit(
                    'crop_parameters[{}].w'.format(index), w),
                h=cls.__validate_resource_limit(
                    'crop_parameters[{}].h'.format(index), h),
            ))
        return regions

    @staticmethod
    def __need_fill_background(crop_region, image_size, background_luminance):
        """Return true if crop operation require background fill operation.

        Parameters
        ----------
        crop_region: CroppingRegion
            The crop region to check.
        image_size: [int, int]
            Size of input image.
        background_luminance: float
            Requested background luminance.

        Returns
        ----------
        bool
            True if crop operation require background fill operation.
        """
        return (
            (
                (crop_region.x + crop_region.w > image_size[0])
                or
                (crop_region.y + crop_region.h > image_size[1])
            )
            and (background_luminance != 0.5)
        )

    @staticmethod
    def __iter_jpeg_header_segments(jpeg_data):
        """Yield marker and payload boundaries before entropy-coded data."""
        jpeg_data = bytes(jpeg_data)
        offset = 0
        standalone_markers = set(range(0xD0, 0xDA)) | {0x01}
        while offset < len(jpeg_data):
            marker_start = jpeg_data.find(b'\xFF', offset)
            if marker_start == -1:
                return
            marker_offset = marker_start + 1
            while marker_offset < len(jpeg_data) and \
                    jpeg_data[marker_offset] == 0xFF:
                marker_offset += 1
            if marker_offset >= len(jpeg_data):
                raise ValueError('Truncated JPEG marker')

            marker = jpeg_data[marker_offset]
            offset = marker_offset + 1
            if marker == 0x00:
                continue
            if marker in standalone_markers:
                continue
            # Header metadata needed by crop transforms must precede the first
            # scan.  Do not interpret marker-like bytes in entropy-coded data.
            if marker == 0xDA:
                return
            if offset + 2 > len(jpeg_data):
                raise ValueError('Truncated JPEG segment')
            segment_length = unpack('>H', jpeg_data[offset:offset + 2])[0]
            if segment_length < 2:
                raise ValueError('Invalid JPEG segment length')
            segment_end = offset + segment_length
            if segment_end > len(jpeg_data):
                raise ValueError('Truncated JPEG segment')
            yield marker, offset + 2, segment_end
            offset = segment_end

    @classmethod
    def __find_dqt(cls, jpeg_data, dqt_index):
        """Return the table-info offset for a DQT table in JPEG data.

        Parameters
        ----------
        jpeg_data: bytes
            Jpeg data.
        dqt_index: int
            Index of quantificatin table to find (0 - luminance).

        Returns
        ----------
        Optional[int]
            Byte offset to the matching table's Pq/Tq byte, or None.
        """
        jpeg_data = bytes(jpeg_data)
        matching_offset = None
        for marker, segment_start, segment_end in \
                cls.__iter_jpeg_header_segments(jpeg_data):
            if marker == 0xDB:
                if segment_start >= segment_end:
                    raise ValueError('Invalid DQT segment length')
                table_offset = segment_start
                while table_offset < segment_end:
                    precision, table_index = split_byte_into_nibbles(
                        jpeg_data[table_offset])
                    if precision not in (0, 1):
                        raise ValueError(
                            'Not valid precision definition in DQT')
                    element_size = 1 if precision == 0 else 2
                    next_table = table_offset + 1 + 64 * element_size
                    if next_table > segment_end:
                        raise ValueError('Truncated quantization table')
                    if table_index == dqt_index:
                        # A later DQT definition supersedes an earlier one.
                        matching_offset = table_offset
                    table_offset = next_table
        return matching_offset

    @classmethod
    def __get_frame_precision_and_luminance_table(cls, jpeg_data):
        """Return sample precision and first component's DQT selector."""
        jpeg_data = bytes(jpeg_data)
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3,
            0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB,
            0xCD, 0xCE, 0xCF,
        }
        for marker, segment_start, segment_end in \
                cls.__iter_jpeg_header_segments(jpeg_data):
            if marker not in sof_markers:
                continue
            if segment_end - segment_start < 9:
                raise ValueError('Truncated JPEG frame header')
            component_count = jpeg_data[segment_start + 5]
            expected_size = 6 + 3 * component_count
            if component_count == 0 or \
                    segment_start + expected_size > segment_end:
                raise ValueError('Truncated JPEG frame components')
            precision = jpeg_data[segment_start]
            table_index = jpeg_data[segment_start + 8]
            if table_index > 3:
                raise ValueError(
                    'Invalid luminance quantization table selector')
            return precision, table_index
        raise ValueError('JPEG frame header not found')

    @classmethod
    def __get_dc_dqt_element(cls, jpeg_data, dqt_index):
        """Return dc quantification element from jpeg_data for quantification
        table dqt_index.

        Parameters
        ----------
        jpeg_data: bytes
            Jpeg data containing quantification table(s).
        dqt_index: int
            Index of quantificatin table to get (0 - luminance).

        Returns
        ----------
        int
            Dc quantification element.
        """
        table_offset = cls.__find_dqt(jpeg_data, dqt_index)
        if table_offset is None:
            raise ValueError(
                "Quantisation table {dqt_index} not found in header".format(
                    dqt_index=dqt_index)
            )
        precision = split_byte_into_nibbles(jpeg_data[table_offset])[0]
        if precision == 0:
            unpack_type = '>B'
        elif precision == 1:
            unpack_type = '>H'
        else:
            raise ValueError('Not valid precision definition in DQT')
        dc_offset = table_offset + 1
        dc_length = 1 if precision == 0 else 2
        dc_value = unpack(
            unpack_type,
            jpeg_data[dc_offset:dc_offset+dc_length]
        )[0]
        return dc_value

    @classmethod
    def __map_luminance_to_dc_dct_coefficient(cls, jpeg_data, luminance):
        """Map a luminance level (0 - 1) to quantified dc dct coefficient.
        The unquantized coefficient range depends on the JPEG sample precision.
        This function maps the input luminance level to that range, then applies
        the quantization factor selected by the first image component.

        Parameters
        ----------
        jpeg_data: bytes
            Jpeg data containing quantification table(s).
        luminance: float
            Luminance level (0 - black, 1 - white).

        Returns
        ----------
        int
            Quantified luminance dc dct coefficent.
        """
        luminance = min(max(luminance, 0), 1)
        precision, table_index = \
            cls.__get_frame_precision_and_luminance_table(jpeg_data)
        if precision < 2 or precision > 12:
            raise ValueError(
                'Background fill requires a lossy JPEG image with no more '
                'than 12 bits of precision')
        dc_dqt_coefficient = cls.__get_dc_dqt_element(
            jpeg_data, table_index)
        if dc_dqt_coefficient <= 0:
            raise ValueError('Luminance quantization coefficient must be positive')
        maximum_sample = (1 << precision) - 1
        sample_center = 1 << (precision - 1)
        unquantized_dc = 8 * (
            luminance * maximum_sample - sample_center)
        return int(round(unquantized_dc / dc_dqt_coefficient))

    def __report_error(self, handle):
        """reports error while error occurred"""
        # tj3GetErrorCode always returns the error code
        if self.__get_error_code(handle) == TJERR_WARNING:
            warnings.warn(self.__get_error_string(handle))
            return
        # fatal error occurred
        raise IOError(self.__get_error_string(handle))

    def __get_error_string(self, handle):
        """returns error string"""
        # tj3GetErrorStr always takes handle parameter
        return self.__get_error_str(handle).decode()

    def __find_turbojpeg(self):
        """returns default turbojpeg library path if possible"""
        lib_path = find_library('turbojpeg')
        if lib_path is not None:
            return lib_path
        for lib_path in DEFAULT_LIB_PATHS[platform.system()]:
            if os.path.exists(lib_path):
                return lib_path
        if platform.system() == 'Linux' and 'LD_LIBRARY_PATH' in os.environ:
            ld_library_path = os.environ['LD_LIBRARY_PATH']
            for path in ld_library_path.split(':'):
                lib_path = os.path.join(path, 'libturbojpeg.so.0')
                if os.path.exists(lib_path):
                    return lib_path
        raise RuntimeError(
            'Unable to locate turbojpeg library automatically. '
            'You may specify the turbojpeg library path manually.\n'
            'e.g. jpeg = TurboJPEG(lib_path)')

    def __getaddr(self, nda):
        """returns the memory address for a given ndarray"""
        return cast(nda.__array_interface__['data'][0], POINTER(c_ubyte))
    
    def __getaddr_uint16(self, nda):
        """returns the memory address for a given uint16 ndarray"""
        return cast(nda.__array_interface__['data'][0], POINTER(c_ushort))

    @staticmethod
    def __validate_resource_limit(argument_name, value):
        """Validate a non-negative value passed through tj3Set(int)."""
        if type(value) is int and 0 <= value <= _MAX_TJPARAM_VALUE:
            return value
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise TypeError(
                "'{}' argument must be a non-negative integer".format(
                    argument_name))
        value = int(value)
        if value < 0 or value > _MAX_TJPARAM_VALUE:
            raise ValueError(
                "'{}' argument must be between 0 and {}".format(
                    argument_name, _MAX_TJPARAM_VALUE))
        return value

    @staticmethod
    def __validate_positive_integer(argument_name, value):
        """Validate a positive integer that is passed to a native int."""
        if type(value) is int and 1 <= value <= _MAX_TJPARAM_VALUE:
            return value
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise TypeError(
                "'{}' argument must be a positive integer".format(
                    argument_name))
        value = int(value)
        if value < 1 or value > _MAX_TJPARAM_VALUE:
            raise ValueError(
                "'{}' argument must be a positive integer no greater than {}"
                .format(argument_name, _MAX_TJPARAM_VALUE))
        return value

    @staticmethod
    def __validate_bounded_integer(argument_name, value, minimum, maximum):
        """Validate an integer enum or numeric option within fixed bounds."""
        if type(value) is int and minimum <= value <= maximum:
            return value
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)):
            raise TypeError(
                "'{}' argument must be an integer".format(argument_name))
        value = int(value)
        if value < minimum or value > maximum:
            raise ValueError(
                "'{}' argument must be between {} and {}".format(
                    argument_name, minimum, maximum))
        return value

    @classmethod
    def __validate_pixel_format(cls, pixel_format):
        if type(pixel_format) is int and 0 <= pixel_format < len(tjPixelSize):
            return pixel_format
        return cls.__validate_bounded_integer(
            'pixel_format', pixel_format, 0, len(tjPixelSize) - 1)

    @classmethod
    def __validate_subsampling(cls, jpeg_subsample):
        if type(jpeg_subsample) is int and \
                TJSAMP_444 <= jpeg_subsample <= TJSAMP_441:
            return jpeg_subsample
        return cls.__validate_bounded_integer(
            'jpeg_subsample', jpeg_subsample, TJSAMP_444, TJSAMP_441)

    @classmethod
    def __validate_quality(cls, quality):
        if type(quality) is int and 1 <= quality <= 100:
            return quality
        return cls.__validate_bounded_integer('quality', quality, 1, 100)

    @classmethod
    def __prepare_image(
            cls, img_array, dtype, pixel_format, operation,
            maximum_value=None):
        """Validate and pack an image before exposing its memory to C."""
        if not (type(pixel_format) is int and
                0 <= pixel_format < len(tjPixelSize)):
            pixel_format = cls.__validate_pixel_format(pixel_format)
        image = img_array if type(img_array) is np.ndarray \
            else np.asarray(img_array)
        if image.dtype != dtype:
            if dtype == np.uint8:
                detail = 'uint8 array with values in range 0-255'
            elif maximum_value == 4095:
                detail = 'uint16 array with values in range 0-4095'
            else:
                detail = 'uint16 array with values in range 0-65535'
            raise ValueError('{} requires {}'.format(operation, detail))

        channels = tjPixelSize[pixel_format]
        if channels == 1:
            valid_shape = (
                image.ndim == 2 or
                (image.ndim == 3 and image.shape[2] == 1)
            )
        else:
            valid_shape = image.ndim == 3 and image.shape[2] == channels
        if not valid_shape:
            raise ValueError(
                'Invalid shape for {} with pixel_format={}: expected '
                '{} channel(s)'.format(operation, pixel_format, channels))

        height, width = image.shape[:2]
        if not 1 <= height <= _MAX_TJPARAM_VALUE:
            height = cls.__validate_positive_integer('height', height)
        if not 1 <= width <= _MAX_TJPARAM_VALUE:
            width = cls.__validate_positive_integer('width', width)
        if maximum_value is not None and image.size \
                and int(image.max()) > maximum_value:
            raise ValueError(
                '{} values must be between 0 and {}'.format(
                    operation, maximum_value))
        return np.ascontiguousarray(image), height, width, pixel_format

    @classmethod
    def __validate_alignment(cls, value, argument_name='align'):
        """Validate a TurboJPEG YUV row alignment."""
        if type(value) is int:
            if 1 <= value <= _MAX_TJPARAM_VALUE \
                    and not value & (value - 1):
                return value
            raise ValueError(
                "'{}' argument must be a positive power of two".format(
                    argument_name))
        try:
            value = cls.__validate_positive_integer(argument_name, value)
        except ValueError as exc:
            raise ValueError(
                "'{}' argument must be a positive power of two".format(
                    argument_name)) from exc
        if value & (value - 1):
            raise ValueError(
                "'{}' argument must be a positive power of two".format(
                    argument_name))
        return value

    @staticmethod
    def __get_buffer_view(value, argument_name, writable):
        """Return a contiguous memoryview suitable for a native call."""
        try:
            view = memoryview(value)
        except TypeError as exc:
            raise TypeError(
                '{} must support the buffer protocol'.format(
                    argument_name)) from exc
        if not view.c_contiguous:
            raise ValueError(
                '{} must be C-contiguous'.format(argument_name))
        if writable and view.readonly:
            raise TypeError(
                '{} must be writable'.format(argument_name))
        return view

    @property
    def scaling_factors(self):
        return self.__scaling_factors

if __name__ == '__main__':
    jpeg = TurboJPEG()
    in_file = open('input.jpg', 'rb')
    img_array = jpeg.decode(in_file.read())
    in_file.close()
    out_file = open('output.jpg', 'wb')
    out_file.write(jpeg.encode(img_array))
    out_file.close()
    import cv2
    cv2.imshow('image', img_array)
    cv2.waitKey(0)

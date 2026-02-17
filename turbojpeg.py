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
__version__ = '2.1.0'

from ctypes import *
from ctypes.util import find_library
import platform
import numpy as np
import math
import warnings
import os
from struct import unpack, calcsize

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
tjMCUWidth = [8, 16, 16, 8, 8, 32]

# MCU block height (in pixels) for a given level of chrominance subsampling.
# MCU block sizes:
#  - 8x8 for no subsampling or grayscale
#  - 16x8 for 4:2:2
#  - 8x16 for 4:4:0
#  - 16x16 for 4:2:0
#  - 32x8 for 4:1:1
tjMCUHeight = [8, 8, 16, 8, 16, 8]

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
TJPARAM_MAXPIXELS = 23
TJPARAM_MAXMEMORY = 24

class CroppingRegion(Structure):
    _fields_ = [("x", c_int), ("y", c_int), ("w", c_int), ("h", c_int)]

class ScalingFactor(Structure):
    _fields_ = ('num', c_int), ('denom', c_int)

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

def fill_background(coeffs_ptr, arrayRegion, planeRegion, componentID, transformID, transform_ptr):
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
        for x in range(background_data.w//MCU_WIDTH, planeRegion.w//MCU_WIDTH):
            for y in range(
                left_start_row//MCU_HEIGHT,
                left_end_row//MCU_HEIGHT
            ):
                coeffs[y][x][0] = background_data.lum

        # fill mcus under image
        bottom_start_row = (
            max(arrayRegion.y, background_data.h) - arrayRegion.y
        )
        bottom_end_row = (
            max(arrayRegion.y+arrayRegion.h, background_data.h)
            - arrayRegion.y
        )
        for x in range(0, planeRegion.w//MCU_WIDTH):
            for y in range(
                bottom_start_row//MCU_HEIGHT,
                bottom_end_row//MCU_HEIGHT
            ):
                coeffs[y][x][0] = background_data.lum

    return 1


def split_byte_into_nibbles(value):
    """Split byte int into 2 nibbles (4 bits)."""
    first = value >> 4
    second = value & 0x0F
    return first, second


class TurboJPEG(object):
    """A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image."""
    def __init__(self, lib_path=None):
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

        # tjGetScalingFactors - still the current API in 3.1.x
        get_scaling_factors = turbo_jpeg.tjGetScalingFactors
        get_scaling_factors.argtypes = [POINTER(c_int)]
        get_scaling_factors.restype = POINTER(ScalingFactor)
        num_scaling_factors = c_int()
        scaling_factors = get_scaling_factors(byref(num_scaling_factors))
        self.__scaling_factors = frozenset(
            (scaling_factors[i].num, scaling_factors[i].denom)
            for i in range(num_scaling_factors.value)
        )

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
            
            if return_precision:
                precision = self.__get(handle, TJPARAM_PRECISION)
                if precision < 0:
                    self.__report_error(handle)
                return (width, height, jpeg_subsample, jpeg_colorspace, precision)
            else:
                return (width, height, jpeg_subsample, jpeg_colorspace)
        finally:
            self.__destroy(handle)

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
            Destination array (optional)
            
        Returns
        -------
        ndarray
            Decoded image as numpy array (uint8)
        """
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            # Set decompression parameters using tj3Set
            if flags & TJFLAG_BOTTOMUP:
                self.__set(handle, TJPARAM_BOTTOMUP, 1)
            if flags & TJFLAG_FASTUPSAMPLE:
                self.__set(handle, TJPARAM_FASTUPSAMPLE, 1)
            if flags & TJFLAG_FASTDCT:
                self.__set(handle, TJPARAM_FASTDCT, 1)
            if flags & TJFLAG_STOPONWARNING:
                self.__set(handle, TJPARAM_STOPONWARNING, 1)
            
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, _, _ = \
                self.__get_header_and_dimensions(handle, jpeg_array.size, src_addr, scaling_factor)
            
            dtype = np.uint8
            if ((type(dst) == np.ndarray) and
                (dst.shape == (scaled_height, scaled_width, tjPixelSize[pixel_format])) and
                (dst.dtype == dtype)):
                img_array = dst
            else:
                img_array = np.empty(
                    [scaled_height, scaled_width, tjPixelSize[pixel_format]],
                    dtype=dtype)
            dest_addr = self.__getaddr(img_array)
            # pitch should be width * bytes_per_pixel (samples per row)
            pitch = scaled_width * tjPixelSize[pixel_format]
            status = self.__decompress(
                handle, src_addr, jpeg_array.size, dest_addr, pitch, pixel_format)
            
            if status != 0:
                self.__report_error(handle)
            return img_array
        finally:
            self.__destroy(handle)

    def decode_to_yuv(self, jpeg_buf, scaling_factor=None, pad=4, flags=0):
        """decodes JPEG memory buffer to yuv array."""
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, jpeg_subsample, _ = \
                self.__get_header_and_dimensions(handle, jpeg_array.size, src_addr, scaling_factor)
            buffer_size = self.__buffer_size_YUV(scaled_width, pad, scaled_height, jpeg_subsample)
            buffer_array = np.empty(buffer_size, dtype=np.uint8)
            dest_addr = self.__getaddr(buffer_array)
            status = self.__decompressToYUV(
                handle, src_addr, jpeg_array.size, dest_addr, pad)
            if status != 0:
                self.__report_error(handle)
            plane_sizes = list()
            plane_sizes.append((scaled_height, scaled_width))
            if jpeg_subsample != TJSAMP_GRAY:
                for i in range(1, 3):
                    plane_sizes.append((
                        self.__plane_height(i, scaled_height, jpeg_subsample),
                        self.__plane_width(i, scaled_width, jpeg_subsample)))
            return buffer_array, plane_sizes
        finally:
            self.__destroy(handle)

    def decode_to_yuv_planes(self, jpeg_buf, scaling_factor=None, strides=(0, 0, 0), flags=0):
        """decodes JPEG memory buffer to yuv planes."""
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, jpeg_subsample, _ = \
                self.__get_header_and_dimensions(handle, jpeg_array.size, src_addr, scaling_factor)
            num_planes = 3
            if jpeg_subsample == TJSAMP_GRAY:
                num_planes = 1
            strides_addr = (c_int * num_planes)()
            dest_addr = (POINTER(c_ubyte) * num_planes)()
            planes = list()
            for i in range(num_planes):
                if strides[i] == 0:
                    strides_addr[i] = self.__plane_width(i, scaled_width, jpeg_subsample)
                else:
                    strides_addr[i] = strides[i]
                planes.append(np.empty(
                    (self.__plane_height(i, scaled_height, jpeg_subsample), strides_addr[i]), dtype=np.uint8))
                dest_addr[i] = self.__getaddr(planes[i])
            status = self.__decompressToYUVPlanes(
                handle, src_addr, jpeg_array.size, dest_addr, strides_addr)
            if status != 0:
                self.__report_error(handle)
            return planes
        finally:
            self.__destroy(handle)

    def encode(self, img_array, quality=85, pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_422, flags=0, dst=None):
        """encodes numpy array to JPEG memory buffer.
        
        Parameters
        ----------
        img_array : ndarray
            Image data to encode (uint8)
        quality : int
            JPEG quality (1-100)
        pixel_format : int
            Pixel format (TJPF_RGB, TJPF_BGR, etc.)
        jpeg_subsample : int
            Chroma subsampling (TJSAMP_444, TJSAMP_422, etc.)
        flags : int
            Compression flags
        dst : buffer or None
            Destination buffer (optional)
            
        Returns
        -------
        bytes
            JPEG image data
        """
        handle = self.__init(TJINIT_COMPRESS)
        try:
            # Set compression parameters using tj3Set
            if self.__set(handle, TJPARAM_SUBSAMP, jpeg_subsample) != 0:
                self.__report_error(handle)
            if self.__set(handle, TJPARAM_QUALITY, quality) != 0:
                self.__report_error(handle)
            if flags & TJFLAG_PROGRESSIVE:
                if self.__set(handle, TJPARAM_PROGRESSIVE, 1) != 0:
                    self.__report_error(handle)
            if flags & TJFLAG_FASTDCT:
                if self.__set(handle, TJPARAM_FASTDCT, 1) != 0:
                    self.__report_error(handle)
            
            img_array = np.ascontiguousarray(img_array)
            
            # Validate dtype is uint8
            if img_array.dtype != np.uint8:
                raise ValueError('encode() requires uint8 array (values 0-255); use encode_12bit() for 12-bit images (uint16, 0-4095) or encode_16bit() for 16-bit images (uint16, 0-65535)')
            
            if dst is not None and not self.__is_buffer(dst):
                raise TypeError('\'dst\' argument must support buffer protocol')
            if (dst is not None and
                (len(dst) >= self.buffer_size(img_array, jpeg_subsample))):
                dst_array = np.frombuffer(dst, dtype=np.uint8)
                jpeg_buf = dst_array.ctypes.data_as(c_void_p)
                jpeg_size = c_size_t(len(dst))
            else:
                dst_array = None
                jpeg_buf = c_void_p()
                jpeg_size = c_size_t()
            height, width = img_array.shape[:2]
            channel = tjPixelSize[pixel_format]
            if channel > 1 and (len(img_array.shape) < 3 or img_array.shape[2] != channel):
                raise ValueError('Invalid shape for image data')
            
            src_addr = self.__getaddr(img_array)
            status = self.__compress(
                handle, src_addr, width, img_array.strides[0], height, pixel_format,
                byref(jpeg_buf), byref(jpeg_size))
            
            if status != 0:
                self.__report_error(handle)
            if dst_array is None or jpeg_buf.value != dst_array.ctypes.data:
                result = self.__copy_from_buffer(jpeg_buf.value, jpeg_size.value)
                self.__free(jpeg_buf)
            else:
                result = dst
            return result if dst is None else (result, jpeg_size.value)
        finally:
            self.__destroy(handle)

    def encode_from_yuv(self, img_array, height, width, quality=85, jpeg_subsample=TJSAMP_420, flags=0):
        """encodes numpy array to JPEG memory buffer."""
        handle = self.__init(TJINIT_COMPRESS)
        try:
            # Set compression parameters using tj3Set
            if self.__set(handle, TJPARAM_SUBSAMP, jpeg_subsample) != 0:
                self.__report_error(handle)
            if self.__set(handle, TJPARAM_QUALITY, quality) != 0:
                self.__report_error(handle)
            
            jpeg_buf = c_void_p()
            jpeg_size = c_size_t()
            img_array = np.ascontiguousarray(img_array)
            src_addr = self.__getaddr(img_array)
            status = self.__compressFromYUV(
                handle, src_addr, width, 4, height,
                byref(jpeg_buf), byref(jpeg_size))
            if status != 0:
                self.__report_error(handle)
            dest_buf = create_string_buffer(jpeg_size.value)
            memmove(dest_buf, jpeg_buf.value, jpeg_size.value)
            self.__free(jpeg_buf)
            return dest_buf.raw
        finally:
            self.__destroy(handle)

    def scale_with_quality(self, jpeg_buf, scaling_factor=None, quality=85, flags=0):
        """decompresstoYUV with scale factor, recompresstoYUV with quality factor"""
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            scaled_width, scaled_height, jpeg_subsample, _ = self.__get_header_and_dimensions(
                handle, jpeg_array.size, src_addr, scaling_factor)
            buffer_YUV_size = self.__buffer_size_YUV(
                scaled_width, 4, scaled_height, jpeg_subsample)
            img_array = np.empty([buffer_YUV_size])
            dest_addr = self.__getaddr(img_array)
            status = self.__decompressToYUV(
                handle, src_addr, jpeg_array.size, dest_addr, 4)
            if status != 0:
                self.__report_error(handle)
            self.__destroy(handle)
            handle = self.__init(TJINIT_COMPRESS)
            # Set compression parameters
            if self.__set(handle, TJPARAM_SUBSAMP, jpeg_subsample) != 0:
                self.__report_error(handle)
            if self.__set(handle, TJPARAM_QUALITY, quality) != 0:
                self.__report_error(handle)
            jpeg_buf = c_void_p()
            jpeg_size = c_size_t()
            status = self.__compressFromYUV(
                handle, dest_addr, scaled_width, 4, scaled_height, byref(jpeg_buf),
                byref(jpeg_size))
            if status != 0:
                self.__report_error(handle)
            dest_buf = create_string_buffer(jpeg_size.value)
            memmove(dest_buf, jpeg_buf.value, jpeg_size.value)
            self.__free(jpeg_buf)
            return dest_buf.raw
        finally:
            self.__destroy(handle)
    
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
        handle = self.__init(TJINIT_COMPRESS)
        try:
            # Set compression parameters using tj3Set
            # Enable lossless mode if requested
            if lossless:
                if self.__set(handle, TJPARAM_LOSSLESS, 1) != 0:
                    self.__report_error(handle)
                # In lossless mode, subsampling is automatically set to 4:4:4
                # and quality parameter is ignored
            else:
                # Set standard lossy parameters
                if self.__set(handle, TJPARAM_SUBSAMP, jpeg_subsample) != 0:
                    self.__report_error(handle)
                if self.__set(handle, TJPARAM_QUALITY, quality) != 0:
                    self.__report_error(handle)
            if flags & TJFLAG_PROGRESSIVE:
                if self.__set(handle, TJPARAM_PROGRESSIVE, 1) != 0:
                    self.__report_error(handle)
            if flags & TJFLAG_FASTDCT:
                if self.__set(handle, TJPARAM_FASTDCT, 1) != 0:
                    self.__report_error(handle)
            
            img_array = np.ascontiguousarray(img_array)
            
            # Validate dtype is uint16 for 12-bit precision
            if img_array.dtype != np.uint16:
                raise ValueError('encode_12bit() requires uint16 array with values in range 0-4095')
            
            jpeg_buf = c_void_p()
            jpeg_size = c_size_t()
            height, width = img_array.shape[:2]
            channel = tjPixelSize[pixel_format]
            if channel > 1 and (len(img_array.shape) < 3 or img_array.shape[2] != channel):
                raise ValueError('Invalid shape for image data')
            
            # 12-bit precision
            src_addr = self.__getaddr_uint16(img_array)
            # For 12-bit, stride is in samples (uint16), not bytes
            # Note: uint16 is 2 bytes on all supported platforms
            stride_samples = img_array.strides[0] // 2  # Convert bytes to uint16 count (2 bytes per uint16)
            
            status = self.__compress12(
                handle, src_addr, width, stride_samples, height, pixel_format,
                byref(jpeg_buf), byref(jpeg_size))
            
            if status != 0:
                self.__report_error(handle)
            result = self.__copy_from_buffer(jpeg_buf.value, jpeg_size.value)
            self.__free(jpeg_buf)
            return result
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
        handle = self.__init(TJINIT_COMPRESS)
        try:
            # Set compression parameters using tj3Set
            # 16-bit requires lossless mode
            if self.__set(handle, TJPARAM_LOSSLESS, 1) != 0:
                self.__report_error(handle)
            # In lossless mode, subsampling is automatically set to 4:4:4
            if flags & TJFLAG_PROGRESSIVE:
                if self.__set(handle, TJPARAM_PROGRESSIVE, 1) != 0:
                    self.__report_error(handle)
            if flags & TJFLAG_FASTDCT:
                if self.__set(handle, TJPARAM_FASTDCT, 1) != 0:
                    self.__report_error(handle)
            
            img_array = np.ascontiguousarray(img_array)
            
            # Validate dtype is uint16 for 16-bit precision
            if img_array.dtype != np.uint16:
                raise ValueError('encode_16bit() requires uint16 array with values in range 0-65535')
            
            jpeg_buf = c_void_p()
            jpeg_size = c_size_t()
            height, width = img_array.shape[:2]
            channel = tjPixelSize[pixel_format]
            if channel > 1 and (len(img_array.shape) < 3 or img_array.shape[2] != channel):
                raise ValueError('Invalid shape for image data')
            
            # 16-bit precision
            src_addr = self.__getaddr_uint16(img_array)
            # For 16-bit, stride is in samples (uint16), not bytes
            # Note: uint16 is 2 bytes on all supported platforms
            stride_samples = img_array.strides[0] // 2  # Convert bytes to uint16 count (2 bytes per uint16)
            
            status = self.__compress16(
                handle, src_addr, width, stride_samples, height, pixel_format,
                byref(jpeg_buf), byref(jpeg_size))
            
            if status != 0:
                self.__report_error(handle)
            result = self.__copy_from_buffer(jpeg_buf.value, jpeg_size.value)
            self.__free(jpeg_buf)
            return result
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
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            # Set decompression parameters using tj3Set
            if flags & TJFLAG_BOTTOMUP:
                self.__set(handle, TJPARAM_BOTTOMUP, 1)
            if flags & TJFLAG_FASTUPSAMPLE:
                self.__set(handle, TJPARAM_FASTUPSAMPLE, 1)
            if flags & TJFLAG_FASTDCT:
                self.__set(handle, TJPARAM_FASTDCT, 1)
            if flags & TJFLAG_STOPONWARNING:
                self.__set(handle, TJPARAM_STOPONWARNING, 1)
            
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
        handle = self.__init(TJINIT_DECOMPRESS)
        try:
            # Set decompression parameters using tj3Set
            if flags & TJFLAG_BOTTOMUP:
                self.__set(handle, TJPARAM_BOTTOMUP, 1)
            if flags & TJFLAG_FASTUPSAMPLE:
                self.__set(handle, TJPARAM_FASTUPSAMPLE, 1)
            if flags & TJFLAG_FASTDCT:
                self.__set(handle, TJPARAM_FASTDCT, 1)
            if flags & TJFLAG_STOPONWARNING:
                self.__set(handle, TJPARAM_STOPONWARNING, 1)
            
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
        """losslessly crop a jpeg image with optional grayscale"""
        handle = self.__init(TJINIT_TRANSFORM)
        try:
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = self.__getaddr(jpeg_array)
            # Get header information using tj3DecompressHeader
            status = self.__decompress_header(handle, src_addr, jpeg_array.size)
            if status != 0:
                self.__report_error(handle)
            width = self.__get(handle, TJPARAM_JPEGWIDTH)
            height = self.__get(handle, TJPARAM_JPEGHEIGHT)
            jpeg_subsample = self.__get(handle, TJPARAM_SUBSAMP)
            
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

            # Define cropping regions from input parameters and image size
            crop_regions = self.__define_cropping_regions(crop_parameters)
            number_of_operations = len(crop_regions)

            # Define crop transforms from cropping_regions
            crop_transforms = (TransformStruct * number_of_operations)()
            for i, crop_region in enumerate(crop_regions):
                # The fill_background callback is slow, only use it if needed
                if self.__need_fill_background(
                    crop_region,
                    (image_width, image_height),
                    background_luminance
                ):
                    # Use callback to fill in background post-transform
                    callback_data = BackgroundStruct(
                        image_width,
                        image_height,
                        self.__map_luminance_to_dc_dct_coefficient(
                            bytearray(jpeg_buf),
                            background_luminance
                        )
                    )
                    callback = CUSTOMFILTER(fill_background)
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

    def buffer_size(self, img_array, jpeg_subsample=TJSAMP_422):
        """Get maximum number of bytes of compressed jpeg data"""
        img_array = np.ascontiguousarray(img_array)
        height, width = img_array.shape[:2]
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
        dest_buf = create_string_buffer(size)
        memmove(dest_buf, buffer, size)
        return dest_buf.raw

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
        
        # Set scaling factor if provided - must be done AFTER reading header
        scaled_width = width
        scaled_height = height
        if scaling_factor is not None:
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

    def __axis_to_image_boundaries(self, a, b, img_boundary, preserve, mcuBlock):
        if preserve:
            original_a = a
            a = int(math.ceil(float(original_a) / mcuBlock) * mcuBlock)
            b -= (a - original_a)
            if (a + b) > img_boundary:
                b = img_boundary - a
        else:
            img_b = img_boundary - (img_boundary % mcuBlock)
            delta_a = a % mcuBlock
            if a > img_b:
                a = img_b
            else:
                a = a - delta_a
            b = b + delta_a
            if (a + b) > img_b:
                b = img_b - a
        return a, b

    @staticmethod
    def __define_cropping_regions(crop_parameters):
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
        return [
            CroppingRegion(x=crop[0], y=crop[1], w=crop[2], h=crop[3])
            for crop in crop_parameters
        ]

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
    def __find_dqt(jpeg_data, dqt_index):
        """Return byte offset to quantification table with index dqt_index in
        jpeg_data.

        Parameters
        ----------
        jpeg_data: bytes
            Jpeg data.
        dqt_index: int
            Index of quantificatin table to find (0 - luminance).

        Returns
        ----------
        Optional[int]
            Byte offset to quantification table, or None if not found.
        """
        offset = 0
        while offset < len(jpeg_data):
            dct_table_offset = jpeg_data[offset:].find(b'\xFF\xDB')
            if dct_table_offset == -1:
                break
            dct_table_offset += offset
            dct_table_length = unpack(
                '>H',
                jpeg_data[dct_table_offset+2:dct_table_offset+4]
            )[0]
            dct_table_id_offset = dct_table_offset + 4
            table_index, _ = split_byte_into_nibbles(
                jpeg_data[dct_table_id_offset]
            )
            if table_index == dqt_index:
                return dct_table_offset
            offset += dct_table_offset+dct_table_length
        return None

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
        dqt_offset = cls.__find_dqt(jpeg_data, dqt_index)
        if dqt_offset is None:
            raise ValueError(
                "Quantisation table {dqt_index} not found in header".format(
                    dqt_index=dqt_index)
            )
        precision_offset = dqt_offset+4
        precision = split_byte_into_nibbles(jpeg_data[precision_offset])[0]
        if precision == 0:
            unpack_type = '>b'
        elif precision == 1:
            unpack_type = '>h'
        else:
            raise ValueError('Not valid precision definition in dqt')
        dc_offset = dqt_offset + 5
        dc_length = calcsize(unpack_type)
        dc_value = unpack(
            unpack_type,
            jpeg_data[dc_offset:dc_offset+dc_length]
        )[0]
        return dc_value

    @classmethod
    def __map_luminance_to_dc_dct_coefficient(cls, jpeg_data, luminance):
        """Map a luminance level (0 - 1) to quantified dc dct coefficient.
        Before quantification dct coefficient have a range -1024 - 1023. This
        is reduced upon quantification by the quantification factor. This
        function maps the input luminance level range to the quantified dc dct
        coefficient range.

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
        dc_dqt_coefficient = cls.__get_dc_dqt_element(jpeg_data, 0)
        return int(round((luminance * 2047 - 1024) / dc_dqt_coefficient))

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

    def __is_buffer(self, x):
        result = True
        try:
            memoryview(x)
        except Exception:
            result = False
        return result

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

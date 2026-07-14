"""
Comprehensive unit tests for PyTurboJPEG library.

This module contains unit tests for all core functions of the PyTurboJPEG library,
covering various input scenarios, edge cases, and error conditions.

This also includes regression tests for:
1. Robustness of Buffer Handling (empty bytes, truncated JPEG headers)
2. Library Loading (missing library, clear error messages)
3. Colorspace Consistency (all TJPF/TJSAMP combinations)
4. Memory Management (stress testing with 1000+ cycles)
5. Crop Functionality (with real input image)

Note: These tests require TurboJPEG 3.0+ as PyTurboJPEG 2.0+ uses the new
function-based TurboJPEG 3 API. Some tests account for differences in error
messages and DCT implementation compared to TurboJPEG 2.x.
"""
import pytest
import numpy as np
import os
import struct
from unittest.mock import patch
from ctypes import POINTER, c_size_t, c_void_p, cast
from ctypes.util import find_library

from turbojpeg import (
    CroppingRegion, TurboJPEG, YUVPlaneInfo, fill_background,
    tjMCUHeight, tjMCUWidth,
    TJPF_RGB, TJPF_BGR, TJPF_GRAY, TJPF_RGBA, TJPF_BGRA, TJPF_RGBX, TJPF_BGRX,
    TJPF_XBGR, TJPF_XRGB, TJPF_ABGR, TJPF_ARGB,
    TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY, TJSAMP_440, TJSAMP_411,
    TJSAMP_441,
    TJCS_RGB, TJCS_YCbCr, TJCS_GRAY,
    TJFLAG_ACCURATEDCT, TJFLAG_BOTTOMUP, TJFLAG_PROGRESSIVE,
    TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT, TJFLAG_STOPONWARNING,
    TJFLAG_LIMITSCANS,
    TJINIT_COMPRESS, TJINIT_DECOMPRESS,
    TJPARAM_BOTTOMUP, TJPARAM_FASTDCT,
    TJPARAM_SCANLIMIT, TJPARAM_MAXMEMORY, TJPARAM_MAXPIXELS,
    TJPARAM_STOPONWARNING,
    DEFAULT_MAX_PIXELS, DEFAULT_MAX_MEMORY, DEFAULT_SCAN_LIMIT,
)



# Test fixtures
@pytest.fixture(scope='module')
def jpeg_instance():
    """Create a TurboJPEG instance for testing."""
    lib_path = os.environ.get('TURBOJPEG_LIB_PATH', None)
    return TurboJPEG(lib_path=lib_path)


@pytest.fixture(scope='module')
def sample_bgr_image():
    """Create a sample BGR image for testing."""
    # Create a 100x100 BGR image with gradient colors
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = [i * 2, j * 2, (i + j) % 256]
    return img


@pytest.fixture(scope='module')
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    # Create a 100x100 RGB image with gradient colors
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = [(i + j) % 256, j * 2, i * 2]
    return img


@pytest.fixture(scope='module')
def sample_gray_image():
    """Create a sample grayscale image for testing."""
    # Create a 100x100 grayscale image
    img = np.zeros((100, 100, 1), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = [(i + j) % 256]
    return img


@pytest.fixture(scope='module')
def sample_image():
    """Create a sample BGR image for testing (alias for regression tests)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = [i * 2, j * 2, (i + j) % 256]
    return img


@pytest.fixture(scope='module')
def valid_jpeg(jpeg_instance, sample_bgr_image):
    """Create a valid encoded JPEG for testing."""
    return jpeg_instance.encode(sample_bgr_image)


@pytest.fixture(scope='module')
def encoded_sample_jpeg(jpeg_instance, sample_bgr_image):
    """Create an encoded JPEG from sample BGR image."""
    return jpeg_instance.encode(sample_bgr_image)


@pytest.fixture(scope='module')
def sample_12bit_image():
    """Create a sample 12-bit image (uint16 with values 0-4095)."""
    img = np.zeros((100, 100, 3), dtype=np.uint16)
    for i in range(100):
        for j in range(100):
            img[i, j] = [i * 40, j * 40, (i + j) * 20]
    # Ensure values are in 12-bit range (0-4095)
    img = np.clip(img, 0, 4095)
    return img


@pytest.fixture(scope='module')
def sample_16bit_image():
    """Create a sample 16-bit image (uint16 with full range 0-65535)."""
    img = np.zeros((100, 100, 3), dtype=np.uint16)
    for i in range(100):
        for j in range(100):
            img[i, j] = [i * 655, j * 655, (i + j) * 327]
    # Ensure values are in 16-bit range
    img = np.clip(img, 0, 65535)
    return img


class TestTurboJPEGInitialization:
    """Test TurboJPEG initialization."""
    
    def test_default_initialization(self):
        """Test TurboJPEG can be initialized with default parameters."""
        tj = TurboJPEG()
        assert tj is not None
    
    def test_scaling_factors_property(self, jpeg_instance):
        """Test that scaling_factors property returns expected values."""
        factors = jpeg_instance.scaling_factors
        assert isinstance(factors, frozenset)
        assert len(factors) > 0
        # Common scaling factors should be present
        assert (1, 1) in factors  # No scaling
        assert (1, 2) in factors  # Half size
        assert (1, 4) in factors  # Quarter size


class TestDecodeHeader:
    """Test decode_header function."""
    
    def test_decode_header_basic(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding JPEG header returns correct properties."""
        # Test backward compatible usage (4-tuple)
        width, height, subsample, colorspace = jpeg_instance.decode_header(encoded_sample_jpeg)
        assert width == 100
        assert height == 100
        assert subsample in [TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY]
        assert colorspace in [TJCS_RGB, TJCS_YCbCr, TJCS_GRAY]
        
        # Test new usage with precision (5-tuple)
        width, height, subsample, colorspace, precision = jpeg_instance.decode_header(encoded_sample_jpeg, return_precision=True)
        assert width == 100
        assert height == 100
        assert precision == 8  # Standard JPEG is 8-bit
    
    def test_decode_header_invalid_data(self, jpeg_instance):
        """Test decode_header with invalid JPEG data raises error."""
        with pytest.raises(OSError):
            jpeg_instance.decode_header(b'not a jpeg')
    
    def test_decode_header_empty_data(self, jpeg_instance):
        """Test decode_header with empty data raises error."""
        with pytest.raises(OSError):
            jpeg_instance.decode_header(b'')


class TestDecodeResourceLimits:
    """Regression tests for bounded decompression and transformation."""

    @staticmethod
    def _new_jpeg(**kwargs):
        return TurboJPEG(
            lib_path=os.environ.get('TURBOJPEG_LIB_PATH'),
            **kwargs,
        )

    @staticmethod
    def _replace_dimensions(jpeg_buf, width, height):
        """Replace dimensions in the first start-of-frame marker."""
        data = bytearray(jpeg_buf)
        sof_markers = (
            0xC0, 0xC1, 0xC2, 0xC3,
            0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB,
            0xCD, 0xCE, 0xCF,
        )
        for marker_code in sof_markers:
            marker_offset = data.find(bytes((0xFF, marker_code)))
            if marker_offset >= 0:
                data[marker_offset + 5:marker_offset + 7] = \
                    height.to_bytes(2, 'big')
                data[marker_offset + 7:marker_offset + 9] = \
                    width.to_bytes(2, 'big')
                return bytes(data)
        raise AssertionError('JPEG start-of-frame marker was not found')

    def test_resource_parameter_constants_match_turbojpeg_3(self):
        """MAXMEMORY precedes MAXPIXELS in the TurboJPEG 3 enum."""
        assert TJPARAM_MAXMEMORY == 23
        assert TJPARAM_MAXPIXELS == 24

    def test_default_resource_limits_skip_native_setters(
            self, encoded_sample_jpeg):
        """Disabled limits do not add native calls to the hot path."""
        jpeg = self._new_jpeg()
        native_set = jpeg._TurboJPEG__set
        calls = []

        def recording_set(handle, parameter, value):
            calls.append((parameter, value))
            return native_set(handle, parameter, value)

        jpeg._TurboJPEG__set = recording_set
        jpeg.decode_header(encoded_sample_jpeg)

        assert DEFAULT_MAX_PIXELS == 0
        assert DEFAULT_MAX_MEMORY == 0
        assert DEFAULT_SCAN_LIMIT == 0
        assert calls == []

    def test_default_limits_allow_libturbojpeg_maximum_dimensions(
            self, jpeg_instance):
        """The wrapper accepts a 65500x65500 header without allocating it."""
        source = jpeg_instance.encode(
            np.zeros((8, 8, 3), dtype=np.uint8),
            jpeg_subsample=TJSAMP_444,
        )
        maximum_header = self._replace_dimensions(
            source, width=65_500, height=65_500)
        jpeg = self._new_jpeg()

        assert jpeg.decode_header(maximum_header)[:2] == (65_500, 65_500)

    def test_configured_resource_limits_are_forwarded(
            self, encoded_sample_jpeg):
        """Constructor values are forwarded to the correct native parameters."""
        jpeg = self._new_jpeg(
            max_pixels=20_000,
            max_memory=64,
            scan_limit=7,
        )
        native_set = jpeg._TurboJPEG__set
        calls = []

        def recording_set(handle, parameter, value):
            calls.append((parameter, value))
            return native_set(handle, parameter, value)

        jpeg._TurboJPEG__set = recording_set
        jpeg.decode_header(encoded_sample_jpeg)

        if jpeg._TurboJPEG__supports_native_resource_limits:
            assert (TJPARAM_MAXPIXELS, 20_000) in calls
            assert (TJPARAM_MAXMEMORY, 64) in calls
        assert (TJPARAM_SCANLIMIT, 7) in calls

    @pytest.mark.parametrize(('kwargs', 'exception_type'), [
        ({'max_pixels': -1}, ValueError),
        ({'max_memory': -1}, ValueError),
        ({'scan_limit': -1}, ValueError),
        ({'max_pixels': 1.5}, TypeError),
        ({'max_memory': True}, TypeError),
        ({'scan_limit': '500'}, TypeError),
        ({'max_pixels': 2 ** 31}, ValueError),
    ])
    def test_constructor_rejects_invalid_resource_limits(
            self, kwargs, exception_type):
        """Resource limits must fit the signed integer native API."""
        with pytest.raises(exception_type):
            self._new_jpeg(**kwargs)

    def test_oversized_header_is_rejected_before_numpy_allocation(
            self, jpeg_instance):
        """A malicious header cannot request a giant output allocation."""
        source = jpeg_instance.encode(
            np.zeros((8, 8, 3), dtype=np.uint8),
            jpeg_subsample=TJSAMP_444,
        )
        oversized = self._replace_dimensions(
            source, width=50_000, height=50_000)
        limited = self._new_jpeg(max_pixels=1_000_000)

        with patch(
                'turbojpeg.np.empty',
                side_effect=AssertionError('output allocation was attempted')):
            with pytest.raises(ValueError, match='max_pixels'):
                limited.decode(oversized)

    @pytest.mark.parametrize('operation', [
        'decode_header',
        'decode',
        'decode_to_yuv',
        'decode_to_yuv_planes',
        'scale_with_quality',
        'decode_12bit',
        'decode_16bit',
        'crop',
        'crop_multiple',
        'optimize',
    ])
    def test_max_pixels_applies_to_decode_and_transform_paths(
            self, jpeg_instance, operation):
        """Every source-image path enforces the configured pixel bound."""
        source = jpeg_instance.encode(
            np.zeros((16, 16, 3), dtype=np.uint8))
        limited = self._new_jpeg(max_pixels=255)

        with pytest.raises(ValueError, match='max_pixels'):
            if operation == 'crop':
                limited.crop(source, 0, 0, 16, 16)
            elif operation == 'crop_multiple':
                limited.crop_multiple(source, [(0, 0, 16, 16)])
            else:
                getattr(limited, operation)(source)

    def test_scan_limit_rejects_progressive_scan_bomb(
            self, jpeg_instance):
        """Progressive decoding stops when the configured scan limit is hit."""
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        sequential = jpeg_instance.encode(image)
        progressive = jpeg_instance.encode(
            image, flags=TJFLAG_PROGRESSIVE)
        limited = self._new_jpeg(
            max_pixels=0,
            max_memory=0,
            scan_limit=1,
        )

        assert limited.decode(sequential).shape == image.shape
        with pytest.raises(OSError, match='more than 1 scans'):
            limited.decode(progressive)

    def test_limitscans_flag_maps_to_legacy_500_scan_limit(
            self, encoded_sample_jpeg):
        """The legacy flag enables its historical 500-scan bound."""
        jpeg = self._new_jpeg(
            max_pixels=0,
            max_memory=0,
            scan_limit=0,
        )
        native_set = jpeg._TurboJPEG__set
        calls = []

        def recording_set(handle, parameter, value):
            calls.append((parameter, value))
            return native_set(handle, parameter, value)

        jpeg._TurboJPEG__set = recording_set
        jpeg.decode(encoded_sample_jpeg, flags=TJFLAG_LIMITSCANS)

        assert (TJPARAM_SCANLIMIT, 500) in calls


class TestDecode:
    """Test decode function."""
    
    def test_decode_basic(self, jpeg_instance, encoded_sample_jpeg):
        """Test basic JPEG decoding to BGR array."""
        img_array = jpeg_instance.decode(encoded_sample_jpeg)
        assert img_array is not None
        assert img_array.shape == (100, 100, 3)
        assert img_array.dtype == np.uint8
    
    def test_decode_to_rgb(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding JPEG to RGB format."""
        img_array = jpeg_instance.decode(encoded_sample_jpeg, pixel_format=TJPF_RGB)
        assert img_array.shape == (100, 100, 3)
        assert img_array.dtype == np.uint8
    
    def test_decode_to_grayscale(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding JPEG to grayscale format."""
        img_array = jpeg_instance.decode(encoded_sample_jpeg, pixel_format=TJPF_GRAY)
        assert img_array.shape == (100, 100, 1)
        assert img_array.dtype == np.uint8
    
    def test_decode_to_rgba(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding JPEG to RGBA format."""
        img_array = jpeg_instance.decode(encoded_sample_jpeg, pixel_format=TJPF_RGBA)
        assert img_array.shape == (100, 100, 4)
        assert img_array.dtype == np.uint8
    
    def test_decode_to_bgra(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding JPEG to BGRA format."""
        img_array = jpeg_instance.decode(encoded_sample_jpeg, pixel_format=TJPF_BGRA)
        assert img_array.shape == (100, 100, 4)
        assert img_array.dtype == np.uint8
    
    def test_decode_with_scaling_half(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding JPEG with 1/2 scaling factor."""
        img_array = jpeg_instance.decode(encoded_sample_jpeg, scaling_factor=(1, 2))
        assert img_array.shape[0] == 50  # Half of 100
        assert img_array.shape[1] == 50
        assert img_array.shape[2] == 3
    
    def test_decode_with_scaling_quarter(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding JPEG with 1/4 scaling factor."""
        img_array = jpeg_instance.decode(encoded_sample_jpeg, scaling_factor=(1, 4))
        assert img_array.shape[0] == 25  # Quarter of 100
        assert img_array.shape[1] == 25
        assert img_array.shape[2] == 3
    
    def test_decode_with_fast_flags(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding with fast upsample and fast DCT flags."""
        img_array = jpeg_instance.decode(
            encoded_sample_jpeg, 
            flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT
        )
        assert img_array.shape == (100, 100, 3)
    
    def test_decode_with_invalid_scaling_factor(self, jpeg_instance, encoded_sample_jpeg):
        """Test decode with invalid scaling factor raises ValueError."""
        with pytest.raises(ValueError):
            jpeg_instance.decode(encoded_sample_jpeg, scaling_factor=(1, 3))
    
    def test_decode_in_place(self, jpeg_instance, encoded_sample_jpeg):
        """Test in-place decoding to pre-allocated array."""
        dst_array = np.empty((100, 100, 3), dtype=np.uint8)
        result = jpeg_instance.decode(encoded_sample_jpeg, dst=dst_array)
        assert result is dst_array
        assert id(result) == id(dst_array)

    def test_decode_in_place_accepts_ndarray_subclass(
            self, jpeg_instance, encoded_sample_jpeg):
        """Writable contiguous ndarray subclasses are valid destinations."""
        class DestinationArray(np.ndarray):
            pass

        dst_array = np.empty(
            (100, 100, 3), dtype=np.uint8).view(DestinationArray)
        result = jpeg_instance.decode(encoded_sample_jpeg, dst=dst_array)
        assert result is dst_array

    def test_decode_rejects_non_array_destination(
            self, jpeg_instance, encoded_sample_jpeg):
        """Decode destinations must be numpy arrays."""
        dst = bytearray(100 * 100 * 3)
        with pytest.raises(TypeError, match='numpy array'):
            jpeg_instance.decode(encoded_sample_jpeg, dst=dst)

    def test_decode_rejects_wrong_destination_shape(
            self, jpeg_instance, encoded_sample_jpeg):
        """Decode must not silently replace a destination of the wrong shape."""
        dst = np.empty((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match='must have shape'):
            jpeg_instance.decode(encoded_sample_jpeg, dst=dst)

    def test_decode_rejects_wrong_destination_dtype(
            self, jpeg_instance, encoded_sample_jpeg):
        """Decode destinations must use uint8 storage."""
        dst = np.empty((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match='dtype uint8'):
            jpeg_instance.decode(encoded_sample_jpeg, dst=dst)

    def test_decode_rejects_noncontiguous_destination_without_writing(
            self, jpeg_instance, encoded_sample_jpeg):
        """A strided destination must be rejected before the native call."""
        backing = np.full((100, 100, 6), 0xA5, dtype=np.uint8)
        dst = backing[:, :, ::2]
        before = backing.copy()
        assert dst.shape == (100, 100, 3)
        assert not dst.flags.c_contiguous

        with pytest.raises(ValueError, match='C-contiguous'):
            jpeg_instance.decode(encoded_sample_jpeg, dst=dst)

        assert np.array_equal(backing, before)

    def test_decode_rejects_readonly_destination_without_writing(
            self, jpeg_instance, encoded_sample_jpeg):
        """A read-only destination must be rejected before the native call."""
        dst = np.full((100, 100, 3), 0xA5, dtype=np.uint8)
        before = dst.copy()
        dst.flags.writeable = False

        with pytest.raises(ValueError, match='writable'):
            jpeg_instance.decode(encoded_sample_jpeg, dst=dst)

        assert np.array_equal(dst, before)
    
    def test_decode_invalid_data(self, jpeg_instance):
        """Test decode with invalid JPEG data raises error."""
        with pytest.raises(OSError):
            jpeg_instance.decode(b'not a jpeg')


class TestDecodeToYUV:
    """Test decode_to_yuv function."""
    
    def test_decode_to_yuv_basic(self, jpeg_instance, encoded_sample_jpeg):
        """Test basic decoding to YUV format."""
        buffer_array, plane_sizes = jpeg_instance.decode_to_yuv(encoded_sample_jpeg)
        assert buffer_array is not None
        assert isinstance(buffer_array, np.ndarray)
        assert isinstance(plane_sizes, list)
        assert len(plane_sizes) >= 1
        # First plane should be Y (luminance) with original dimensions
        assert plane_sizes[0] == (100, 100)
    
    def test_decode_to_yuv_with_scaling(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding to YUV with scaling factor."""
        buffer_array, plane_sizes = jpeg_instance.decode_to_yuv(
            encoded_sample_jpeg, 
            scaling_factor=(1, 2)
        )
        assert plane_sizes[0] == (50, 50)
    
    def test_decode_to_yuv_custom_pad(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding to YUV with custom padding."""
        buffer_array, plane_sizes = jpeg_instance.decode_to_yuv(
            encoded_sample_jpeg, 
            pad=8
        )
        assert buffer_array is not None


class TestDecodeToYUVPlanes:
    """Test decode_to_yuv_planes function."""
    
    def test_decode_to_yuv_planes_basic(self, jpeg_instance, encoded_sample_jpeg):
        """Test basic decoding to YUV planes."""
        planes = jpeg_instance.decode_to_yuv_planes(encoded_sample_jpeg)
        assert isinstance(planes, list)
        assert len(planes) in [1, 3]  # 1 for grayscale, 3 for color
        # All planes should be numpy arrays
        for plane in planes:
            assert isinstance(plane, np.ndarray)
            assert plane.dtype == np.uint8
    
    def test_decode_to_yuv_planes_with_scaling(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding to YUV planes with scaling factor."""
        planes = jpeg_instance.decode_to_yuv_planes(
            encoded_sample_jpeg, 
            scaling_factor=(1, 2)
        )
        assert len(planes) in [1, 3]
        # First plane (Y) should be scaled
        assert planes[0].shape[0] == 50
    
    def test_decode_to_yuv_planes_custom_strides(self, jpeg_instance, encoded_sample_jpeg):
        """Test decoding to YUV planes with custom strides."""
        planes = jpeg_instance.decode_to_yuv_planes(
            encoded_sample_jpeg, 
            strides=(128, 64, 64)
        )
        assert len(planes) in [1, 3]


class TestEncode:
    """Test encode function."""
    
    def test_encode_basic(self, jpeg_instance, sample_bgr_image):
        """Test basic encoding of BGR image."""
        jpeg_buf = jpeg_instance.encode(sample_bgr_image)
        assert jpeg_buf is not None
        assert isinstance(jpeg_buf, bytes)
        assert len(jpeg_buf) > 0
    
    def test_encode_quality_variations(self, jpeg_instance, sample_bgr_image):
        """Test encoding with different quality levels."""
        # Low quality
        jpeg_low = jpeg_instance.encode(sample_bgr_image, quality=50)
        # High quality
        jpeg_high = jpeg_instance.encode(sample_bgr_image, quality=95)
        # Higher quality should result in larger file
        assert len(jpeg_high) > len(jpeg_low)
    
    def test_encode_rgb_format(self, jpeg_instance, sample_rgb_image):
        """Test encoding RGB format image."""
        jpeg_buf = jpeg_instance.encode(sample_rgb_image, pixel_format=TJPF_RGB)
        assert jpeg_buf is not None
        assert len(jpeg_buf) > 0
    
    def test_encode_grayscale(self, jpeg_instance, sample_gray_image):
        """Test encoding grayscale image."""
        jpeg_buf = jpeg_instance.encode(
            sample_gray_image, 
            pixel_format=TJPF_GRAY,
            jpeg_subsample=TJSAMP_GRAY
        )
        assert jpeg_buf is not None
        assert len(jpeg_buf) > 0
    
    def test_encode_subsample_variations(self, jpeg_instance, sample_bgr_image):
        """Test encoding with different subsample settings."""
        # Test different subsample modes
        for subsample in [TJSAMP_444, TJSAMP_422, TJSAMP_420]:
            jpeg_buf = jpeg_instance.encode(sample_bgr_image, jpeg_subsample=subsample)
            assert len(jpeg_buf) > 0
    
    def test_encode_with_progressive_flag(self, jpeg_instance, sample_bgr_image):
        """Test encoding with progressive flag."""
        jpeg_buf = jpeg_instance.encode(sample_bgr_image, flags=TJFLAG_PROGRESSIVE)
        assert jpeg_buf is not None
        assert len(jpeg_buf) > 0
    
    def test_encode_in_place(self, jpeg_instance, sample_bgr_image):
        """Test in-place encoding to pre-allocated buffer."""
        buffer_size = jpeg_instance.buffer_size(sample_bgr_image)
        dst_buf = bytearray(buffer_size)
        result, n_bytes = jpeg_instance.encode(sample_bgr_image, dst=dst_buf)
        assert result is dst_buf
        assert id(result) == id(dst_buf)
        assert n_bytes > 0
        assert n_bytes <= buffer_size

    def test_encode_rejects_immutable_destination_without_mutating_it(
            self, jpeg_instance, sample_bgr_image):
        """Immutable bytes must never be passed to TurboJPEG as output."""
        dst = bytes(jpeg_instance.buffer_size(sample_bgr_image))
        before = dst
        before_hash = hash(dst)

        with pytest.raises(TypeError, match='writable'):
            jpeg_instance.encode(sample_bgr_image, dst=dst)

        assert dst == before
        assert hash(dst) == before_hash

    def test_encode_rejects_noncontiguous_destination_without_writing(
            self, jpeg_instance, sample_bgr_image):
        """Strided buffer views must be rejected before native compression."""
        buffer_size = jpeg_instance.buffer_size(sample_bgr_image)
        backing = bytearray([0xA5]) * (buffer_size * 2)
        dst = memoryview(backing)[::2]
        before = bytes(backing)
        assert dst.nbytes == buffer_size
        assert not dst.c_contiguous

        with pytest.raises(ValueError, match='C-contiguous'):
            jpeg_instance.encode(sample_bgr_image, dst=dst)

        assert bytes(backing) == before

    def test_encode_rejects_destination_smaller_than_worst_case(
            self, jpeg_instance, sample_bgr_image):
        """A short destination must raise instead of triggering reallocation."""
        buffer_size = jpeg_instance.buffer_size(sample_bgr_image)
        dst = bytearray([0xA5]) * (buffer_size - 1)
        before = bytes(dst)

        with pytest.raises(ValueError, match='buffer is too small'):
            jpeg_instance.encode(sample_bgr_image, dst=dst)

        assert bytes(dst) == before

    def test_encode_uses_destination_nbytes_not_element_count(
            self, jpeg_instance, sample_bgr_image):
        """Typed writable buffers are sized in bytes, not Python elements."""
        buffer_size = jpeg_instance.buffer_size(sample_bgr_image)
        dst = np.zeros((buffer_size + 3) // 4, dtype=np.uint32)
        assert len(dst) < buffer_size
        assert memoryview(dst).nbytes >= buffer_size

        result, n_bytes = jpeg_instance.encode(sample_bgr_image, dst=dst)

        assert result is dst
        jpeg_data = memoryview(dst).cast('B')[:n_bytes].tobytes()
        assert jpeg_data[:2] == b'\xff\xd8'
        assert jpeg_instance.decode(jpeg_data).shape == sample_bgr_image.shape

    def test_lossless_encode_uses_444_size_for_destination(
            self, jpeg_instance, sample_bgr_image):
        """Lossless in-place encoding must reserve the 4:4:4 worst case."""
        buffer_size = jpeg_instance.buffer_size(
            sample_bgr_image, jpeg_subsample=TJSAMP_444)
        dst = bytearray(buffer_size)

        result, n_bytes = jpeg_instance.encode(
            sample_bgr_image, dst=dst, lossless=True)

        assert result is dst
        decoded = jpeg_instance.decode(bytes(dst[:n_bytes]))
        assert np.array_equal(decoded, sample_bgr_image)

    def test_encode_frees_native_buffer_after_fatal_error(
            self, jpeg_instance, sample_bgr_image, monkeypatch):
        """A native output allocation is freed when compression fails."""
        native_free = jpeg_instance._TurboJPEG__free
        allocated = jpeg_instance._TurboJPEG__alloc(128)
        free_calls = []

        def fail_after_allocating(
                handle, src, width, pitch, height, pixel_format,
                jpeg_buf, jpeg_size):
            cast(jpeg_buf, POINTER(c_void_p))[0] = c_void_p(allocated)
            cast(jpeg_size, POINTER(c_size_t))[0] = c_size_t(128)
            return -1

        def raise_native_error(handle):
            raise OSError('injected compression failure')

        def recording_free(buffer):
            free_calls.append(buffer.value)
            native_free(buffer)

        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__compress', fail_after_allocating)
        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__report_error', raise_native_error)
        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__free', recording_free)

        try:
            with pytest.raises(OSError, match='injected compression failure'):
                jpeg_instance.encode(sample_bgr_image)
        finally:
            if not free_calls:
                native_free(c_void_p(allocated))

        assert free_calls == [allocated]

    def test_encode_does_not_free_destination_after_fatal_error(
            self, jpeg_instance, sample_bgr_image, monkeypatch):
        """A caller-provided destination remains caller-owned on failure."""
        dst = bytearray(jpeg_instance.buffer_size(sample_bgr_image))
        free_calls = []

        def fail_without_reallocating(
                handle, src, width, pitch, height, pixel_format,
                jpeg_buf, jpeg_size):
            return -1

        def raise_native_error(handle):
            raise OSError('injected compression failure')

        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__compress', fail_without_reallocating)
        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__report_error', raise_native_error)
        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__free',
            lambda buffer: free_calls.append(buffer.value))

        with pytest.raises(OSError, match='injected compression failure'):
            jpeg_instance.encode(sample_bgr_image, dst=dst)

        assert free_calls == []
    
    def test_encode_decode_roundtrip(self, jpeg_instance, sample_bgr_image):
        """Test that encoding and decoding preserves image dimensions."""
        jpeg_buf = jpeg_instance.encode(sample_bgr_image)
        decoded = jpeg_instance.decode(jpeg_buf)
        assert decoded.shape == sample_bgr_image.shape
    
    def test_encode_invalid_shape(self, jpeg_instance):
        """Test encode with invalid image shape raises ValueError."""
        # 1D array instead of 2D/3D
        invalid_img = np.zeros(100, dtype=np.uint8)
        with pytest.raises(ValueError):
            jpeg_instance.encode(invalid_img)


class TestEncodeFromYUV:
    """Test encode_from_yuv function."""
    
    def test_encode_from_yuv_basic(self, jpeg_instance, encoded_sample_jpeg):
        """Test encoding from YUV buffer."""
        # First decode to YUV
        yuv_buffer, plane_sizes = jpeg_instance.decode_to_yuv(encoded_sample_jpeg)
        # Then encode back from YUV
        jpeg_buf = jpeg_instance.encode_from_yuv(
            yuv_buffer, 
            height=100, 
            width=100,
            jpeg_subsample=TJSAMP_422
        )
        assert jpeg_buf is not None
        assert len(jpeg_buf) > 0
    
    def test_encode_from_yuv_quality(self, jpeg_instance, encoded_sample_jpeg):
        """Test encoding from YUV with different quality levels."""
        yuv_buffer, _ = jpeg_instance.decode_to_yuv(encoded_sample_jpeg)
        jpeg_low = jpeg_instance.encode_from_yuv(yuv_buffer, 100, 100, quality=50)
        jpeg_high = jpeg_instance.encode_from_yuv(yuv_buffer, 100, 100, quality=95)
        assert len(jpeg_high) > len(jpeg_low)

    @pytest.mark.parametrize('align', [1, 2, 4, 8])
    def test_encode_from_yuv_accepts_matching_alignment(
            self, jpeg_instance, align):
        """YUV buffers decoded with a custom alignment can be re-encoded."""
        height, width = 19, 17
        image = np.arange(
            height * width * 3, dtype=np.uint8).reshape(height, width, 3)
        source = jpeg_instance.encode(
            image, jpeg_subsample=TJSAMP_420)
        yuv_buffer, _ = jpeg_instance.decode_to_yuv(source, pad=align)

        jpeg_buf = jpeg_instance.encode_from_yuv(
            yuv_buffer,
            height=height,
            width=width,
            jpeg_subsample=TJSAMP_420,
            align=align,
        )

        assert jpeg_instance.decode_header(jpeg_buf)[:2] == (width, height)

    @pytest.mark.parametrize('jpeg_subsample', [
        TJSAMP_444,
        TJSAMP_422,
        TJSAMP_420,
        TJSAMP_GRAY,
        TJSAMP_440,
        TJSAMP_411,
        TJSAMP_441,
    ])
    def test_encode_from_yuv_validates_all_subsampling_layouts(
            self, jpeg_instance, jpeg_subsample):
        """Capacity calculation follows every TurboJPEG YUV layout."""
        height, width = 19, 17
        image = np.arange(
            height * width * 3, dtype=np.uint8).reshape(height, width, 3)
        source = jpeg_instance.encode(
            image, jpeg_subsample=jpeg_subsample)
        yuv_buffer, _ = jpeg_instance.decode_to_yuv(source, pad=4)

        jpeg_buf = jpeg_instance.encode_from_yuv(
            yuv_buffer,
            height=height,
            width=width,
            jpeg_subsample=jpeg_subsample,
            align=4,
        )

        decoded_header = jpeg_instance.decode_header(jpeg_buf)
        assert decoded_header[:3] == (width, height, jpeg_subsample)

    def test_encode_from_yuv_rejects_short_buffer(
            self, jpeg_instance, encoded_sample_jpeg):
        """A short logical view is rejected before native code can read it."""
        yuv_buffer, _ = jpeg_instance.decode_to_yuv(
            encoded_sample_jpeg, pad=8)
        short_buffer = yuv_buffer[:-1]
        before = short_buffer.copy()

        with pytest.raises(ValueError, match='requires at least'):
            jpeg_instance.encode_from_yuv(
                short_buffer,
                height=100,
                width=100,
                jpeg_subsample=TJSAMP_422,
                align=8,
            )

        assert np.array_equal(short_buffer, before)

    def test_encode_from_yuv_rejects_empty_buffer(self, jpeg_instance):
        """An empty source cannot satisfy a non-empty YUV layout."""
        with pytest.raises(ValueError, match='requires at least'):
            jpeg_instance.encode_from_yuv(
                np.empty(0, dtype=np.uint8),
                height=16,
                width=16,
            )

    def test_encode_from_yuv_rejects_wrong_dtype(self, jpeg_instance):
        """Element count cannot disguise a non-byte YUV source."""
        source = np.zeros(1024, dtype=np.uint16)

        with pytest.raises(ValueError, match='uint8'):
            jpeg_instance.encode_from_yuv(
                source,
                height=16,
                width=16,
            )

    def test_encode_from_yuv_rejects_non_contiguous_source(
            self, jpeg_instance):
        """A strided source is not silently copied or read as packed YUV."""
        backing = np.zeros(4096, dtype=np.uint8)
        source = backing[::2]

        with pytest.raises(ValueError, match='C-contiguous'):
            jpeg_instance.encode_from_yuv(
                source,
                height=16,
                width=16,
            )

    @pytest.mark.parametrize('align', [0, -1, 3, 6])
    def test_encode_from_yuv_rejects_invalid_alignment(
            self, jpeg_instance, align):
        """TurboJPEG requires a positive power-of-two row alignment."""
        source = np.zeros(4096, dtype=np.uint8)

        with pytest.raises(ValueError, match='power of two'):
            jpeg_instance.encode_from_yuv(
                source,
                height=16,
                width=16,
                align=align,
            )

    def test_encode_from_yuv_rejects_unrepresentable_alignment_early(
            self, jpeg_instance):
        """A pathological alignment is rejected before old native size code."""
        source = np.zeros(4096, dtype=np.uint8)

        with patch.object(
                jpeg_instance,
                '_TurboJPEG__buffer_size_YUV',
                side_effect=AssertionError('native size calculation called')):
            with pytest.raises(ValueError, match='buffer is too small'):
                jpeg_instance.encode_from_yuv(
                    source,
                    height=16,
                    width=16,
                    align=1 << 30,
                )

    @pytest.mark.parametrize(('height', 'width'), [
        (0, 16),
        (16, 0),
        (-1, 16),
        (16, -1),
    ])
    def test_encode_from_yuv_rejects_invalid_dimensions(
            self, jpeg_instance, height, width):
        """Dimensions must describe a positive native YUV layout."""
        source = np.zeros(4096, dtype=np.uint8)

        with pytest.raises(ValueError, match='positive integer'):
            jpeg_instance.encode_from_yuv(
                source,
                height=height,
                width=width,
            )


class TestScaleWithQuality:
    """Test scale_with_quality function."""
    
    def test_scale_with_quality_basic(self, jpeg_instance, encoded_sample_jpeg):
        """Test scaling JPEG with quality adjustment."""
        scaled_jpeg = jpeg_instance.scale_with_quality(
            encoded_sample_jpeg,
            scaling_factor=(1, 2),
            quality=85
        )
        assert scaled_jpeg is not None
        assert isinstance(scaled_jpeg, bytes)
        # Verify the scaled image is smaller
        width, height, _, _ = jpeg_instance.decode_header(scaled_jpeg)
        assert width == 50
        assert height == 50
    
    def test_scale_with_quality_no_scaling(self, jpeg_instance, encoded_sample_jpeg):
        """Test quality adjustment without scaling."""
        scaled_jpeg = jpeg_instance.scale_with_quality(
            encoded_sample_jpeg,
            scaling_factor=(1, 1),
            quality=70
        )
        assert scaled_jpeg is not None
    
    def test_scale_with_quality_quarter(self, jpeg_instance, encoded_sample_jpeg):
        """Test scaling to quarter size with quality."""
        scaled_jpeg = jpeg_instance.scale_with_quality(
            encoded_sample_jpeg,
            scaling_factor=(1, 4),
            quality=80
        )
        width, height, _, _ = jpeg_instance.decode_header(scaled_jpeg)
        assert width == 25
        assert height == 25


class TestCrop:
    """Test crop function."""
    
    def test_crop_basic(self, jpeg_instance, encoded_sample_jpeg):
        """Test basic lossless crop operation."""
        # Crop a 50x50 region from the center (aligned to MCU blocks)
        cropped = jpeg_instance.crop(encoded_sample_jpeg, 16, 16, 64, 64)
        assert cropped is not None
        assert isinstance(cropped, bytes)
        # Verify cropped dimensions
        width, height, _, _ = jpeg_instance.decode_header(cropped)
        assert width == 64
        assert height == 64
    
    def test_crop_with_gray(self, jpeg_instance, encoded_sample_jpeg):
        """Test crop with grayscale conversion."""
        cropped = jpeg_instance.crop(encoded_sample_jpeg, 0, 0, 64, 64, gray=True)
        assert cropped is not None
        width, height, subsample, _ = jpeg_instance.decode_header(cropped)
        assert subsample == TJSAMP_GRAY
    
    def test_crop_with_preserve(self, jpeg_instance, encoded_sample_jpeg):
        """Test crop with preserve flag."""
        # preserve flag adjusts boundaries to MCU block size
        cropped = jpeg_instance.crop(encoded_sample_jpeg, 10, 10, 50, 50, preserve=True)
        assert cropped is not None


class TestCropMultiple:
    """Test crop_multiple function."""
    
    def test_crop_multiple_basic(self, jpeg_instance, encoded_sample_jpeg):
        """Test multiple crop operations."""
        crop_params = [
            (0, 0, 48, 48),
            (16, 16, 48, 48),
        ]
        cropped_list = jpeg_instance.crop_multiple(encoded_sample_jpeg, crop_params)
        assert isinstance(cropped_list, list)
        assert len(cropped_list) == 2
        for cropped in cropped_list:
            assert isinstance(cropped, bytes)
            assert len(cropped) > 0
    
    def test_crop_multiple_with_background(self, jpeg_instance, encoded_sample_jpeg):
        """Test crop multiple with background luminance."""
        crop_params = [
            (0, 0, 48, 48),
        ]
        cropped_list = jpeg_instance.crop_multiple(
            encoded_sample_jpeg, 
            crop_params,
            background_luminance=0.5
        )
        assert len(cropped_list) == 1
    
    def test_crop_multiple_with_gray(self, jpeg_instance, encoded_sample_jpeg):
        """Test crop multiple with grayscale conversion."""
        crop_params = [
            (0, 0, 48, 48),
        ]
        cropped_list = jpeg_instance.crop_multiple(
            encoded_sample_jpeg, 
            crop_params,
            gray=True
        )
        assert len(cropped_list) == 1
        width, height, subsample, _ = jpeg_instance.decode_header(cropped_list[0])
        assert subsample == TJSAMP_GRAY


class TestOptimize:
    """Test optimize function."""

    def test_optimize_is_lossless(self, jpeg_instance, encoded_sample_jpeg):
        """Test optimization returns a valid JPEG with unchanged pixels."""
        optimized = jpeg_instance.optimize(encoded_sample_jpeg)
        original_pixels = jpeg_instance.decode(encoded_sample_jpeg)
        optimized_pixels = jpeg_instance.decode(optimized)
        assert np.array_equal(original_pixels, optimized_pixels)


class TestBufferSize:
    """Test buffer_size function."""
    
    def test_buffer_size_basic(self, jpeg_instance, sample_bgr_image):
        """Test buffer size calculation."""
        size = jpeg_instance.buffer_size(sample_bgr_image)
        assert size > 0
        assert isinstance(size, int)
    
    def test_buffer_size_different_subsamples(self, jpeg_instance, sample_bgr_image):
        """Test buffer size with different subsample modes."""
        size_444 = jpeg_instance.buffer_size(sample_bgr_image, jpeg_subsample=TJSAMP_444)
        size_420 = jpeg_instance.buffer_size(sample_bgr_image, jpeg_subsample=TJSAMP_420)
        # 4:4:4 should require more space than 4:2:0
        assert size_444 >= size_420
    
    def test_buffer_size_sufficient(self, jpeg_instance, sample_bgr_image):
        """Test that calculated buffer size is sufficient for encoding."""
        buffer_size = jpeg_instance.buffer_size(sample_bgr_image)
        jpeg_buf = jpeg_instance.encode(sample_bgr_image)
        assert len(jpeg_buf) <= buffer_size


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_jpeg_decode(self, jpeg_instance):
        """Test decoding invalid JPEG data."""
        with pytest.raises(OSError):
            jpeg_instance.decode(b'invalid jpeg data')
    
    def test_invalid_jpeg_decode_header(self, jpeg_instance):
        """Test decode_header with invalid data."""
        with pytest.raises(OSError):
            jpeg_instance.decode_header(b'invalid')
    
    def test_empty_buffer_decode(self, jpeg_instance):
        """Test decoding empty buffer."""
        with pytest.raises(OSError):
            jpeg_instance.decode(b'')
    
    def test_empty_buffer_decode_header(self, jpeg_instance):
        """Test decode_header with empty buffer."""
        with pytest.raises(OSError):
            jpeg_instance.decode_header(b'')


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_encode_decode_cycle(self, jpeg_instance):
        """Test complete encode-decode cycle preserves data."""
        # Create test image
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Encode
        jpeg_buf = jpeg_instance.encode(original, quality=95)
        
        # Decode
        decoded = jpeg_instance.decode(jpeg_buf)
        
        # Check dimensions match
        assert decoded.shape == original.shape
        
        # JPEG is lossy, but with high quality the difference should be small
        # We just verify dimensions and data type
        assert decoded.dtype == original.dtype
    
    def test_various_pixel_formats_roundtrip(self, jpeg_instance, sample_bgr_image):
        """Test encoding and decoding with various pixel formats."""
        pixel_formats = [TJPF_RGB, TJPF_BGR, TJPF_GRAY, TJPF_RGBA, TJPF_BGRA]
        
        for pf in pixel_formats:
            # For GRAY format, use grayscale image
            if pf == TJPF_GRAY:
                img = sample_bgr_image[:, :, 0:1]
                subsample = TJSAMP_GRAY
            # For RGBA/BGRA formats, add alpha channel
            elif pf in [TJPF_RGBA, TJPF_BGRA]:
                alpha = np.full((100, 100, 1), 255, dtype=np.uint8)
                img = np.concatenate([sample_bgr_image, alpha], axis=2)
                subsample = TJSAMP_422
            else:
                img = sample_bgr_image
                subsample = TJSAMP_422
            
            # Encode
            jpeg_buf = jpeg_instance.encode(img, pixel_format=pf, jpeg_subsample=subsample)
            
            # Decode
            decoded = jpeg_instance.decode(jpeg_buf, pixel_format=pf)
            
            # Verify shape matches
            assert decoded.shape[:2] == img.shape[:2]
    
    def test_multiple_instances(self):
        """Test that multiple TurboJPEG instances can coexist."""
        tj1 = TurboJPEG()
        tj2 = TurboJPEG()
        
        assert tj1 is not None
        assert tj2 is not None
        assert tj1.scaling_factors == tj2.scaling_factors
    
    def test_image_with_different_sizes(self, jpeg_instance):
        """Test encoding/decoding images of various sizes."""
        sizes = [(50, 50), (100, 200), (256, 256), (1024, 768)]
        
        for width, height in sizes:
            img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            jpeg_buf = jpeg_instance.encode(img)
            decoded = jpeg_instance.decode(jpeg_buf)
            assert decoded.shape == (height, width, 3)



# ============================================================================
# Regression Tests for Historical Bugs and Edge Cases
# ============================================================================


class TestBufferHandlingRobustness:
    """
    Test robustness of buffer handling with invalid, empty, and truncated data.
    
    These tests ensure that invalid buffers raise RuntimeError or OSError
    instead of crashing the interpreter.
    """
    
    def test_decode_empty_buffer(self, jpeg_instance):
        """Test that decoding an empty buffer raises an error instead of crashing."""
        with pytest.raises((RuntimeError, OSError)):
            jpeg_instance.decode(b'')
    
    def test_decode_header_empty_buffer(self, jpeg_instance):
        """Test that decode_header with empty buffer raises an error."""
        with pytest.raises((RuntimeError, OSError)):
            jpeg_instance.decode_header(b'')
    
    def test_decode_truncated_jpeg_header_very_short(self, jpeg_instance):
        """Test that decoding a very short truncated JPEG handles error gracefully."""
        # JPEG files start with FF D8 FF, but this is incomplete
        truncated_header = b'\xFF\xD8'
        
        # Should either raise an error or return empty array with warning
        # TurboJPEG 3.0+ may raise ValueError for negative dimensions or emit warning
        try:
            with pytest.warns(UserWarning, match="(JPEG datastream|Premature end of JPEG file)"):
                result = jpeg_instance.decode(truncated_header)
                # If it doesn't raise, should return empty or minimal array
                assert result.size == 0 or result.shape[0] == 0 or result.shape[1] == 0
        except (RuntimeError, OSError, IOError, ValueError):
            # This is also acceptable - raising an error for invalid data
            # TurboJPEG 3.0+ may raise ValueError for negative dimensions
            pass
    
    def test_decode_truncated_jpeg_header_partial(self, jpeg_instance):
        """Test that decoding a partially truncated JPEG header handles error gracefully."""
        # Partial JPEG header (missing actual image data)
        truncated_header = b'\xFF\xD8\xFF\xE0\x00\x10JFIF'
        
        # Should either raise an error or return empty/minimal array
        try:
            result = jpeg_instance.decode(truncated_header)
            # If it doesn't raise, should return empty or minimal array
            assert result.size == 0 or result.shape[0] == 0 or result.shape[1] == 0
        except (RuntimeError, OSError, IOError, ValueError):
            # This is also acceptable - raising an error for invalid data
            # TurboJPEG 3.0+ may raise ValueError for negative dimensions
            pass
    
    def test_decode_truncated_jpeg_data(self, jpeg_instance, valid_jpeg):
        """Test that decoding truncated JPEG data handles error gracefully."""
        # Take only first 50 bytes of a valid JPEG
        truncated_jpeg = valid_jpeg[:50]
        
        # Should either raise an error or return empty/minimal array
        try:
            result = jpeg_instance.decode(truncated_jpeg)
            # If it doesn't raise, should return empty or minimal array
            assert result.size == 0 or result.shape[0] == 0 or result.shape[1] == 0
        except (RuntimeError, OSError, IOError, ValueError):
            # This is also acceptable - raising an error for invalid data
            # TurboJPEG 3.0+ may raise ValueError for negative dimensions
            pass
    
    def test_decode_invalid_jpeg_magic_number(self, jpeg_instance):
        """Test that decoding data with invalid JPEG magic number raises an error."""
        # Invalid magic number (JPEG should start with FF D8)
        invalid_data = b'\x00\x00\xFF\xE0\x00\x10JFIF\x00\x01\x01'
        with pytest.raises((RuntimeError, OSError)):
            jpeg_instance.decode(invalid_data)
    
    def test_decode_random_bytes(self, jpeg_instance):
        """Test that decoding random bytes raises an error."""
        random_data = np.random.bytes(1000)
        with pytest.raises((RuntimeError, OSError)):
            jpeg_instance.decode(random_data)
    
    def test_decode_non_jpeg_image_data(self, jpeg_instance):
        """Test that decoding non-JPEG data (e.g., PNG header) raises an error."""
        # PNG file signature
        png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        with pytest.raises((RuntimeError, OSError)):
            jpeg_instance.decode(png_header)
    
    def test_decode_header_truncated_data(self, jpeg_instance, valid_jpeg):
        """Test that decode_header with truncated data handles error gracefully."""
        # Truncate to just a few bytes
        truncated = valid_jpeg[:20]
        
        # Should either raise an error or return zeros/empty values
        try:
            width, height, subsample, colorspace = jpeg_instance.decode_header(truncated)
            # If it doesn't raise, should return zeros, minimal values, or -1 (TJ 3.0+)
            assert width == 0 or height == 0 or width == -1 or height == -1
        except (RuntimeError, OSError, IOError):
            # This is also acceptable - raising an error for invalid data
            pass
    
    def test_decode_corrupted_jpeg_middle(self, jpeg_instance, valid_jpeg):
        """Test that decoding a JPEG with corrupted middle section handles error gracefully."""
        # Corrupt the middle of the JPEG
        corrupted = bytearray(valid_jpeg)
        mid_point = len(corrupted) // 2
        corrupted[mid_point:mid_point+10] = b'\x00' * 10
        
        # libturbojpeg is resilient and may decode with a warning or raise an error
        try:
            # It may issue a warning but still decode
            result = jpeg_instance.decode(bytes(corrupted))
            # If it decodes, result should be valid dimensions
            assert result.shape[0] > 0 and result.shape[1] > 0
        except (RuntimeError, OSError):
            # This is also acceptable - raising an error for corrupted data
            pass
    
    def test_decode_to_yuv_empty_buffer(self, jpeg_instance):
        """Test that decode_to_yuv with empty buffer raises an error."""
        with pytest.raises((RuntimeError, OSError)):
            jpeg_instance.decode_to_yuv(b'')
    
    def test_decode_to_yuv_invalid_data(self, jpeg_instance):
        """Test that decode_to_yuv with invalid data raises an error."""
        with pytest.raises((RuntimeError, OSError)):
            jpeg_instance.decode_to_yuv(b'invalid jpeg data')


class TestLibraryLoading:
    """
    Test library loading logic to ensure clear error messages when library is not found.
    
    These tests verify that helpful error messages are provided for both Linux and Windows.
    """
    
    def test_library_loading_with_invalid_path(self):
        """Test that providing an invalid library path raises an appropriate error."""
        with pytest.raises((OSError, RuntimeError)):
            TurboJPEG(lib_path='/nonexistent/path/to/libturbojpeg.so')
    
    def test_library_loading_error_message(self):
        """Test that library loading failure provides a helpful error message."""
        # Mock find_library to return None and mock os.path.exists to return False
        with patch('turbojpeg.find_library', return_value=None):
            with patch('os.path.exists', return_value=False):
                with pytest.raises(RuntimeError) as excinfo:
                    TurboJPEG()
                
                # Verify error message is helpful
                error_msg = str(excinfo.value)
                assert 'Unable to locate turbojpeg library' in error_msg
                assert 'TurboJPEG(lib_path)' in error_msg
    
    def test_successful_library_loading_with_explicit_path(self):
        """Test that library can be loaded with an explicit valid path."""
        # Find the actual library path
        lib_path = find_library('turbojpeg')
        if lib_path is None:
            # Try default paths
            import platform
            from turbojpeg import DEFAULT_LIB_PATHS
            for path in DEFAULT_LIB_PATHS[platform.system()]:
                if os.path.exists(path):
                    lib_path = path
                    break
        
        if lib_path:
            # This should succeed
            tj = TurboJPEG(lib_path=lib_path)
            assert tj is not None
        else:
            pytest.skip("Could not find turbojpeg library path")
    
    def test_version_detection_rejects_old_library(self):
        """Test that PyTurboJPEG 2.0 rejects TurboJPEG 2.x library with clear error."""
        from unittest.mock import Mock, patch
        from ctypes import CDLL
        
        # Create a mock library that simulates TurboJPEG 2.x (missing tj3Init)
        mock_old_lib = Mock(spec=CDLL)
        
        # Add TurboJPEG 2.x functions but NOT tj3Init
        mock_old_lib.tjInitDecompress = Mock()
        mock_old_lib.tjInitCompress = Mock()
        mock_old_lib.tjDestroy = Mock()
        mock_old_lib.tjGetScalingFactors = Mock(return_value=Mock())
        
        # Patch cdll.LoadLibrary to return our mock old library
        with patch('turbojpeg.cdll.LoadLibrary', return_value=mock_old_lib):
            with pytest.raises(RuntimeError) as excinfo:
                TurboJPEG(lib_path='/fake/path/libturbojpeg.so')
            
            # Verify error message is clear and actionable
            error_msg = str(excinfo.value)
            assert 'PyTurboJPEG 2.0 requires libjpeg-turbo 3.0 or later' in error_msg
            assert 'libjpeg-turbo 2.x or older' in error_msg
            assert 'upgrade' in error_msg.lower() or 'install' in error_msg.lower()
            # Should suggest using PyTurboJPEG 1.x as alternative
            assert 'PyTurboJPEG 1.x' in error_msg or '1.x' in error_msg

    def test_version_detection_accepts_new_library(self):
        """A real TurboJPEG 3.x library must complete construction."""
        lib_path = os.environ.get('TURBOJPEG_LIB_PATH')
        if lib_path is None:
            lib_path = find_library('turbojpeg')
        if lib_path is None:
            pytest.skip('Could not find a TurboJPEG 3.x library')

        tj = TurboJPEG(lib_path=lib_path)

        assert tj.scaling_factors


class TestColorspaceConsistency:
    """
    Test colorspace consistency across all supported TJPF and TJSAMP combinations.
    
    These tests verify that the output buffer size matches expected dimensions
    for all combinations of pixel formats and subsampling modes.
    """
    
    # Pixel formats and their expected channel counts
    PIXEL_FORMATS = [
        (TJPF_RGB, 3),
        (TJPF_BGR, 3),
        (TJPF_GRAY, 1),
        (TJPF_RGBA, 4),
        (TJPF_BGRA, 4),
        (TJPF_RGBX, 4),
        (TJPF_BGRX, 4),
        (TJPF_XBGR, 4),
        (TJPF_XRGB, 4),
        (TJPF_ABGR, 4),
        (TJPF_ARGB, 4),
    ]
    
    # Subsampling modes compatible with color images
    SUBSAMPLE_MODES_COLOR = [
        TJSAMP_444,
        TJSAMP_422,
        TJSAMP_420,
        TJSAMP_440,
        TJSAMP_411,
        TJSAMP_441,
    ]
    
    @pytest.mark.parametrize("pixel_format,expected_channels", PIXEL_FORMATS)
    def test_encode_decode_all_pixel_formats(self, jpeg_instance, sample_image, pixel_format, expected_channels):
        """Test encoding and decoding with all supported pixel formats."""
        # Create an image with the right number of channels
        if expected_channels == 1:
            test_img = sample_image[:, :, 0:1]
            subsample = TJSAMP_GRAY
        elif expected_channels == 4:
            alpha = np.full((100, 100, 1), 255, dtype=np.uint8)
            test_img = np.concatenate([sample_image, alpha], axis=2)
            subsample = TJSAMP_422
        else:
            test_img = sample_image
            subsample = TJSAMP_422
        
        # Encode
        jpeg_buf = jpeg_instance.encode(test_img, pixel_format=pixel_format, jpeg_subsample=subsample)
        assert isinstance(jpeg_buf, bytes)
        assert len(jpeg_buf) > 0
        
        # Decode
        decoded = jpeg_instance.decode(jpeg_buf, pixel_format=pixel_format)
        
        # Verify dimensions match
        assert decoded.shape[0] == 100  # height
        assert decoded.shape[1] == 100  # width
        assert decoded.shape[2] == expected_channels
        assert decoded.dtype == np.uint8
    
    @pytest.mark.parametrize("subsample", SUBSAMPLE_MODES_COLOR)
    def test_encode_all_subsample_modes(self, jpeg_instance, sample_image, subsample):
        """Test encoding with all supported chrominance subsampling modes."""
        jpeg_buf = jpeg_instance.encode(sample_image, jpeg_subsample=subsample)
        assert isinstance(jpeg_buf, bytes)
        assert len(jpeg_buf) > 0
        
        # Verify we can decode it back
        decoded = jpeg_instance.decode(jpeg_buf)
        assert decoded.shape == (100, 100, 3)
    
    def test_encode_decode_gray_subsample(self, jpeg_instance, sample_image):
        """Test encoding and decoding with grayscale subsampling."""
        gray_img = sample_image[:, :, 0:1]
        
        # Encode
        jpeg_buf = jpeg_instance.encode(
            gray_img, 
            pixel_format=TJPF_GRAY,
            jpeg_subsample=TJSAMP_GRAY
        )
        assert isinstance(jpeg_buf, bytes)
        assert len(jpeg_buf) > 0
        
        # Decode
        decoded = jpeg_instance.decode(jpeg_buf, pixel_format=TJPF_GRAY)
        assert decoded.shape == (100, 100, 1)
    
    @pytest.mark.parametrize("pixel_format,expected_channels", [
        (TJPF_RGB, 3),
        (TJPF_BGR, 3),
        (TJPF_RGBA, 4),
        (TJPF_BGRA, 4),
    ])
    @pytest.mark.parametrize("subsample", [TJSAMP_444, TJSAMP_422, TJSAMP_420])
    def test_encode_decode_combinations(self, jpeg_instance, sample_image, pixel_format, expected_channels, subsample):
        """Test various combinations of pixel formats and subsampling modes."""
        # Create appropriate image
        if expected_channels == 4:
            alpha = np.full((100, 100, 1), 255, dtype=np.uint8)
            test_img = np.concatenate([sample_image, alpha], axis=2)
        else:
            test_img = sample_image
        
        # Encode
        jpeg_buf = jpeg_instance.encode(
            test_img, 
            pixel_format=pixel_format,
            jpeg_subsample=subsample
        )
        assert len(jpeg_buf) > 0
        
        # Decode
        decoded = jpeg_instance.decode(jpeg_buf, pixel_format=pixel_format)
        
        # Verify buffer size matches expected dimensions
        assert decoded.shape == (100, 100, expected_channels)
        assert decoded.dtype == np.uint8
    
    def test_buffer_size_calculation_consistency(self, jpeg_instance, sample_image):
        """Test that buffer_size calculation is consistent with actual encoding."""
        for subsample in [TJSAMP_444, TJSAMP_422, TJSAMP_420]:
            buffer_size = jpeg_instance.buffer_size(sample_image, jpeg_subsample=subsample)
            jpeg_buf = jpeg_instance.encode(sample_image, jpeg_subsample=subsample)
            
            # Actual encoded size should not exceed calculated buffer size
            assert len(jpeg_buf) <= buffer_size
            
            # Buffer size should be reasonable (not excessively large - within 3x of raw data)
            # The buffer size is conservative to ensure sufficient space
            assert buffer_size < len(sample_image.tobytes()) * 3


class TestMemoryManagement:
    """
    Test memory management stability with stress testing.
    
    These tests perform 1000+ compression/decompression cycles to check for
    memory leaks, segfaults, or other stability issues.
    
    pytest-memray is used to detect unexpected memory growth during repeated
    execution to catch slow memory leaks.
    """
    
    @pytest.mark.limit_memory("50 MB")
    def test_encode_decode_stress_1000_cycles(self, jpeg_instance):
        """Test 1000+ encode/decode cycles for memory stability with leak detection."""
        # Create a test image
        test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Perform 1000 cycles
        for i in range(1000):
            # Encode
            jpeg_buf = jpeg_instance.encode(test_img, quality=85)
            assert len(jpeg_buf) > 0
            
            # Decode
            decoded = jpeg_instance.decode(jpeg_buf)
            assert decoded.shape == test_img.shape
            
            # Occasionally modify the image to avoid caching effects
            if i % 100 == 0:
                test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    @pytest.mark.limit_memory("100 MB")
    def test_encode_decode_varying_sizes_stress(self, jpeg_instance):
        """Test encode/decode cycles with varying image sizes for memory stability."""
        sizes = [(50, 50), (100, 100), (200, 200), (150, 100), (100, 150)]
        
        # Perform 200 cycles per size (1000 total)
        for size in sizes:
            height, width = size
            for i in range(200):
                # Create image
                test_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                
                # Encode
                jpeg_buf = jpeg_instance.encode(test_img)
                assert len(jpeg_buf) > 0
                
                # Decode
                decoded = jpeg_instance.decode(jpeg_buf)
                assert decoded.shape == (height, width, 3)
    
    @pytest.mark.limit_memory("50 MB")
    def test_encode_decode_different_formats_stress(self, jpeg_instance):
        """Test encode/decode cycles with different pixel formats for stability."""
        formats = [TJPF_RGB, TJPF_BGR, TJPF_GRAY]
        
        # Perform ~333 cycles per format (1000 total)
        for pixel_format in formats:
            for i in range(334):
                # Create appropriate image
                if pixel_format == TJPF_GRAY:
                    test_img = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
                    subsample = TJSAMP_GRAY
                else:
                    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
                    subsample = TJSAMP_422
                
                # Encode
                jpeg_buf = jpeg_instance.encode(
                    test_img, 
                    pixel_format=pixel_format,
                    jpeg_subsample=subsample
                )
                assert len(jpeg_buf) > 0
                
                # Decode
                decoded = jpeg_instance.decode(jpeg_buf, pixel_format=pixel_format)
                assert decoded.shape == test_img.shape
    
    @pytest.mark.limit_memory("20 MB")
    def test_decode_header_stress(self, jpeg_instance, valid_jpeg):
        """Test decode_header repeatedly for memory stability."""
        # Perform 1000 decode_header operations
        for i in range(1000):
            width, height, subsample, colorspace = jpeg_instance.decode_header(valid_jpeg)
            assert width == 100
            assert height == 100
            assert subsample in [TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY]
            assert colorspace in [TJCS_RGB, TJCS_YCbCr, TJCS_GRAY]
    
    @pytest.mark.limit_memory("20 MB")
    def test_buffer_size_calculation_stress(self, jpeg_instance):
        """Test buffer_size calculation repeatedly for memory stability."""
        test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Perform 1000 buffer_size calculations
        for i in range(1000):
            buffer_size = jpeg_instance.buffer_size(test_img)
            assert buffer_size > 0
            assert isinstance(buffer_size, int)
    
    @pytest.mark.limit_memory("50 MB")
    def test_multiple_instances_stress(self):
        """Test creating and using multiple TurboJPEG instances for stability."""
        test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Create 100 instances and perform 10 operations each (1000 total)
        for i in range(100):
            tj = TurboJPEG()
            for j in range(10):
                jpeg_buf = tj.encode(test_img)
                decoded = tj.decode(jpeg_buf)
                assert decoded.shape == test_img.shape


class TestCropFunctionality:
    """
    Test crop function with real input image based on Issue #88.
    
    These tests use a test input image to verify that crop operations
    produce expected results with correct dimensions and content.
    """
    
    @pytest.fixture(scope='class')
    def test_crop_image(self):
        """Load the test crop input image."""
        test_image_path = os.path.join(os.path.dirname(__file__), 'test_crop_input.jpg')
        if not os.path.exists(test_image_path):
            pytest.skip(f"Test image not found: {test_image_path}")
        
        with open(test_image_path, 'rb') as f:
            return f.read()
    
    def test_crop_input_image_loads(self, jpeg_instance, test_crop_image):
        """Test that the input image loads correctly."""
        # Verify we can decode the header
        width, height, subsample, colorspace = jpeg_instance.decode_header(test_crop_image)
        assert width == 200
        assert height == 200
        assert subsample in [TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY]
        assert colorspace in [TJCS_RGB, TJCS_YCbCr, TJCS_GRAY]
    
    def test_crop_top_left_quadrant(self, jpeg_instance, test_crop_image):
        """Test cropping the top-left quadrant (red area)."""
        # Crop top-left 96x96 region (MCU-aligned for 4:2:0 subsampling)
        cropped = jpeg_instance.crop(test_crop_image, 0, 0, 96, 96)
        
        # Verify cropped image properties
        assert cropped is not None
        assert isinstance(cropped, bytes)
        assert len(cropped) > 0
        
        # Verify dimensions
        width, height, _, _ = jpeg_instance.decode_header(cropped)
        assert width == 96
        assert height == 96
        
        # Decode and verify we got something reasonable
        decoded = jpeg_instance.decode(cropped)
        assert decoded.shape == (96, 96, 3)
    
    def test_crop_center_region(self, jpeg_instance, test_crop_image):
        """Test cropping a center region that spans multiple quadrants."""
        # Crop center 64x64 region (MCU-aligned)
        x, y = 64, 64
        w, h = 64, 64
        cropped = jpeg_instance.crop(test_crop_image, x, y, w, h)
        
        # Verify dimensions
        width, height, _, _ = jpeg_instance.decode_header(cropped)
        assert width == w
        assert height == h
        
        # Decode to verify
        decoded = jpeg_instance.decode(cropped)
        assert decoded.shape == (h, w, 3)
    
    def test_crop_with_mcu_alignment(self, jpeg_instance, test_crop_image):
        """Test that crop respects MCU block alignment."""
        # Test various crop positions that should align to MCU blocks
        test_cases = [
            (0, 0, 48, 48),      # Top-left, aligned
            (48, 0, 48, 48),     # Top, aligned
            (0, 48, 48, 48),     # Left, aligned
            (48, 48, 48, 48),    # Center, aligned
        ]
        
        for x, y, w, h in test_cases:
            cropped = jpeg_instance.crop(test_crop_image, x, y, w, h)
            width, height, _, _ = jpeg_instance.decode_header(cropped)
            assert width == w, f"Width mismatch for crop at ({x},{y},{w},{h})"
            assert height == h, f"Height mismatch for crop at ({x},{y},{w},{h})"
    
    def test_crop_with_preserve_flag(self, jpeg_instance, test_crop_image):
        """Test crop with preserve flag adjusts to MCU boundaries."""
        # Try to crop at non-aligned position with preserve=True
        # The preserve flag should adjust boundaries
        cropped = jpeg_instance.crop(test_crop_image, 10, 10, 50, 50, preserve=True)
        
        assert cropped is not None
        assert isinstance(cropped, bytes)
        
        # Dimensions may be adjusted to MCU boundaries
        width, height, _, _ = jpeg_instance.decode_header(cropped)
        assert width > 0
        assert height > 0
    
    def test_crop_to_grayscale(self, jpeg_instance, test_crop_image):
        """Test crop with grayscale conversion."""
        cropped = jpeg_instance.crop(test_crop_image, 0, 0, 96, 96, gray=True)
        
        assert cropped is not None
        
        # Verify grayscale subsampling
        width, height, subsample, _ = jpeg_instance.decode_header(cropped)
        assert width == 96
        assert height == 96
        assert subsample == TJSAMP_GRAY
    
    def test_crop_full_image(self, jpeg_instance, test_crop_image):
        """Cropping the full image retains right/bottom partial iMCUs."""
        # Get original dimensions
        orig_width, orig_height, _, _ = jpeg_instance.decode_header(test_crop_image)
        
        # Crop the entire image, including partial edge iMCUs.
        cropped = jpeg_instance.crop(test_crop_image, 0, 0, orig_width, orig_height)

        width, height, _, _ = jpeg_instance.decode_header(cropped)
        assert (width, height) == (orig_width, orig_height)
    
    def test_crop_multiple_regions(self, jpeg_instance, test_crop_image):
        """Test crop_multiple function with the test image."""
        crop_params = [
            (0, 0, 48, 48),      # Top-left quadrant
            (48, 0, 48, 48),     # Top-right area
            (0, 48, 48, 48),     # Bottom-left area
            (48, 48, 48, 48),    # Center area
        ]
        
        cropped_list = jpeg_instance.crop_multiple(test_crop_image, crop_params)
        
        assert isinstance(cropped_list, list)
        assert len(cropped_list) == len(crop_params)
        
        # Verify each cropped image
        for i, cropped in enumerate(cropped_list):
            assert isinstance(cropped, bytes)
            assert len(cropped) > 0
            
            # Verify dimensions
            width, height, _, _ = jpeg_instance.decode_header(cropped)
            expected_w, expected_h = crop_params[i][2], crop_params[i][3]
            assert width == expected_w
            assert height == expected_h
    
    def test_crop_edge_cases(self, jpeg_instance, test_crop_image):
        """Test crop at image edges."""
        orig_width, orig_height, _, _ = jpeg_instance.decode_header(test_crop_image)
        
        # Start on the final aligned iMCU and retain the partial right edge.
        right_edge_x = orig_width - (orig_width % 16 or 16)
        right_width = orig_width - right_edge_x
        cropped_right = jpeg_instance.crop(
            test_crop_image, right_edge_x, 0, right_width, 48)
        width, height, _, _ = jpeg_instance.decode_header(cropped_right)
        assert width == right_width
        assert height == 48

        # Start on the final aligned iMCU and retain the partial bottom edge.
        bottom_edge_y = orig_height - (orig_height % 16 or 16)
        bottom_height = orig_height - bottom_edge_y
        cropped_bottom = jpeg_instance.crop(
            test_crop_image, 0, bottom_edge_y, 48, bottom_height)
        width, height, _, _ = jpeg_instance.decode_header(cropped_bottom)
        assert width == 48
        assert height == bottom_height
    
    def test_crop_preserves_quality(self, jpeg_instance, test_crop_image):
        """Test that crop is lossless (same quality)."""
        # Crop a region
        cropped = jpeg_instance.crop(test_crop_image, 16, 16, 64, 64)
        
        # Decode both original region and cropped
        original_decoded = jpeg_instance.decode(test_crop_image)
        cropped_decoded = jpeg_instance.decode(cropped)
        
        # Original cropped region
        original_region = original_decoded[16:16+64, 16:16+64, :]
        
        # The shapes should match
        assert cropped_decoded.shape == original_region.shape
        
        # Due to JPEG being lossy, pixel values may differ slightly,
        # but the overall structure should be similar
        # We check that most pixels are close (within a tolerance)
        diff = np.abs(original_region.astype(np.int16) - cropped_decoded.astype(np.int16))
        # Allow up to 10 pixel value difference for most pixels (JPEG artifacts)
        close_pixels = np.sum(diff <= 10, axis=2) == 3  # All 3 channels close
        percentage_close = np.sum(close_pixels) / close_pixels.size
        assert percentage_close > 0.95, f"Only {percentage_close*100:.1f}% of pixels are close"


def check_16bit_support(jpeg_instance):
    """Check if TurboJPEG library supports 16-bit lossless precision."""
    try:
        # Try encoding a tiny 16-bit image in lossless mode
        img = np.full((2, 2, 3), 100, dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img)
        # Try decoding it back
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        return np.array_equal(img, decoded)
    except (IOError, OSError, NotImplementedError, ValueError):
        return False


@pytest.fixture(scope='module')
def supports_16bit(jpeg_instance):
    """Fixture to check if 16-bit is supported."""
    return check_16bit_support(jpeg_instance)


class TestHighPrecision:
    """Comprehensive tests for 12-bit and 16-bit precision JPEG support."""
    
    def test_encode_decode_12bit_basic(self, jpeg_instance, sample_12bit_image):
        """Test basic 12-bit encode/decode roundtrip."""
        # Encode 12-bit image
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        assert isinstance(jpeg_buf, bytes)
        assert len(jpeg_buf) > 0
        
        # Decode back to 12-bit
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert decoded.shape == sample_12bit_image.shape
        assert decoded.dtype == np.uint16
    
    def test_encode_decode_16bit_basic(self, jpeg_instance, sample_16bit_image, supports_16bit):
        """Test basic 16-bit encode/decode roundtrip."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        # Encode 16-bit image
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        assert isinstance(jpeg_buf, bytes)
        assert len(jpeg_buf) > 0
        
        # Decode back to 16-bit
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.shape == sample_16bit_image.shape
        assert decoded.dtype == np.uint16
    
    def test_12bit_image_shape_preservation(self, jpeg_instance, sample_12bit_image):
        """Test that 12-bit image dimensions are preserved."""
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert decoded.shape == sample_12bit_image.shape
    
    def test_16bit_image_shape_preservation(self, jpeg_instance, sample_16bit_image, supports_16bit):
        """Test that 16-bit image dimensions are preserved."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.shape == sample_16bit_image.shape
    
    def test_12bit_dtype_verification(self, jpeg_instance, sample_12bit_image):
        """Test that 12-bit decode returns uint16."""
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert decoded.dtype == np.uint16
    
    def test_16bit_dtype_verification(self, jpeg_instance, sample_16bit_image, supports_16bit):
        """Test that 16-bit decode returns uint16."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.dtype == np.uint16
    
    def test_12bit_value_range(self, jpeg_instance, sample_12bit_image):
        """Test that 12-bit values stay within 0-4095 range."""
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert np.all(decoded >= 0)
        assert np.all(decoded <= 4095)
    
    def test_16bit_value_range(self, jpeg_instance, sample_16bit_image, supports_16bit):
        """Test that 16-bit values stay within 0-65535 range."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert np.all(decoded >= 0)
        assert np.all(decoded <= 65535)
    
    def test_12bit_quality_levels(self, jpeg_instance, sample_12bit_image):
        """Test 12-bit encoding with different quality levels."""
        quality_levels = [50, 75, 85, 95, 100]
        sizes = []
        for quality in quality_levels:
            jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image, quality=quality)
            sizes.append(len(jpeg_buf))
            decoded = jpeg_instance.decode_12bit(jpeg_buf)
            assert decoded.shape == sample_12bit_image.shape
        
        # Higher quality should generally produce larger files
        # (though not strictly monotonic due to compression characteristics)
        assert sizes[-1] >= sizes[0]  # quality 100 >= quality 50
    
    def test_12bit_different_subsampling(self, jpeg_instance, sample_12bit_image):
        """Test 12-bit with different chroma subsampling."""
        subsamplings = [TJSAMP_444, TJSAMP_422, TJSAMP_420]
        for subsample in subsamplings:
            jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image, jpeg_subsample=subsample)
            decoded = jpeg_instance.decode_12bit(jpeg_buf)
            assert decoded.shape == sample_12bit_image.shape
            assert decoded.dtype == np.uint16
    
    def test_12bit_different_pixel_formats(self, jpeg_instance):
        """Test 12-bit with different pixel formats (BGR, GRAY)."""
        # BGR (default) should work
        img_bgr = np.random.randint(0, 4096, (50, 50, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img_bgr, pixel_format=TJPF_BGR)
        decoded = jpeg_instance.decode_12bit(jpeg_buf, pixel_format=TJPF_BGR)
        assert decoded.shape == img_bgr.shape
        
        # GRAY should work
        img_gray = np.random.randint(0, 4096, (50, 50, 1), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img_gray, pixel_format=TJPF_GRAY, jpeg_subsample=TJSAMP_GRAY)
        decoded = jpeg_instance.decode_12bit(jpeg_buf, pixel_format=TJPF_GRAY)
        assert decoded.shape == img_gray.shape
    
    def test_16bit_different_pixel_formats(self, jpeg_instance, supports_16bit):
        """Test 16-bit with different pixel formats (RGB, BGR, GRAY)."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        # RGB
        img_rgb = np.random.randint(0, 65536, (50, 50, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img_rgb, pixel_format=TJPF_RGB)
        decoded = jpeg_instance.decode_16bit(jpeg_buf, pixel_format=TJPF_RGB)
        assert decoded.shape == img_rgb.shape
        
        # BGR
        img_bgr = np.random.randint(0, 65536, (50, 50, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img_bgr, pixel_format=TJPF_BGR)
        decoded = jpeg_instance.decode_16bit(jpeg_buf, pixel_format=TJPF_BGR)
        assert decoded.shape == img_bgr.shape
        
        # GRAY
        img_gray = np.random.randint(0, 65536, (50, 50, 1), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img_gray, pixel_format=TJPF_GRAY)
        decoded = jpeg_instance.decode_16bit(jpeg_buf, pixel_format=TJPF_GRAY)
        assert decoded.shape == img_gray.shape
    
    def test_12bit_grayscale(self, jpeg_instance):
        """Test single-channel grayscale 12-bit images."""
        img_gray = np.random.randint(0, 4096, (100, 100, 1), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img_gray, pixel_format=TJPF_GRAY, jpeg_subsample=TJSAMP_GRAY)
        decoded = jpeg_instance.decode_12bit(jpeg_buf, pixel_format=TJPF_GRAY)
        assert decoded.shape == img_gray.shape
        assert decoded.dtype == np.uint16
    
    def test_12bit_with_flags(self, jpeg_instance, sample_12bit_image):
        """Test 12-bit with compression flags (PROGRESSIVE, FASTDCT)."""
        # Progressive
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image, flags=TJFLAG_PROGRESSIVE)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert decoded.shape == sample_12bit_image.shape
        
        # Fast DCT
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image, flags=TJFLAG_FASTDCT)
        decoded = jpeg_instance.decode_12bit(jpeg_buf, flags=TJFLAG_FASTDCT)
        assert decoded.shape == sample_12bit_image.shape
    
    def test_16bit_with_flags(self, jpeg_instance, sample_16bit_image, supports_16bit):
        """Reject flags whose semantics do not exist for lossless JPEG."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        for flags in (TJFLAG_PROGRESSIVE, TJFLAG_FASTDCT):
            with pytest.raises(ValueError, match='lossless JPEG'):
                jpeg_instance.encode_16bit(sample_16bit_image, flags=flags)
    
    def test_12bit_invalid_precision_parameter(self, jpeg_instance, sample_12bit_image):
        """Test error handling for wrong dtype in 12-bit methods."""
        # Test that encode_12bit requires uint16
        img_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match='encode_12bit\\(\\) requires uint16 array'):
            jpeg_instance.encode_12bit(img_uint8)
    
    def test_16bit_invalid_precision_parameter(self, jpeg_instance, sample_16bit_image, supports_16bit):
        """Test error handling for wrong dtype in 16-bit methods."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        # Test that encode_16bit requires uint16
        img_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match='encode_16bit\\(\\) requires uint16 array'):
            jpeg_instance.encode_16bit(img_uint8)
    
    def test_12bit_wrong_dtype_input(self, jpeg_instance):
        """Test error when uint8 is passed for 12-bit encoding."""
        img_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match='encode_12bit\\(\\) requires uint16 array'):
            jpeg_instance.encode_12bit(img_uint8)
    
    def test_16bit_wrong_dtype_input(self, jpeg_instance, supports_16bit):
        """Test error when uint8 is passed for 16-bit encoding."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        img_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match='encode_16bit\\(\\) requires uint16 array'):
            jpeg_instance.encode_16bit(img_uint8)
    
    def test_mixed_precision_encode_decode(self, jpeg_instance, sample_12bit_image, supports_16bit):
        """Test encoding and decoding at various precision levels."""
        # Encode as 12-bit
        jpeg_buf_12 = jpeg_instance.encode_12bit(sample_12bit_image)
        
        # Decode as 12-bit (should work)
        decoded_12bit = jpeg_instance.decode_12bit(jpeg_buf_12)
        assert decoded_12bit.dtype == np.uint16
        assert decoded_12bit.shape == sample_12bit_image.shape
        
        # Test 8-bit roundtrip for comparison
        img_8bit = (sample_12bit_image / 16).astype(np.uint8)  # Downscale to 8-bit range
        jpeg_buf_8 = jpeg_instance.encode(img_8bit)
        decoded_8bit = jpeg_instance.decode(jpeg_buf_8)
        assert decoded_8bit.dtype == np.uint8
        assert decoded_8bit.shape == img_8bit.shape
        
        # If 16-bit is supported, test it
        if supports_16bit:
            img_16bit = (sample_12bit_image * 16).astype(np.uint16)  # Upscale to 16-bit range
            jpeg_buf_16 = jpeg_instance.encode_16bit(img_16bit)
            decoded_16bit = jpeg_instance.decode_16bit(jpeg_buf_16)
            assert decoded_16bit.dtype == np.uint16
            assert decoded_16bit.shape == img_16bit.shape

    
    def test_12bit_memory_continuity(self, jpeg_instance):
        """Test multiple 12-bit encode/decode cycles (100 iterations)."""
        img = np.random.randint(0, 4096, (50, 50, 3), dtype=np.uint16)
        for _ in range(100):
            jpeg_buf = jpeg_instance.encode_12bit(img)
            decoded = jpeg_instance.decode_12bit(jpeg_buf)
            assert decoded.shape == img.shape
            assert decoded.dtype == np.uint16
    
    def test_16bit_memory_continuity(self, jpeg_instance, supports_16bit):
        """Test multiple 16-bit encode/decode cycles (100 iterations)."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        img = np.random.randint(0, 65536, (50, 50, 3), dtype=np.uint16)
        for _ in range(100):
            jpeg_buf = jpeg_instance.encode_16bit(img)
            decoded = jpeg_instance.decode_16bit(jpeg_buf)
            assert decoded.shape == img.shape
            assert decoded.dtype == np.uint16
    
    def test_12bit_edge_values(self, jpeg_instance):
        """Test 12-bit with min (0) and max (4095) values."""
        # All zeros
        img_min = np.zeros((50, 50, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img_min)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert decoded.shape == img_min.shape
        
        # All max values
        img_max = np.full((50, 50, 3), 4095, dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img_max)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert decoded.shape == img_max.shape
    
    def test_16bit_edge_values(self, jpeg_instance, supports_16bit):
        """Test 16-bit with min (0) and max (65535) values."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        # All zeros
        img_min = np.zeros((50, 50, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img_min)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.shape == img_min.shape
        
        # All max values
        img_max = np.full((50, 50, 3), 65535, dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img_max)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.shape == img_max.shape
    
    def test_convenience_methods_12bit(self, jpeg_instance, sample_12bit_image):
        """Test encode_12bit() and decode_12bit() convenience methods."""
        # Test encode_12bit
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image, quality=90)
        assert isinstance(jpeg_buf, bytes)
        
        # Test decode_12bit
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert decoded.shape == sample_12bit_image.shape
        assert decoded.dtype == np.uint16
    
    def test_convenience_methods_16bit(self, jpeg_instance, sample_16bit_image, supports_16bit):
        """Test encode_16bit() and decode_16bit() convenience methods."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        # Test encode_16bit
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        assert isinstance(jpeg_buf, bytes)
        
        # Test decode_16bit
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.shape == sample_16bit_image.shape
        assert decoded.dtype == np.uint16
    
    def test_12bit_decode_header(self, jpeg_instance, sample_12bit_image):
        """Test that decode_header works with 12-bit JPEGs and returns correct precision."""
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        # Test backward compatible mode
        width, height, subsample, colorspace = jpeg_instance.decode_header(jpeg_buf)
        assert width == sample_12bit_image.shape[1]
        assert height == sample_12bit_image.shape[0]
        # Test with precision
        width, height, subsample, colorspace, precision = jpeg_instance.decode_header(jpeg_buf, return_precision=True)
        assert precision == 12, "12-bit JPEG should report precision of 12"
    
    def test_16bit_decode_header(self, jpeg_instance, sample_16bit_image, supports_16bit):
        """Test that decode_header works with 16-bit JPEGs and returns correct precision."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        # Test backward compatible mode
        width, height, subsample, colorspace = jpeg_instance.decode_header(jpeg_buf)
        assert width == sample_16bit_image.shape[1]
        assert height == sample_16bit_image.shape[0]
        # Test with precision
        width, height, subsample, colorspace, precision = jpeg_instance.decode_header(jpeg_buf, return_precision=True)
        assert precision == 16, "16-bit JPEG should report precision of 16"
    
    def test_decode_header_precision_selection(self, jpeg_instance):
        """Test using decode_header precision to select appropriate decode function."""
        # Test 8-bit
        img_8bit = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        jpeg_8bit = jpeg_instance.encode(img_8bit)
        _, _, _, _, precision = jpeg_instance.decode_header(jpeg_8bit, return_precision=True)
        assert precision == 8
        if precision == 8:
            decoded = jpeg_instance.decode(jpeg_8bit)
            assert decoded.dtype == np.uint8
        
        # Test 12-bit
        img_12bit = np.random.randint(0, 4096, (50, 50, 3), dtype=np.uint16)
        jpeg_12bit = jpeg_instance.encode_12bit(img_12bit)
        _, _, _, _, precision = jpeg_instance.decode_header(jpeg_12bit, return_precision=True)
        assert precision == 12
        if precision == 12:
            decoded = jpeg_instance.decode_12bit(jpeg_12bit)
            assert decoded.dtype == np.uint16
        
        # Test 16-bit (if supported)
        try:
            # np.random.randint upper bound is exclusive, so use 65536 to get values 0-65535
            img_16bit = np.random.randint(0, 65536, (50, 50, 3), dtype=np.uint16)
            jpeg_16bit = jpeg_instance.encode_16bit(img_16bit)
            _, _, _, _, precision = jpeg_instance.decode_header(jpeg_16bit, return_precision=True)
            assert precision == 16
            if precision == 16:
                decoded = jpeg_instance.decode_16bit(jpeg_16bit)
                assert decoded.dtype == np.uint16
        except (OSError, IOError):
            # 16-bit not supported by this build
            pass


class TestLosslessJPEG:
    """Tests for lossless JPEG compression across all precision levels."""

    @pytest.mark.parametrize('method_name', [
        'decode',
        'decode_to_yuv',
        'decode_to_yuv_planes',
        'scale_with_quality',
    ])
    def test_8bit_lossless_rejects_non_identity_scaling(
            self, jpeg_instance, method_name):
        """All 8-bit decode paths must reject unsafe lossless scaling."""
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        jpeg_buf = jpeg_instance.encode(img, lossless=True)
        method = getattr(jpeg_instance, method_name)

        for scaling_factor in sorted(
                jpeg_instance.scaling_factors - {(1, 1)}):
            with pytest.raises(ValueError, match='lossless JPEG'):
                method(jpeg_buf, scaling_factor=scaling_factor)

    def test_8bit_lossless_accepts_identity_scaling(self, jpeg_instance):
        """Identity scaling is safe and preserves the full lossless image."""
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        jpeg_buf = jpeg_instance.encode(img, lossless=True)

        decoded = jpeg_instance.decode(jpeg_buf, scaling_factor=(1, 1))

        assert decoded.shape == img.shape
        assert np.array_equal(decoded, img)

    def test_12bit_lossless_rejects_non_identity_scaling(
            self, jpeg_instance):
        """12-bit lossless decode must reject scaling before allocation."""
        img = np.random.randint(0, 4096, (32, 32, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img, lossless=True)

        for scaling_factor in sorted(
                jpeg_instance.scaling_factors - {(1, 1)}):
            with pytest.raises(ValueError, match='lossless JPEG'):
                jpeg_instance.decode_12bit(
                    jpeg_buf, scaling_factor=scaling_factor)

    def test_12bit_lossless_accepts_identity_scaling(self, jpeg_instance):
        """12-bit identity scaling must preserve the full image."""
        img = np.random.randint(0, 4096, (32, 32, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img, lossless=True)

        decoded = jpeg_instance.decode_12bit(
            jpeg_buf, scaling_factor=(1, 1))

        assert np.array_equal(decoded, img)

    def test_12bit_lossy_scaling_remains_supported(self, jpeg_instance):
        """The lossless guard must not disable valid 12-bit lossy scaling."""
        img = np.random.randint(0, 4096, (32, 32, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img, lossless=False)

        decoded = jpeg_instance.decode_12bit(
            jpeg_buf, scaling_factor=(1, 2))

        assert decoded.shape == (16, 16, 3)

    def test_16bit_lossless_rejects_non_identity_scaling(
            self, jpeg_instance, supports_16bit):
        """16-bit lossless decode must reject scaling before allocation."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        img = np.random.randint(0, 65536, (32, 32, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img)

        for scaling_factor in sorted(
                jpeg_instance.scaling_factors - {(1, 1)}):
            with pytest.raises(ValueError, match='lossless JPEG'):
                jpeg_instance.decode_16bit(
                    jpeg_buf, scaling_factor=scaling_factor)

    def test_16bit_lossless_accepts_identity_scaling(
            self, jpeg_instance, supports_16bit):
        """16-bit identity scaling must preserve the full image."""
        if not supports_16bit:
            pytest.skip("16-bit precision not supported by this TurboJPEG build")
        img = np.random.randint(0, 65536, (32, 32, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img)

        decoded = jpeg_instance.decode_16bit(
            jpeg_buf, scaling_factor=(1, 1))

        assert np.array_equal(decoded, img)

    def test_8bit_lossless_roundtrip(self, jpeg_instance):
        """Test 8-bit lossless encoding provides perfect reconstruction."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        jpeg_buf = jpeg_instance.encode(img, lossless=True)
        decoded = jpeg_instance.decode(jpeg_buf)
        assert np.array_equal(img, decoded), "8-bit lossless should provide perfect reconstruction"
        
    def test_12bit_lossless_roundtrip(self, jpeg_instance):
        """Test 12-bit lossless encoding provides perfect reconstruction."""
        img = np.random.randint(0, 4096, (100, 100, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img, lossless=True)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert np.array_equal(img, decoded), "12-bit lossless should provide perfect reconstruction"
    
    def test_12bit_lossless_convenience_method(self, jpeg_instance):
        """Test encode_12bit with lossless=True provides perfect reconstruction."""
        img = np.random.randint(0, 4096, (100, 100, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img, lossless=True)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert np.array_equal(img, decoded), "encode_12bit with lossless should provide perfect reconstruction"
    
    def test_16bit_lossless_roundtrip(self, jpeg_instance):
        """Test 16-bit lossless encoding provides perfect reconstruction."""
        img = np.random.randint(0, 65536, (100, 100, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert np.array_equal(img, decoded), "16-bit lossless should provide perfect reconstruction"

    def test_8bit_lossless_larger_than_lossy(self, jpeg_instance):
        """Test that lossless encoding produces larger files than lossy."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        lossy_buf = jpeg_instance.encode(img, quality=95, lossless=False)
        lossless_buf = jpeg_instance.encode(img, lossless=True)
        assert len(lossless_buf) > len(lossy_buf), "Lossless should be larger than lossy"

    def test_8bit_lossless_vs_lossy_reconstruction(self, jpeg_instance):
        """Test that lossless provides perfect reconstruction while lossy does not."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Lossy encoding
        lossy_buf = jpeg_instance.encode(img, quality=95, lossless=False)
        jpeg_instance.decode(lossy_buf)
        
        # Lossless encoding
        lossless_buf = jpeg_instance.encode(img, lossless=True)
        lossless_decoded = jpeg_instance.decode(lossless_buf)
        
        # Lossless should be perfect, lossy should not (for most random images)
        assert np.array_equal(img, lossless_decoded), "Lossless should provide perfect reconstruction"
        # Note: lossy may occasionally match by chance with simple patterns, so we don't test inequality
    
    def test_12bit_lossless_larger_than_lossy(self, jpeg_instance):
        """Test that 12-bit lossless encoding produces larger files than lossy."""
        img = np.random.randint(0, 4096, (100, 100, 3), dtype=np.uint16)
        lossy_buf = jpeg_instance.encode_12bit(img, quality=95, lossless=False)
        lossless_buf = jpeg_instance.encode_12bit(img, lossless=True)
        assert len(lossless_buf) > len(lossy_buf), "Lossless should be larger than lossy"
    
    def test_12bit_lossless_vs_lossy_reconstruction(self, jpeg_instance):
        """Test that 12-bit lossless provides perfect reconstruction while lossy does not."""
        img = np.random.randint(0, 4096, (100, 100, 3), dtype=np.uint16)
        
        # Lossy encoding
        lossy_buf = jpeg_instance.encode_12bit(img, quality=95, lossless=False)
        jpeg_instance.decode_12bit(lossy_buf)
        
        # Lossless encoding
        lossless_buf = jpeg_instance.encode_12bit(img, lossless=True)
        lossless_decoded = jpeg_instance.decode_12bit(lossless_buf)
        
        # Lossless should be perfect, lossy should not (for most random images)
        assert np.array_equal(img, lossless_decoded), "Lossless should provide perfect reconstruction"
        # Note: lossy may occasionally match by chance with simple patterns, so we don't test inequality
    
    def test_12bit_lossless_edge_values(self, jpeg_instance):
        """Test 12-bit lossless with edge values (0 and 4095)."""
        img = np.zeros((50, 50, 3), dtype=np.uint16)
        img[0:25, :, :] = 0
        img[25:50, :, :] = 4095
        jpeg_buf = jpeg_instance.encode_12bit(img, lossless=True)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert np.array_equal(img, decoded), "12-bit lossless should preserve edge values"
    
    def test_16bit_lossless_edge_values(self, jpeg_instance):
        """Test 16-bit lossless with edge values (0 and 65535)."""
        img = np.zeros((50, 50, 3), dtype=np.uint16)
        img[0:25, :, :] = 0
        img[25:50, :, :] = 65535
        jpeg_buf = jpeg_instance.encode_16bit(img)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert np.array_equal(img, decoded), "16-bit lossless should preserve edge values"
    
    def test_lossless_different_pixel_formats(self, jpeg_instance):
        """Test lossless encoding with different pixel formats."""
        from turbojpeg import TJPF_RGB, TJPF_GRAY
        
        # RGB format
        img_rgb = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        jpeg_rgb = jpeg_instance.encode(img_rgb, pixel_format=TJPF_RGB, lossless=True)
        decoded_rgb = jpeg_instance.decode(jpeg_rgb, pixel_format=TJPF_RGB)
        assert np.array_equal(img_rgb, decoded_rgb), "Lossless RGB should be perfect"
        
        # Grayscale format
        img_gray = np.random.randint(0, 256, (50, 50, 1), dtype=np.uint8)
        jpeg_gray = jpeg_instance.encode(img_gray, pixel_format=TJPF_GRAY, lossless=True)
        decoded_gray = jpeg_instance.decode(jpeg_gray, pixel_format=TJPF_GRAY)
        assert np.array_equal(img_gray, decoded_gray), "Lossless grayscale should be perfect"


class TestYUVMetadataAndPadding:
    """Regression coverage for PTJ-005."""

    @pytest.mark.parametrize('subsample', [
        TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY,
        TJSAMP_440, TJSAMP_411, TJSAMP_441,
    ])
    def test_unified_plane_sizes_match_native_plane_arrays(
            self, jpeg_instance, subsample):
        image = np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3)
        encoded = jpeg_instance.encode(image, jpeg_subsample=subsample)

        _, plane_sizes = jpeg_instance.decode_to_yuv(encoded, pad=8)
        unified, metadata = jpeg_instance.decode_to_yuv(
            encoded, pad=8, return_metadata=True)
        planes = jpeg_instance.decode_to_yuv_planes(encoded)

        assert plane_sizes == [plane.shape for plane in planes]
        assert [
            (plane.height, plane.width) for plane in metadata
        ] == plane_sizes
        assert metadata[-1].offset + \
            metadata[-1].stride * metadata[-1].height == unified.size

    def test_unified_yuv_padding_is_zero_initialized(
            self, jpeg_instance, monkeypatch):
        image = np.arange(17 * 17 * 3, dtype=np.uint8).reshape(17, 17, 3)
        encoded = jpeg_instance.encode(
            image, jpeg_subsample=TJSAMP_420)
        original_empty = np.empty

        def dirty_empty(shape, *args, **kwargs):
            result = original_empty(shape, *args, **kwargs)
            result.fill(0xA5)
            return result

        monkeypatch.setattr('turbojpeg.np.empty', dirty_empty)
        unified, plane_sizes = jpeg_instance.decode_to_yuv(encoded, pad=8)

        offset = 0
        for height, width in plane_sizes:
            stride = (width + 7) & ~7
            plane = unified[offset:offset + height * stride].reshape(
                height, stride)
            assert np.all(plane[:, width:] == 0)
            offset += height * stride
        assert offset == unified.size

    def test_unified_yuv_can_return_explicit_layout_metadata(
            self, jpeg_instance):
        image = np.arange(17 * 17 * 3, dtype=np.uint8).reshape(17, 17, 3)
        encoded = jpeg_instance.encode(
            image, jpeg_subsample=TJSAMP_420)

        unified, metadata = jpeg_instance.decode_to_yuv(
            encoded, pad=8, return_metadata=True)

        assert metadata == [
            YUVPlaneInfo(offset=0, stride=24, width=18, height=18),
            YUVPlaneInfo(offset=432, stride=16, width=9, height=9),
            YUVPlaneInfo(offset=576, stride=16, width=9, height=9),
        ]
        assert metadata[-1].offset + \
            metadata[-1].stride * metadata[-1].height == unified.size

    def test_separate_yuv_plane_padding_is_zero_initialized(
            self, jpeg_instance, monkeypatch):
        image = np.arange(17 * 17 * 3, dtype=np.uint8).reshape(17, 17, 3)
        encoded = jpeg_instance.encode(
            image, jpeg_subsample=TJSAMP_420)
        tight_planes = jpeg_instance.decode_to_yuv_planes(encoded)
        strides = tuple(plane.shape[1] + 7 for plane in tight_planes)
        original_empty = np.empty

        def dirty_empty(shape, *args, **kwargs):
            result = original_empty(shape, *args, **kwargs)
            result.fill(0xA5)
            return result

        monkeypatch.setattr('turbojpeg.np.empty', dirty_empty)
        padded_planes = jpeg_instance.decode_to_yuv_planes(
            encoded, strides=strides)

        for tight, padded in zip(tight_planes, padded_planes):
            assert padded.shape[1] == tight.shape[1] + 7
            assert np.all(padded[:, tight.shape[1]:] == 0)

    @pytest.mark.parametrize('pad', [0, -1, 3, 1.5])
    def test_decode_to_yuv_rejects_invalid_padding(
            self, jpeg_instance, encoded_sample_jpeg, pad):
        with pytest.raises((TypeError, ValueError)):
            jpeg_instance.decode_to_yuv(encoded_sample_jpeg, pad=pad)

    def test_decode_to_yuv_planes_rejects_short_stride(
            self, jpeg_instance, encoded_sample_jpeg):
        with pytest.raises(ValueError, match='must be at least'):
            jpeg_instance.decode_to_yuv_planes(
                encoded_sample_jpeg, strides=(1, 1, 1))


class TestFlagSemantics:
    """Regression coverage for PTJ-006."""

    def test_bottomup_encode_reverses_source_rows(self, jpeg_instance):
        image = np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3)

        bottom_up = jpeg_instance.encode(
            image, quality=95, jpeg_subsample=TJSAMP_444,
            flags=TJFLAG_BOTTOMUP)
        explicitly_reversed = jpeg_instance.encode(
            image[::-1], quality=95, jpeg_subsample=TJSAMP_444)

        assert bottom_up == explicitly_reversed

    def test_bottomup_decode_reverses_destination_rows(
            self, jpeg_instance, encoded_sample_jpeg):
        normal = jpeg_instance.decode(encoded_sample_jpeg)
        bottom_up = jpeg_instance.decode(
            encoded_sample_jpeg, flags=TJFLAG_BOTTOMUP)

        assert np.array_equal(bottom_up, normal[::-1])

    def test_yuv_and_scale_progressive_flags_have_real_effect(
            self, jpeg_instance, encoded_sample_jpeg):
        width, height, subsample, _ = jpeg_instance.decode_header(
            encoded_sample_jpeg)
        yuv, _ = jpeg_instance.decode_to_yuv(encoded_sample_jpeg)

        from_yuv = jpeg_instance.encode_from_yuv(
            yuv, height, width, jpeg_subsample=subsample,
            flags=TJFLAG_PROGRESSIVE)
        scaled = jpeg_instance.scale_with_quality(
            encoded_sample_jpeg, flags=TJFLAG_PROGRESSIVE)

        assert b'\xff\xc2' in from_yuv
        assert b'\xff\xc2' in scaled

    def test_decompress_flags_are_forwarded_and_checked(
            self, jpeg_instance, encoded_sample_jpeg, monkeypatch):
        native_set = jpeg_instance._TurboJPEG__set
        calls = []

        def recording_set(handle, parameter, value):
            calls.append((parameter, value))
            return native_set(handle, parameter, value)

        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__set', recording_set)
        jpeg_instance.decode(
            encoded_sample_jpeg,
            flags=(
                TJFLAG_BOTTOMUP | TJFLAG_ACCURATEDCT |
                TJFLAG_STOPONWARNING
            ),
        )

        assert (TJPARAM_BOTTOMUP, 1) in calls
        assert (TJPARAM_FASTDCT, 0) in calls
        assert (TJPARAM_STOPONWARNING, 1) in calls

    def test_native_flag_set_failure_is_not_silently_ignored(
            self, jpeg_instance, sample_bgr_image, monkeypatch):
        native_set = jpeg_instance._TurboJPEG__set

        def failing_set(handle, parameter, value):
            if parameter == TJPARAM_BOTTOMUP:
                return -1
            return native_set(handle, parameter, value)

        def raise_native_error(handle):
            raise OSError('injected tj3Set failure')

        monkeypatch.setattr(jpeg_instance, '_TurboJPEG__set', failing_set)
        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__report_error', raise_native_error)

        with pytest.raises(OSError, match='injected tj3Set failure'):
            jpeg_instance.encode(
                sample_bgr_image, flags=TJFLAG_BOTTOMUP)

    def test_operations_reject_flags_they_cannot_support(
            self, jpeg_instance, encoded_sample_jpeg):
        yuv, _ = jpeg_instance.decode_to_yuv(encoded_sample_jpeg)
        width, height, subsample, _ = jpeg_instance.decode_header(
            encoded_sample_jpeg)
        calls = [
            lambda: jpeg_instance.decode_to_yuv(
                encoded_sample_jpeg, flags=TJFLAG_BOTTOMUP),
            lambda: jpeg_instance.decode_to_yuv_planes(
                encoded_sample_jpeg, flags=TJFLAG_FASTUPSAMPLE),
            lambda: jpeg_instance.encode_from_yuv(
                yuv, height, width, jpeg_subsample=subsample,
                flags=TJFLAG_BOTTOMUP),
            lambda: jpeg_instance.scale_with_quality(
                encoded_sample_jpeg, flags=TJFLAG_FASTUPSAMPLE),
        ]

        for call in calls:
            with pytest.raises(ValueError, match='Unsupported flags'):
                call()

    def test_conflicting_and_unknown_flags_are_rejected(
            self, jpeg_instance, sample_bgr_image):
        with pytest.raises(ValueError, match='mutually exclusive'):
            jpeg_instance.encode(
                sample_bgr_image,
                flags=TJFLAG_FASTDCT | TJFLAG_ACCURATEDCT)
        with pytest.raises(ValueError, match='Unsupported flags'):
            jpeg_instance.encode(sample_bgr_image, flags=1 << 20)


class TestPublicInputValidation:
    """Regression coverage for PTJ-007."""

    @pytest.mark.parametrize('pixel_format', [-1, 12, True, 1.5])
    def test_invalid_pixel_formats_raise_public_errors(
            self, jpeg_instance, sample_bgr_image, encoded_sample_jpeg,
            pixel_format):
        with pytest.raises((TypeError, ValueError)):
            jpeg_instance.encode(
                sample_bgr_image, pixel_format=pixel_format)
        with pytest.raises((TypeError, ValueError)):
            jpeg_instance.decode(
                encoded_sample_jpeg, pixel_format=pixel_format)

    @pytest.mark.parametrize('shape', [
        (8, 8, 2),
        (8, 8, 1, 1),
    ])
    def test_grayscale_rejects_ambiguous_shapes(
            self, jpeg_instance, shape):
        image8 = np.zeros(shape, dtype=np.uint8)
        image12 = np.zeros(shape, dtype=np.uint16)

        with pytest.raises(ValueError, match='Invalid shape'):
            jpeg_instance.encode(image8, pixel_format=TJPF_GRAY)
        with pytest.raises(ValueError, match='Invalid shape'):
            jpeg_instance.encode_12bit(image12, pixel_format=TJPF_GRAY)
        with pytest.raises(ValueError, match='Invalid shape'):
            jpeg_instance.encode_16bit(image12, pixel_format=TJPF_GRAY)

    @pytest.mark.parametrize('shape', [(8, 8), (8, 8, 1)])
    def test_grayscale_accepts_only_documented_shapes(
            self, jpeg_instance, shape):
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        if len(shape) == 3:
            image = image[:, :, None]

        encoded = jpeg_instance.encode(
            image, pixel_format=TJPF_GRAY,
            jpeg_subsample=TJSAMP_GRAY)
        decoded = jpeg_instance.decode(encoded, pixel_format=TJPF_GRAY)

        assert decoded.shape == (8, 8, 1)

    @pytest.mark.parametrize('lossless', [False, True])
    @pytest.mark.parametrize('value', [4096, 65535])
    def test_12bit_rejects_samples_outside_12bit_range(
            self, jpeg_instance, lossless, value):
        image = np.full((8, 8, 3), value, dtype=np.uint16)

        with pytest.raises(ValueError, match='between 0 and 4095'):
            jpeg_instance.encode_12bit(image, lossless=lossless)

    @pytest.mark.parametrize('operation', ['encode', 'encode_12bit', 'encode_16bit'])
    def test_encoders_reject_empty_dimensions(
            self, jpeg_instance, operation):
        dtype = np.uint8 if operation == 'encode' else np.uint16
        image = np.empty((0, 8, 3), dtype=dtype)

        with pytest.raises(ValueError, match='height'):
            getattr(jpeg_instance, operation)(image)

    @pytest.mark.parametrize('quality', [0, 101, -1, 1.5, True])
    def test_lossy_encoders_validate_quality(
            self, jpeg_instance, sample_bgr_image, quality):
        with pytest.raises((TypeError, ValueError)):
            jpeg_instance.encode(sample_bgr_image, quality=quality)
        image12 = sample_bgr_image.astype(np.uint16) * 16
        with pytest.raises((TypeError, ValueError)):
            jpeg_instance.encode_12bit(image12, quality=quality)

    @pytest.mark.parametrize('subsample', [-1, 7, True, 1.5])
    def test_encoders_validate_subsampling(
            self, jpeg_instance, sample_bgr_image, subsample):
        with pytest.raises((TypeError, ValueError)):
            jpeg_instance.encode(
                sample_bgr_image, jpeg_subsample=subsample)

    @pytest.mark.parametrize(('argument', 'value'), [
        ('quality', 0),
        ('jpeg_subsample', 999),
    ])
    def test_lossless_encoders_ignore_lossy_parameters(
            self, jpeg_instance, sample_bgr_image, argument, value):
        kwargs = {argument: value, 'lossless': True}

        encoded8 = jpeg_instance.encode(sample_bgr_image, **kwargs)
        decoded8 = jpeg_instance.decode(encoded8)
        assert np.array_equal(decoded8, sample_bgr_image)

        image12 = sample_bgr_image.astype(np.uint16) * 16
        encoded12 = jpeg_instance.encode_12bit(image12, **kwargs)
        decoded12 = jpeg_instance.decode_12bit(encoded12)
        assert np.array_equal(decoded12, image12)


class TestCropAndTransformRegressions:
    """Regression coverage for PTJ-008 through PTJ-011."""

    def test_441_mcu_dimensions_and_full_crop(self, jpeg_instance):
        assert tjMCUWidth[TJSAMP_441] == 8
        assert tjMCUHeight[TJSAMP_441] == 32
        image = np.arange(65 * 65 * 3, dtype=np.uint8).reshape(65, 65, 3)
        encoded = jpeg_instance.encode(
            image, jpeg_subsample=TJSAMP_441)

        cropped = jpeg_instance.crop(encoded, 0, 0, 65, 65)

        assert jpeg_instance.decode_header(cropped)[:2] == (65, 65)

    def test_crop_retains_partial_right_and_bottom_mcus(self, jpeg_instance):
        image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        encoded = jpeg_instance.encode(
            image, jpeg_subsample=TJSAMP_422)

        cropped = jpeg_instance.crop(encoded, 0, 0, 100, 100)

        assert jpeg_instance.decode_header(cropped)[:2] == (100, 100)

    def test_crop_preserve_mode_has_explicit_alignment_semantics(
            self, jpeg_instance):
        image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        encoded = jpeg_instance.encode(
            image, jpeg_subsample=TJSAMP_422)

        expanded = jpeg_instance.crop(
            encoded, 10, 0, 50, 32, preserve=False)
        contained = jpeg_instance.crop(
            encoded, 10, 0, 50, 32, preserve=True)

        assert jpeg_instance.decode_header(expanded)[:2] == (60, 32)
        assert jpeg_instance.decode_header(contained)[:2] == (44, 32)

    def test_crop_multiple_zero_size_uses_native_remainder_semantics(
            self, jpeg_instance):
        image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        encoded = jpeg_instance.encode(
            image, jpeg_subsample=TJSAMP_444)

        full, remainder = jpeg_instance.crop_multiple(
            encoded, [(0, 0, 0, 0), (8, 8, 0, 0)])

        assert jpeg_instance.decode_header(full)[:2] == (32, 24)
        assert jpeg_instance.decode_header(remainder)[:2] == (24, 16)

    def test_dqt_parser_handles_unsigned_and_multiple_tables(
            self, jpeg_instance):
        table_one = bytes([0x01]) + bytes([17]) * 64
        table_zero = bytes([0x10]) + b''.join(
            struct.pack('>H', 40000) for _ in range(64))
        payload = table_one + table_zero
        app_payload = b'metadata\xff\xdb\x00\x03\x00'
        jpeg_data = (
            b'\xff\xd8\xff\xe1' +
            struct.pack('>H', len(app_payload) + 2) + app_payload +
            b'\xff\xdb' + struct.pack('>H', len(payload) + 2) +
            payload + b'\xff\xd9'
        )

        assert jpeg_instance._TurboJPEG__get_dc_dqt_element(
            jpeg_data, 1) == 17
        assert jpeg_instance._TurboJPEG__get_dc_dqt_element(
            jpeg_data, 0) == 40000

    def test_low_quality_dqt_and_background_luminance_are_not_inverted(
            self, jpeg_instance):
        image = np.full((16, 16, 3), 128, dtype=np.uint8)
        encoded = jpeg_instance.encode(
            image, quality=5, jpeg_subsample=TJSAMP_444)

        assert jpeg_instance._TurboJPEG__get_dc_dqt_element(
            encoded, 0) == 160
        black, white = jpeg_instance.crop_multiple(
            encoded,
            [(0, 0, 32, 32), (0, 0, 32, 32)],
            background_luminance=0.0,
        )[0], jpeg_instance.crop_multiple(
            encoded,
            [(0, 0, 32, 32)],
            background_luminance=1.0,
        )[0]
        black_pixels = jpeg_instance.decode(
            black, pixel_format=TJPF_GRAY)[20:, 20:, 0]
        white_pixels = jpeg_instance.decode(
            white, pixel_format=TJPF_GRAY)[20:, 20:, 0]

        assert black_pixels.mean() < 20
        assert white_pixels.mean() > 235

    def test_background_uses_first_component_quantization_table(
            self, jpeg_instance):
        image = np.full((16, 16, 1), 128, dtype=np.uint8)
        encoded = bytearray(jpeg_instance.encode(
            image, quality=50, pixel_format=TJPF_GRAY,
            jpeg_subsample=TJSAMP_GRAY))

        dqt_marker = encoded.index(b'\xff\xdb')
        dqt_info = dqt_marker + 4
        assert encoded[dqt_info] & 0x0F == 0
        encoded[dqt_info] = (encoded[dqt_info] & 0xF0) | 1

        sof_marker = encoded.index(b'\xff\xc0')
        first_component_dqt = sof_marker + 12
        assert encoded[first_component_dqt] == 0
        encoded[first_component_dqt] = 1

        remapped = bytes(encoded)
        assert jpeg_instance.decode(remapped).shape == (16, 16, 3)
        extended = jpeg_instance.crop_multiple(
            remapped, [(0, 0, 32, 32)],
            background_luminance=1.0)[0]
        white_pixels = jpeg_instance.decode(
            extended, pixel_format=TJPF_GRAY)[20:, 20:, 0]

        assert white_pixels.mean() > 235

    def test_12bit_background_luminance_uses_source_precision(
            self, jpeg_instance):
        image = np.full((16, 16, 3), 2048, dtype=np.uint16)
        encoded = jpeg_instance.encode_12bit(
            image, quality=50, jpeg_subsample=TJSAMP_444)

        extended = jpeg_instance.crop_multiple(
            encoded, [(0, 0, 32, 32)],
            background_luminance=1.0)[0]
        white_pixels = jpeg_instance.decode_12bit(
            extended, pixel_format=TJPF_GRAY)[20:, 20:, 0]

        assert white_pixels.mean() > 4000

    def test_dqt_parser_handles_every_lossy_quality(self, jpeg_instance):
        image = np.full((8, 8, 3), 127, dtype=np.uint8)

        for quality in range(1, 101):
            encoded = jpeg_instance.encode(
                image, quality=quality, jpeg_subsample=TJSAMP_444)
            coefficient = jpeg_instance._TurboJPEG__get_dc_dqt_element(
                encoded, 0)
            assert coefficient > 0

    def test_transform_callback_uses_native_return_convention(self):
        assert fill_background(
            None, CroppingRegion(), CroppingRegion(), 1, 0, None) == 0
        assert fill_background(
            None,
            CroppingRegion(0, 0, 8, 8),
            CroppingRegion(0, 0, 8, 8),
            0,
            0,
            None,
        ) == -1


class TestScaleWithQualityResources:
    """Regression coverage for PTJ-012."""

    def test_scale_uses_uint8_yuv_and_separate_handle_lifetimes(
            self, jpeg_instance, encoded_sample_jpeg, monkeypatch):
        native_zeros = np.zeros
        native_init = jpeg_instance._TurboJPEG__init
        native_destroy = jpeg_instance._TurboJPEG__destroy
        allocations = []
        lifecycle = []

        def recording_zeros(shape, *args, **kwargs):
            result = native_zeros(shape, *args, **kwargs)
            allocations.append(result)
            return result

        def recording_init(kind):
            lifecycle.append(('init', kind))
            return native_init(kind)

        def recording_destroy(handle):
            lifecycle.append(('destroy', None))
            native_destroy(handle)

        monkeypatch.setattr('turbojpeg.np.zeros', recording_zeros)
        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__init', recording_init)
        monkeypatch.setattr(
            jpeg_instance, '_TurboJPEG__destroy', recording_destroy)

        result = jpeg_instance.scale_with_quality(encoded_sample_jpeg)

        assert result.startswith(b'\xff\xd8')
        assert len(allocations) == 1
        assert allocations[0].dtype == np.uint8
        assert allocations[0].nbytes == allocations[0].size
        assert lifecycle == [
            ('init', TJINIT_DECOMPRESS),
            ('destroy', None),
            ('init', TJINIT_COMPRESS),
            ('destroy', None),
        ]


def _make_minimal_srgb_icc():
    """Build a minimal but structurally valid ICC profile for sRGB."""
    profile = bytearray(132)
    struct.pack_into('>I', profile, 0, 132)
    struct.pack_into('>I', profile, 8, 0x02100000)
    profile[12:16] = b'mntr'
    profile[16:20] = b'RGB '
    profile[20:24] = b'XYZ '
    profile[36:40] = b'acsp'
    return bytes(profile)

MINIMAL_SRGB_ICC = _make_minimal_srgb_icc()


class TestICCProfile:
    """Test ICC profile embed/extract functionality (requires TurboJPEG 3.1+)."""

    @pytest.fixture(autouse=True)
    def require_icc_support(self, jpeg_instance):
        """Skip the test if the loaded libturbojpeg does not support ICC profiles."""
        try:
            jpeg_instance.get_icc_profile(b'\xff\xd8\xff\xd9')
        except NotImplementedError as e:
            pytest.skip(str(e))
        except (OSError, Exception):
            pass  # Header parse error is fine — the function exists

    def test_get_icc_profile_with_embedded_profile(self, jpeg_instance, sample_bgr_image):
        """Test that get_icc_profile returns non-empty bytes for a JPEG with ICC profile."""
        jpeg_with_icc = jpeg_instance.encode(
            sample_bgr_image, quality=85, icc_profile=MINIMAL_SRGB_ICC)
        icc = jpeg_instance.get_icc_profile(jpeg_with_icc)
        assert icc is not None, "Expected ICC profile but got None"
        assert isinstance(icc, bytes), "ICC profile should be bytes"
        assert len(icc) > 0, "ICC profile should be non-empty"
        assert icc == MINIMAL_SRGB_ICC, "Extracted ICC profile should match embedded profile"

    def test_get_icc_profile_without_profile(self, jpeg_instance, sample_bgr_image):
        """Test that get_icc_profile returns None for a JPEG without ICC profile."""
        jpeg_without_icc = jpeg_instance.encode(sample_bgr_image, quality=85)
        icc = jpeg_instance.get_icc_profile(jpeg_without_icc)
        assert icc is None, \
            "Expected None for JPEG without ICC profile"

    def test_icc_profile_roundtrip(self, jpeg_instance, sample_bgr_image):
        """Test ICC profile survives a full encode→decode_header roundtrip."""
        jpeg_data = jpeg_instance.encode(
            sample_bgr_image, quality=95,
            jpeg_subsample=TJSAMP_444,
            icc_profile=MINIMAL_SRGB_ICC)
        assert jpeg_data is not None
        assert len(jpeg_data) > 0
        extracted_icc = jpeg_instance.get_icc_profile(jpeg_data)
        assert extracted_icc is not None, "ICC profile missing after roundtrip"
        assert extracted_icc == MINIMAL_SRGB_ICC, \
            f"ICC profile mismatch: expected {len(MINIMAL_SRGB_ICC)} bytes, got {len(extracted_icc) if extracted_icc else 0}"

    def test_icc_profile_in_place_encode_does_not_reallocate(
            self, jpeg_instance, sample_bgr_image):
        """ICC bytes are included when validating an in-place destination."""
        buffer_size = (
            jpeg_instance.buffer_size(sample_bgr_image) +
            len(MINIMAL_SRGB_ICC)
        )
        dst = bytearray(buffer_size)

        result, n_bytes = jpeg_instance.encode(
            sample_bgr_image,
            dst=dst,
            icc_profile=MINIMAL_SRGB_ICC,
        )

        assert result is dst
        jpeg_data = bytes(dst[:n_bytes])
        assert jpeg_instance.get_icc_profile(jpeg_data) == MINIMAL_SRGB_ICC

    def test_icc_profile_rejects_destination_without_profile_capacity(
            self, jpeg_instance, sample_bgr_image):
        """A normal JPEG buffer is too small once an ICC profile is attached."""
        dst = bytearray(jpeg_instance.buffer_size(sample_bgr_image))
        before = bytes(dst)

        with pytest.raises(ValueError, match='buffer is too small'):
            jpeg_instance.encode(
                sample_bgr_image,
                dst=dst,
                icc_profile=MINIMAL_SRGB_ICC,
            )

        assert bytes(dst) == before


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

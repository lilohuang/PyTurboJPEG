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
import tempfile
from io import BytesIO
from unittest.mock import patch, MagicMock
from ctypes.util import find_library

from turbojpeg import (
    TurboJPEG,
    TJPF_RGB, TJPF_BGR, TJPF_GRAY, TJPF_RGBA, TJPF_BGRA, TJPF_RGBX, TJPF_BGRX,
    TJPF_XBGR, TJPF_XRGB, TJPF_ABGR, TJPF_ARGB,
    TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY, TJSAMP_440, TJSAMP_411,
    TJCS_RGB, TJCS_YCbCr, TJCS_GRAY,
    TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT,
    TJPRECISION_8, TJPRECISION_12, TJPRECISION_16
)



# Test fixtures
@pytest.fixture(scope='module')
def jpeg_instance():
    """Create a TurboJPEG instance for testing."""
    return TurboJPEG()


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
        width, height, subsample, colorspace = jpeg_instance.decode_header(encoded_sample_jpeg)
        assert width == 100
        assert height == 100
        assert subsample in [TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY]
        assert colorspace in [TJCS_RGB, TJCS_YCbCr, TJCS_GRAY]
    
    def test_decode_header_invalid_data(self, jpeg_instance):
        """Test decode_header with invalid JPEG data raises error."""
        with pytest.raises(OSError):
            jpeg_instance.decode_header(b'not a jpeg')
    
    def test_decode_header_empty_data(self, jpeg_instance):
        """Test decode_header with empty data raises error."""
        with pytest.raises(OSError):
            jpeg_instance.decode_header(b'')


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
        """Test that PyTurboJPEG 2.0 accepts TurboJPEG 3.x library."""
        from unittest.mock import Mock, patch, MagicMock
        from ctypes import CDLL, c_int, c_void_p, c_size_t, c_ubyte, POINTER
        
        # Create a mock library that simulates TurboJPEG 3.x (has tj3Init)
        mock_new_lib = Mock(spec=CDLL)
        
        # Add all required TurboJPEG 3.x functions
        for func_name in ['tj3Init', 'tj3Destroy', 'tj3Set', 'tj3Get', 
                          'tj3SetScalingFactor', 'tj3JPEGBufSize', 'tj3YUVBufSize',
                          'tj3YUVPlaneWidth', 'tj3YUVPlaneHeight', 'tj3DecompressHeader',
                          'tj3Decompress8', 'tj3DecompressToYUV8', 'tj3DecompressToYUVPlanes8',
                          'tj3Compress8', 'tj3CompressFromYUV8', 'tj3Transform',
                          'tj3Free', 'tj3Alloc', 'tj3GetErrorStr', 'tj3GetErrorCode',
                          'tjGetScalingFactors']:
            setattr(mock_new_lib, func_name, Mock())
        
        # Mock tjGetScalingFactors to return proper structure
        mock_scaling_factors = MagicMock()
        mock_scaling_factors.__getitem__ = Mock(side_effect=lambda i: Mock(num=1, denom=1))
        mock_new_lib.tjGetScalingFactors.return_value = mock_scaling_factors
        
        # Mock c_int to return a value object for scaling factors count
        mock_c_int_instance = Mock()
        mock_c_int_instance.value = 1
        
        # Patch cdll.LoadLibrary to return our mock new library
        with patch('turbojpeg.cdll.LoadLibrary', return_value=mock_new_lib):
            with patch('turbojpeg.byref', return_value=Mock()):
                with patch('turbojpeg.c_int', return_value=mock_c_int_instance):
                    # This should NOT raise a RuntimeError about version
                    try:
                        tj = TurboJPEG(lib_path='/fake/path/libturbojpeg.so.0')
                        assert tj is not None
                    except RuntimeError as e:
                        if 'PyTurboJPEG 2.0 requires libjpeg-turbo 3.0' in str(e):
                            pytest.fail(f"Should not reject TurboJPEG 3.x library: {e}")
                        # Other RuntimeErrors are acceptable (e.g., from mock setup)
                    except Exception as e:
                        # Other exceptions from mock setup are acceptable, as long as it's not the version error
                        if 'PyTurboJPEG 2.0 requires libjpeg-turbo 3.0' in str(e):
                            pytest.fail(f"Should not reject TurboJPEG 3.x library: {e}")


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
        """Test cropping the full image (accounts for MCU alignment)."""
        # Get original dimensions
        orig_width, orig_height, _, _ = jpeg_instance.decode_header(test_crop_image)
        
        # Crop entire image (may be adjusted to MCU boundaries)
        cropped = jpeg_instance.crop(test_crop_image, 0, 0, orig_width, orig_height)
        
        # Verify dimensions are close (within MCU block size)
        width, height, _, _ = jpeg_instance.decode_header(cropped)
        # MCU blocks are typically 8x8, 16x8, 16x16, or 32x8
        # Allow for MCU adjustment (up to 16 pixels difference)
        assert abs(width - orig_width) <= 16
        assert abs(height - orig_height) <= 16
        # Should still be substantial portion of original
        assert width >= orig_width - 16
        assert height >= orig_height - 16
    
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
        
        # Crop from right edge (MCU-aligned)
        right_edge_x = orig_width - 48
        cropped_right = jpeg_instance.crop(test_crop_image, right_edge_x, 0, 48, 48)
        width, height, _, _ = jpeg_instance.decode_header(cropped_right)
        assert width == 48
        assert height == 48
        
        # Crop from bottom edge (MCU-aligned)
        bottom_edge_y = orig_height - 48
        cropped_bottom = jpeg_instance.crop(test_crop_image, 0, bottom_edge_y, 48, 48)
        width, height, _, _ = jpeg_instance.decode_header(cropped_bottom)
        assert width == 48
        assert height == 48
    
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
    
    def test_encode_decode_16bit_basic(self, jpeg_instance, sample_16bit_image):
        """Test basic 16-bit encode/decode roundtrip."""
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
    
    def test_16bit_image_shape_preservation(self, jpeg_instance, sample_16bit_image):
        """Test that 16-bit image dimensions are preserved."""
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.shape == sample_16bit_image.shape
    
    def test_12bit_dtype_verification(self, jpeg_instance, sample_12bit_image):
        """Test that 12-bit decode returns uint16."""
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert decoded.dtype == np.uint16
    
    def test_16bit_dtype_verification(self, jpeg_instance, sample_16bit_image):
        """Test that 16-bit decode returns uint16."""
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.dtype == np.uint16
    
    def test_12bit_value_range(self, jpeg_instance, sample_12bit_image):
        """Test that 12-bit values stay within 0-4095 range."""
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        decoded = jpeg_instance.decode_12bit(jpeg_buf)
        assert np.all(decoded >= 0)
        assert np.all(decoded <= 4095)
    
    def test_16bit_value_range(self, jpeg_instance, sample_16bit_image):
        """Test that 16-bit values stay within 0-65535 range."""
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
    
    def test_16bit_quality_levels(self, jpeg_instance, sample_16bit_image):
        """Test 16-bit encoding with different quality levels."""
        quality_levels = [50, 75, 85, 95, 100]
        sizes = []
        for quality in quality_levels:
            jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image, quality=quality)
            sizes.append(len(jpeg_buf))
            decoded = jpeg_instance.decode_16bit(jpeg_buf)
            assert decoded.shape == sample_16bit_image.shape
        
        # Higher quality should generally produce larger files
        assert sizes[-1] >= sizes[0]  # quality 100 >= quality 50
    
    def test_12bit_different_subsampling(self, jpeg_instance, sample_12bit_image):
        """Test 12-bit with different chroma subsampling."""
        subsamplings = [TJSAMP_444, TJSAMP_422, TJSAMP_420]
        for subsample in subsamplings:
            jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image, jpeg_subsample=subsample)
            decoded = jpeg_instance.decode_12bit(jpeg_buf)
            assert decoded.shape == sample_12bit_image.shape
            assert decoded.dtype == np.uint16
    
    def test_16bit_different_subsampling(self, jpeg_instance, sample_16bit_image):
        """Test 16-bit with different chroma subsampling."""
        subsamplings = [TJSAMP_444, TJSAMP_422, TJSAMP_420]
        for subsample in subsamplings:
            jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image, jpeg_subsample=subsample)
            decoded = jpeg_instance.decode_16bit(jpeg_buf)
            assert decoded.shape == sample_16bit_image.shape
            assert decoded.dtype == np.uint16
    
    def test_12bit_different_pixel_formats(self, jpeg_instance):
        """Test 12-bit with different pixel formats (RGB, BGR, GRAY)."""
        # RGB
        img_rgb = np.random.randint(0, 4096, (50, 50, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img_rgb, pixel_format=TJPF_RGB)
        decoded = jpeg_instance.decode_12bit(jpeg_buf, pixel_format=TJPF_RGB)
        assert decoded.shape == img_rgb.shape
        
        # BGR
        img_bgr = np.random.randint(0, 4096, (50, 50, 3), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img_bgr, pixel_format=TJPF_BGR)
        decoded = jpeg_instance.decode_12bit(jpeg_buf, pixel_format=TJPF_BGR)
        assert decoded.shape == img_bgr.shape
        
        # GRAY
        img_gray = np.random.randint(0, 4096, (50, 50, 1), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_12bit(img_gray, pixel_format=TJPF_GRAY)
        decoded = jpeg_instance.decode_12bit(jpeg_buf, pixel_format=TJPF_GRAY)
        assert decoded.shape == img_gray.shape
    
    def test_16bit_different_pixel_formats(self, jpeg_instance):
        """Test 16-bit with different pixel formats (RGB, BGR, GRAY)."""
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
    
    def test_16bit_grayscale(self, jpeg_instance):
        """Test single-channel grayscale 16-bit images."""
        img_gray = np.random.randint(0, 65536, (100, 100, 1), dtype=np.uint16)
        jpeg_buf = jpeg_instance.encode_16bit(img_gray, pixel_format=TJPF_GRAY, jpeg_subsample=TJSAMP_GRAY)
        decoded = jpeg_instance.decode_16bit(jpeg_buf, pixel_format=TJPF_GRAY)
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
    
    def test_16bit_with_flags(self, jpeg_instance, sample_16bit_image):
        """Test 16-bit with compression flags (PROGRESSIVE, FASTDCT)."""
        # Progressive
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image, flags=TJFLAG_PROGRESSIVE)
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.shape == sample_16bit_image.shape
        
        # Fast DCT
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image, flags=TJFLAG_FASTDCT)
        decoded = jpeg_instance.decode_16bit(jpeg_buf, flags=TJFLAG_FASTDCT)
        assert decoded.shape == sample_16bit_image.shape
    
    def test_12bit_invalid_precision_parameter(self, jpeg_instance, sample_12bit_image):
        """Test error handling for invalid precision values."""
        with pytest.raises(ValueError, match='precision must be 8, 12, or 16'):
            jpeg_instance.encode(sample_12bit_image, precision=10)
        
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        with pytest.raises(ValueError, match='precision must be 8, 12, or 16'):
            jpeg_instance.decode(jpeg_buf, precision=10)
    
    def test_16bit_invalid_precision_parameter(self, jpeg_instance, sample_16bit_image):
        """Test error handling for invalid precision values."""
        with pytest.raises(ValueError, match='precision must be 8, 12, or 16'):
            jpeg_instance.encode(sample_16bit_image, precision=24)
        
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        with pytest.raises(ValueError, match='precision must be 8, 12, or 16'):
            jpeg_instance.decode(jpeg_buf, precision=0)
    
    def test_12bit_wrong_dtype_input(self, jpeg_instance):
        """Test error when uint8 is passed for 12-bit encoding."""
        img_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match='img_array must be uint16 for 12/16-bit precision'):
            jpeg_instance.encode(img_uint8, precision=12)
    
    def test_16bit_wrong_dtype_input(self, jpeg_instance):
        """Test error when uint8 is passed for 16-bit encoding."""
        img_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match='img_array must be uint16 for 12/16-bit precision'):
            jpeg_instance.encode(img_uint8, precision=16)
    
    def test_mixed_precision_encode_decode(self, jpeg_instance, sample_12bit_image):
        """Test encoding at one precision and decoding at another."""
        # Encode as 12-bit
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        
        # Decode as 8-bit (downconversion)
        decoded_8bit = jpeg_instance.decode(jpeg_buf, precision=8)
        assert decoded_8bit.dtype == np.uint8
        assert decoded_8bit.shape == sample_12bit_image.shape
        
        # Decode as 16-bit (upconversion)
        decoded_16bit = jpeg_instance.decode(jpeg_buf, precision=16)
        assert decoded_16bit.dtype == np.uint16
        assert decoded_16bit.shape == sample_12bit_image.shape
    
    def test_12bit_memory_continuity(self, jpeg_instance):
        """Test multiple 12-bit encode/decode cycles (100 iterations)."""
        img = np.random.randint(0, 4096, (50, 50, 3), dtype=np.uint16)
        for _ in range(100):
            jpeg_buf = jpeg_instance.encode_12bit(img)
            decoded = jpeg_instance.decode_12bit(jpeg_buf)
            assert decoded.shape == img.shape
            assert decoded.dtype == np.uint16
    
    def test_16bit_memory_continuity(self, jpeg_instance):
        """Test multiple 16-bit encode/decode cycles (100 iterations)."""
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
    
    def test_16bit_edge_values(self, jpeg_instance):
        """Test 16-bit with min (0) and max (65535) values."""
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
    
    def test_convenience_methods_16bit(self, jpeg_instance, sample_16bit_image):
        """Test encode_16bit() and decode_16bit() convenience methods."""
        # Test encode_16bit
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image, quality=90)
        assert isinstance(jpeg_buf, bytes)
        
        # Test decode_16bit
        decoded = jpeg_instance.decode_16bit(jpeg_buf)
        assert decoded.shape == sample_16bit_image.shape
        assert decoded.dtype == np.uint16
    
    def test_12bit_decode_header(self, jpeg_instance, sample_12bit_image):
        """Test that decode_header works with 12-bit JPEGs."""
        jpeg_buf = jpeg_instance.encode_12bit(sample_12bit_image)
        width, height, subsample, colorspace = jpeg_instance.decode_header(jpeg_buf)
        assert width == sample_12bit_image.shape[1]
        assert height == sample_12bit_image.shape[0]
        assert subsample >= 0
        assert colorspace >= 0
    
    def test_16bit_decode_header(self, jpeg_instance, sample_16bit_image):
        """Test that decode_header works with 16-bit JPEGs."""
        jpeg_buf = jpeg_instance.encode_16bit(sample_16bit_image)
        width, height, subsample, colorspace = jpeg_instance.decode_header(jpeg_buf)
        assert width == sample_16bit_image.shape[1]
        assert height == sample_16bit_image.shape[0]
        assert subsample >= 0
        assert colorspace >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

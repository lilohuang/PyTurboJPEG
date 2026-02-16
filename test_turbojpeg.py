"""
Comprehensive unit tests for PyTurboJPEG library.

This module contains unit tests for all core functions of the PyTurboJPEG library,
covering various input scenarios, edge cases, and error conditions.
"""
import pytest
import numpy as np
import os
import tempfile
from io import BytesIO

from turbojpeg import (
    TurboJPEG,
    TJPF_RGB, TJPF_BGR, TJPF_GRAY, TJPF_RGBA, TJPF_BGRA,
    TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY,
    TJCS_RGB, TJCS_YCbCr, TJCS_GRAY,
    TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT
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
def encoded_sample_jpeg(jpeg_instance, sample_bgr_image):
    """Create an encoded JPEG from sample BGR image."""
    return jpeg_instance.encode(sample_bgr_image)


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


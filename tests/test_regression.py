"""
Regression tests for PyTurboJPEG focusing on historical bugs and edge cases.

This module contains regression tests for:
1. Robustness of Buffer Handling (empty bytes, truncated JPEG headers)
2. Library Loading (missing library, clear error messages)
3. Colorspace Consistency (all TJPF/TJSAMP combinations)
4. Memory Management (stress testing with 1000+ cycles)
"""

import pytest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
from ctypes.util import find_library

# Add parent directory to path to import turbojpeg
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbojpeg import (
    TurboJPEG,
    TJPF_RGB, TJPF_BGR, TJPF_GRAY, TJPF_RGBA, TJPF_BGRA, TJPF_RGBX, TJPF_BGRX,
    TJPF_XBGR, TJPF_XRGB, TJPF_ABGR, TJPF_ARGB,
    TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY, TJSAMP_440, TJSAMP_411,
    TJCS_RGB, TJCS_YCbCr, TJCS_GRAY,
)


# Test fixtures
@pytest.fixture(scope='module')
def jpeg_instance():
    """Create a TurboJPEG instance for testing."""
    return TurboJPEG()


@pytest.fixture(scope='module')
def sample_image():
    """Create a sample BGR image for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            img[i, j] = [i * 2, j * 2, (i + j) % 256]
    return img


@pytest.fixture(scope='module')
def valid_jpeg(jpeg_instance, sample_image):
    """Create a valid encoded JPEG for testing."""
    return jpeg_instance.encode(sample_image)


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
        try:
            with pytest.warns(UserWarning, match="JPEG datastream"):
                result = jpeg_instance.decode(truncated_header)
                # If it doesn't raise, should return empty or minimal array
                assert result.size == 0 or result.shape[0] == 0 or result.shape[1] == 0
        except (RuntimeError, OSError):
            # This is also acceptable - raising an error for invalid data
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
        except (RuntimeError, OSError):
            # This is also acceptable - raising an error for invalid data
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
        except (RuntimeError, OSError):
            # This is also acceptable - raising an error for invalid data
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
            # If it doesn't raise, should return zeros or minimal values
            assert width == 0 or height == 0
        except (RuntimeError, OSError):
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
    """
    
    def test_encode_decode_stress_1000_cycles(self, jpeg_instance):
        """Test 1000+ encode/decode cycles for memory stability."""
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
    
    def test_decode_header_stress(self, jpeg_instance, valid_jpeg):
        """Test decode_header repeatedly for memory stability."""
        # Perform 1000 decode_header operations
        for i in range(1000):
            width, height, subsample, colorspace = jpeg_instance.decode_header(valid_jpeg)
            assert width == 100
            assert height == 100
            assert subsample in [TJSAMP_444, TJSAMP_422, TJSAMP_420, TJSAMP_GRAY]
            assert colorspace in [TJCS_RGB, TJCS_YCbCr, TJCS_GRAY]
    
    def test_buffer_size_calculation_stress(self, jpeg_instance):
        """Test buffer_size calculation repeatedly for memory stability."""
        test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Perform 1000 buffer_size calculations
        for i in range(1000):
            buffer_size = jpeg_instance.buffer_size(test_img)
            assert buffer_size > 0
            assert isinstance(buffer_size, int)
    
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

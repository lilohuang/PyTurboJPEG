# PyTurboJPEG Test Suite

This directory contains comprehensive unit tests for the PyTurboJPEG library.

## Test Coverage

The test suite covers all core functions of PyTurboJPEG:

### Core Functions Tested
- **TurboJPEG Initialization** - Tests default initialization and library loading
- **decode_header()** - Tests JPEG header decoding with valid and invalid data
- **decode()** - Tests JPEG decoding with various pixel formats, scaling factors, and flags
- **decode_to_yuv()** - Tests YUV decoding with different parameters
- **decode_to_yuv_planes()** - Tests YUV plane decoding
- **encode()** - Tests JPEG encoding with various quality levels, pixel formats, and subsample modes
- **encode_from_yuv()** - Tests encoding from YUV data
- **scale_with_quality()** - Tests scaling and quality adjustment
- **crop()** - Tests lossless crop operations
- **crop_multiple()** - Tests multiple crop operations with background handling
- **buffer_size()** - Tests buffer size calculation
- **scaling_factors** - Tests the scaling factors property

### Test Categories

1. **Initialization Tests** - Verify TurboJPEG instance creation and properties
2. **Decode Tests** - Test various decoding scenarios including different pixel formats and scaling
3. **Encode Tests** - Test encoding with different quality levels and formats
4. **YUV Tests** - Test YUV encoding/decoding
5. **Transformation Tests** - Test crop and scale operations
6. **Error Handling Tests** - Test error conditions and edge cases
7. **Integration Tests** - Test complete workflows and roundtrip operations

## Running the Tests

### Prerequisites
```bash
# Install dependencies
sudo apt-get install libturbojpeg  # On Ubuntu/Debian
# OR
brew install jpeg-turbo  # On macOS

# Install Python dependencies
pip install numpy pytest
```

### Run All Tests
```bash
pytest test_turbojpeg.py -v
```

### Run Specific Test Classes
```bash
# Run only decode tests
pytest test_turbojpeg.py::TestDecode -v

# Run only encode tests
pytest test_turbojpeg.py::TestEncode -v

# Run only integration tests
pytest test_turbojpeg.py::TestIntegration -v
```

### Run Specific Tests
```bash
# Run a single test
pytest test_turbojpeg.py::TestDecode::test_decode_basic -v
```

### Generate Coverage Report
```bash
pytest test_turbojpeg.py --cov=turbojpeg --cov-report=html
```

## Test Structure

Each test class focuses on a specific function or feature:

- `TestTurboJPEGInitialization` - Tests initialization and properties
- `TestDecodeHeader` - Tests header decoding functionality
- `TestDecode` - Tests image decoding functionality
- `TestDecodeToYUV` - Tests YUV buffer decoding
- `TestDecodeToYUVPlanes` - Tests YUV plane decoding
- `TestEncode` - Tests image encoding functionality
- `TestEncodeFromYUV` - Tests encoding from YUV data
- `TestScaleWithQuality` - Tests scaling with quality adjustment
- `TestCrop` - Tests lossless crop operations
- `TestCropMultiple` - Tests multiple crop operations
- `TestBufferSize` - Tests buffer size calculation
- `TestErrorHandling` - Tests error conditions
- `TestIntegration` - Tests complete workflows

## Test Data

The test suite uses synthetic test images generated via fixtures:
- `sample_bgr_image` - 100x100 BGR gradient image
- `sample_rgb_image` - 100x100 RGB gradient image
- `sample_gray_image` - 100x100 grayscale gradient image
- `encoded_sample_jpeg` - Pre-encoded JPEG for decoding tests

## Edge Cases and Error Handling

The tests verify:
- Invalid JPEG data handling
- Empty buffer handling
- Invalid scaling factors
- Invalid image shapes
- Buffer size validation
- Various pixel format conversions
- Different quality levels and their impact

## Contributing

When adding new features to PyTurboJPEG, please:
1. Add corresponding tests to this test suite
2. Ensure all existing tests still pass
3. Document test cases clearly
4. Test both success and failure scenarios

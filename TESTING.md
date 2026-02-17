# PyTurboJPEG Test Suite

This repository contains comprehensive unit tests for the PyTurboJPEG library.

## Requirements

**TurboJPEG 3.0 or later is required** for running these tests. PyTurboJPEG 2.0+ uses the new function-based TurboJPEG 3 API and is not compatible with TurboJPEG 2.x.

The tests account for TurboJPEG 3.0+ specific behavior:
- Error messages differ from TJ 2.x ("Premature end of JPEG file" vs "JPEG datastream")
- DCT implementation may produce different rounding results in decoded images
- Invalid JPEG data may return -1 for dimensions instead of 0

## Test Coverage

The test suite covers all core functions of PyTurboJPEG plus regression tests for historical bugs:

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
8. **Regression Tests** - Tests for historical bugs and edge cases:
   - **Buffer Handling Robustness** (12 tests) - Empty buffers, truncated headers, corrupted data
   - **Library Loading** (3 tests) - Missing library error handling with clear messages
   - **Colorspace Consistency** (31 tests) - All TJPF/TJSAMP combinations
   - **Memory Management** (6 tests) - 1000+ encode/decode cycles with memory leak detection using pytest-memray
   - **Crop Functionality** (10 tests) - Comprehensive crop testing with real input image

## Running the Tests

### Prerequisites
```bash
# Install dependencies
sudo apt-get install libturbojpeg  # On Ubuntu/Debian
# OR
brew install jpeg-turbo  # On macOS

# Install Python dependencies
pip install numpy pytest pytest-memray
```

### Run All Tests
```bash
pytest tests/test_turbojpeg.py -v
# Or run all tests in the tests directory
pytest tests/ -v
```

### Run Specific Test Classes
```bash
# Run only decode tests
pytest tests/test_turbojpeg.py::TestDecode -v

# Run only encode tests
pytest tests/test_turbojpeg.py::TestEncode -v

# Run only integration tests
pytest tests/test_turbojpeg.py::TestIntegration -v

# Run regression tests
pytest tests/test_turbojpeg.py::TestBufferHandlingRobustness -v
pytest tests/test_turbojpeg.py::TestColorspaceConsistency -v
pytest tests/test_turbojpeg.py::TestMemoryManagement -v
pytest tests/test_turbojpeg.py::TestCropFunctionality -v
```

### Run Specific Tests
```bash
# Run a single test
pytest tests/test_turbojpeg.py::TestDecode::test_decode_basic -v
```

### Generate Coverage Report
```bash
pytest tests/test_turbojpeg.py --cov=turbojpeg --cov-report=html
```

## Test Structure

Each test class focuses on a specific function or feature:

### Core Function Tests
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

### Regression Tests
- `TestBufferHandlingRobustness` - Invalid buffer handling tests
- `TestLibraryLoading` - Library loading, version detection, and error message tests
- `TestColorspaceConsistency` - All pixel format/subsampling combinations
- `TestMemoryManagement` - Stress testing with 1000+ cycles and memory leak detection
- `TestCropFunctionality` - Crop function with real input image

## Memory Leak Detection

The `TestMemoryManagement` class uses **pytest-memray** to detect memory leaks during repeated function execution. Each test is decorated with `@pytest.mark.limit_memory()` to set memory growth limits:

- Tests will fail if memory usage exceeds the specified limit
- Helps catch slow memory leaks that accumulate over many iterations
- Memory limits are tuned based on expected working set size for each test

Example memory limits:
- `test_encode_decode_stress_1000_cycles`: 50 MB limit
- `test_encode_decode_varying_sizes_stress`: 100 MB limit (larger images)
- `test_decode_header_stress`: 20 MB limit (header-only operations)

To run tests with memory profiling:
```bash
# Run with memray memory tracking
pytest tests/test_turbojpeg.py::TestMemoryManagement -v

# Generate memory flamegraph for a specific test
pytest --memray tests/test_turbojpeg.py::TestMemoryManagement::test_encode_decode_stress_1000_cycles
```

## Test Data

The test suite uses synthetic test images generated via fixtures:
- `sample_bgr_image` - 100x100 BGR gradient image
- `sample_rgb_image` - 100x100 RGB gradient image
- `sample_gray_image` - 100x100 grayscale gradient image
- `sample_image` - Alias for sample_bgr_image (used in regression tests)
- `encoded_sample_jpeg` - Pre-encoded JPEG for decoding tests
- `valid_jpeg` - Valid encoded JPEG for testing
- `tests/test_crop_input.jpg` - 200x200 image with colored quadrants for crop tests

## Test Statistics

- **Total Tests**: 114
- **Passing**: 114 (100%)
- **Skipped**: 0
- **Core Function Tests**: 53
- **Regression Tests**: 61
  - Buffer Handling: 12 (updated for TJ 3.0+ error messages)
  - Library Loading: 3
  - Colorspace Consistency: 31
  - Memory Management: 6 (with pytest-memray leak detection)
  - Crop Functionality: 10 (all passing)

## Edge Cases and Error Handling

The tests verify:
- Invalid JPEG data handling
- Empty buffer handling
- Truncated JPEG headers and data
- Corrupted JPEG data
- Invalid scaling factors
- Invalid image shapes
- Buffer size validation
- Various pixel format conversions
- Different quality levels and their impact
- MCU alignment in crop operations
- Memory stability under stress
- Memory leak detection during repeated operations

## Contributing

When adding new features to PyTurboJPEG, please:
1. Add corresponding tests to this test suite
2. Ensure all existing tests still pass
3. Document test cases clearly
4. Test both success and failure scenarios

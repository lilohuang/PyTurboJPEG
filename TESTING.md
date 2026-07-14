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
- **optimize()** - Tests lossless Huffman table optimization
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
   - **Buffer Handling Robustness** - Empty buffers, truncated headers, corrupted data
   - **Library Loading** - Missing/old libraries and real TurboJPEG 3 construction
   - **Colorspace Consistency** - All TJPF/TJSAMP combinations, including 4:4:1
   - **Memory Management** - Repeated encode/decode cycles with pytest-memray limits
   - **Crop Functionality** - MCU alignment, partial edges, DQT, and background fill
   - **Security Regressions** - Buffer bounds, resource limits, flags, and input validation

## Running the Tests

### Prerequisites
```bash
# Install dependencies
sudo apt-get install libturbojpeg  # On Ubuntu/Debian
# OR
brew install jpeg-turbo  # On macOS

# Install the project and its complete test toolchain
python -m pip install -e ".[test]"
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
- `TestOptimize` - Tests lossless Huffman table optimization
- `TestBufferSize` - Tests buffer size calculation
- `TestErrorHandling` - Tests error conditions
- `TestIntegration` - Tests complete workflows

### Regression Tests
- `TestBufferHandlingRobustness` - Invalid buffer handling tests
- `TestLibraryLoading` - Library loading, version detection, and error message tests
- `TestColorspaceConsistency` - All pixel format/subsampling combinations
- `TestMemoryManagement` - Stress testing with 1000+ cycles and memory leak detection
- `TestCropFunctionality` - Crop function with real input image
- `TestYUVMetadataAndPadding` - Odd dimensions, explicit layout metadata, and zeroed padding
- `TestFlagSemantics` - Native flag mapping, output semantics, and rejected combinations
- `TestPublicInputValidation` - Shape, enum, range, and empty-image validation
- `TestCropAndTransformRegressions` - 4:4:1, partial iMCUs, DQT, and callbacks
- `TestScaleWithQualityResources` - Temporary-buffer dtype and handle lifetime

## Memory Leak Detection

The `TestMemoryManagement` class uses **pytest-memray** to detect memory leaks
during repeated function execution. Installing the `test` extra installs the
plugin on supported platforms. Each test is decorated with
`@pytest.mark.limit_memory()` to set memory growth limits:

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

The suite changes whenever regression coverage is added, so this document does
not store a fixed test count. Use pytest as the source of truth:

```bash
pytest --collect-only -q
pytest -q
```

## Performance Regression Benchmark

The performance suite is opt-in and is not part of the default pytest or CI
run. It compares the current working tree with the newest release tag whose
version is lower than the current `turbojpeg.__version__`. Both implementations
are loaded into the same Python process, pinned to one CPU when supported, and
their default encode, decode, header, and YUV paths are measured in alternating
order:

```bash
PYTURBOJPEG_RUN_BENCHMARKS=1 \
pytest benchmarks/test_performance.py -v -s
```

The test covers 8x8, 32x32, and 1280x720 images. It fails when the current
implementation exceeds both the default 5% relative tolerance and the 0.25
microsecond absolute noise allowance. These settings can be overridden when
investigating a regression:

```bash
PYTURBOJPEG_RUN_BENCHMARKS=1 \
PYTURBOJPEG_BENCHMARK_BASELINE=v2.4.0 \
PYTURBOJPEG_BENCHMARK_ROUNDS=15 \
PYTURBOJPEG_BENCHMARK_MAX_PERCENT=3 \
PYTURBOJPEG_BENCHMARK_MAX_US=0.15 \
pytest benchmarks/test_performance.py -v -s
```

Always compare revisions on the same machine with the same Python, NumPy, and
libjpeg-turbo versions. Absolute timings are intentionally not committed
because they are not portable across systems.

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

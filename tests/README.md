# PyTurboJPEG Regression Tests

This directory contains regression tests for PyTurboJPEG, focusing on historical pain points and edge cases.

## Test Files

### `test_regression.py`
Comprehensive regression tests covering:

#### 1. Buffer Handling Robustness (12 tests)
Tests for invalid buffers to ensure they raise appropriate errors instead of crashing:
- Empty buffer decoding and header decoding
- Truncated JPEG headers (very short, partial)
- Truncated JPEG data
- Invalid JPEG magic numbers
- Random bytes and non-JPEG data (PNG headers)
- Corrupted JPEG data
- YUV decoding with invalid data

These tests verify that the library handles malformed input gracefully with proper error messages or warnings.

#### 2. Library Loading (3 tests)
Tests for library loading edge cases:
- Invalid library path handling
- Helpful error messages when library is not found
- Successful loading with explicit path

These tests ensure users get clear guidance when libturbojpeg is not installed.

#### 3. Colorspace Consistency (31 tests)
Comprehensive tests for all pixel format and subsampling combinations:
- All 11 pixel formats (RGB, BGR, GRAY, RGBA, BGRA, RGBX, BGRX, XBGR, XRGB, ABGR, ARGB)
- All 5 subsampling modes (444, 422, 420, 440, 411, GRAY)
- Various combinations of pixel formats and subsampling
- Buffer size calculation consistency

These tests verify that output buffer dimensions match expected values for all combinations.

#### 4. Memory Management (5 tests)
Stress tests for memory stability:
- 1000+ encode/decode cycles with same image
- 1000 cycles with varying image sizes (50x50 to 200x200)
- 1000+ cycles with different pixel formats (RGB, BGR, GRAY)
- 1000 decode_header operations
- 1000 buffer_size calculations
- Multiple TurboJPEG instances creation and usage

These tests verify no memory leaks or stability issues occur during extended use.

## Running the Tests

### Run all regression tests:
```bash
pytest tests/test_regression.py -v
```

### Run specific test classes:
```bash
# Buffer handling tests
pytest tests/test_regression.py::TestBufferHandlingRobustness -v

# Library loading tests
pytest tests/test_regression.py::TestLibraryLoading -v

# Colorspace consistency tests
pytest tests/test_regression.py::TestColorspaceConsistency -v

# Memory management tests
pytest tests/test_regression.py::TestMemoryManagement -v
```

### Run with coverage:
```bash
pytest tests/test_regression.py --cov=turbojpeg --cov-report=html
```

## Test Statistics

- **Total Regression Tests**: 51
- **Buffer Handling Tests**: 12
- **Library Loading Tests**: 3
- **Colorspace Consistency Tests**: 31
- **Memory Management Tests**: 5

Combined with the existing test suite in `test_turbojpeg.py` (53 tests), this brings the total test count to **104 tests**.

## Requirements

- pytest
- numpy
- libturbojpeg (system library)

## Notes

- Some tests may issue warnings (e.g., for truncated JPEGs) - this is expected behavior
- The library is resilient and may decode corrupted data with warnings rather than raising exceptions
- Buffer size calculations are intentionally conservative (may be up to 3x raw data size)
- Memory stress tests take approximately 1-2 seconds to complete

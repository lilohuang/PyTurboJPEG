# PyTurboJPEG

A Python wrapper for libjpeg-turbo that enables efficient JPEG image decoding and encoding.

## Prerequisites

- [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases) **3.0 or later** (required for PyTurboJPEG 2.0+)
- [numpy](https://github.com/numpy/numpy)

**Important:** PyTurboJPEG 2.0+ requires libjpeg-turbo 3.0 or later as it uses the new function-based TurboJPEG 3 API. For libjpeg-turbo 2.x compatibility, please use PyTurboJPEG 1.x.

## Installation

### macOS
```bash
brew install jpeg-turbo
pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
```

### Windows
1. Download the [libjpeg-turbo official installer](https://github.com/libjpeg-turbo/libjpeg-turbo/releases)
2. Install PyTurboJPEG:
   ```bash
   pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
   ```

### Linux
1. Download the [libjpeg-turbo official installer](https://github.com/libjpeg-turbo/libjpeg-turbo/releases)
2. Install PyTurboJPEG:
   ```bash
   pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
   ```

## Basic Usage

### Initialization

```python
from turbojpeg import TurboJPEG

# Use default library installation
jpeg = TurboJPEG()

# Or specify library path explicitly
# jpeg = TurboJPEG(r'D:\turbojpeg.dll')  # Windows
# jpeg = TurboJPEG('/usr/lib64/libturbojpeg.so')  # Linux
# jpeg = TurboJPEG('/usr/local/lib/libturbojpeg.dylib')  # macOS
```

### Decoding

```python
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT

jpeg = TurboJPEG()

# Basic decoding to BGR array
with open('input.jpg', 'rb') as f:
    bgr_array = jpeg.decode(f.read())
cv2.imshow('bgr_array', bgr_array)
cv2.waitKey(0)

# Fast decoding (lower accuracy, higher speed)
with open('input.jpg', 'rb') as f:
    bgr_array = jpeg.decode(f.read(), flags=TJFLAG_FASTUPSAMPLE|TJFLAG_FASTDCT)

# Decode with direct rescaling (1/2 size)
with open('input.jpg', 'rb') as f:
    bgr_array_half = jpeg.decode(f.read(), scaling_factor=(1, 2))

# Get available scaling factors
scaling_factors = jpeg.scaling_factors

# Decode to grayscale
with open('input.jpg', 'rb') as f:
    gray_array = jpeg.decode(f.read(), pixel_format=TJPF_GRAY)
```

### Decoding Header Information

```python
# Get image properties without full decoding
with open('input.jpg', 'rb') as f:
    width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(f.read())
```

### YUV Decoding

```python
# Decode to YUV buffer
with open('input.jpg', 'rb') as f:
    buffer_array, plane_sizes = jpeg.decode_to_yuv(f.read())

# Decode to YUV planes
with open('input.jpg', 'rb') as f:
    planes = jpeg.decode_to_yuv_planes(f.read())
```

### Encoding

```python
from turbojpeg import TJSAMP_GRAY, TJFLAG_PROGRESSIVE

# Basic encoding with default settings
with open('output.jpg', 'wb') as f:
    f.write(jpeg.encode(bgr_array))

# Encode with grayscale subsample
with open('output_gray.jpg', 'wb') as f:
    f.write(jpeg.encode(bgr_array, jpeg_subsample=TJSAMP_GRAY))

# Encode with custom quality
with open('output_quality_50.jpg', 'wb') as f:
    f.write(jpeg.encode(bgr_array, quality=50))

# Encode with progressive entropy coding
with open('output_progressive.jpg', 'wb') as f:
    f.write(jpeg.encode(bgr_array, quality=100, flags=TJFLAG_PROGRESSIVE))
```

### Advanced Operations

```python
# Scale with quality (without color conversion)
with open('input.jpg', 'rb') as f:
    scaled_data = jpeg.scale_with_quality(f.read(), scaling_factor=(1, 4), quality=70)
with open('scaled_output.jpg', 'wb') as f:
    f.write(scaled_data)

# Lossless crop
with open('input.jpg', 'rb') as f:
    cropped_data = jpeg.crop(f.read(), 8, 8, 320, 240)
with open('cropped_output.jpg', 'wb') as f:
    f.write(cropped_data)
```

### In-Place Operations

```python
import numpy as np

# In-place decoding (reuse existing array)
img_array = np.empty((640, 480, 3), dtype=np.uint8)
with open('input.jpg', 'rb') as f:
    result = jpeg.decode(f.read(), dst=img_array)
# result is the same as img_array: id(result) == id(img_array)

# In-place encoding (reuse existing buffer)
buffer_size = jpeg.buffer_size(img_array)
dest_buf = bytearray(buffer_size)
result, n_bytes = jpeg.encode(img_array, dst=dest_buf)
with open('output.jpg', 'wb') as f:
    f.write(dest_buf[:n_bytes])
# result is the same as dest_buf: id(result) == id(dest_buf)
```

### EXIF Orientation Handling

```python
import cv2
import numpy as np
import exifread
from turbojpeg import TurboJPEG

def transpose_image(image, orientation):
    """Transpose image based on EXIF Orientation tag.
    
    See: https://www.exif.org/Exif2-2.PDF
    """
    if orientation is None:
        return image
    
    val = orientation.values[0]
    if val == 1: return image
    elif val == 2: return np.fliplr(image)
    elif val == 3: return np.rot90(image, 2)
    elif val == 4: return np.flipud(image)
    elif val == 5: return np.rot90(np.flipud(image), -1)
    elif val == 6: return np.rot90(image, -1)
    elif val == 7: return np.rot90(np.flipud(image))
    elif val == 8: return np.rot90(image)

jpeg = TurboJPEG()

with open('foobar.jpg', 'rb') as f:
    # Parse EXIF orientation
    orientation = exifread.process_file(f).get('Image Orientation', None)
    
    # Decode image
    f.seek(0)
    image = jpeg.decode(f.read())
    
    # Apply orientation transformation
    transposed_image = transpose_image(image, orientation)

cv2.imshow('transposed_image', transposed_image)
cv2.waitKey(0)
```

## High-Precision JPEG Support

PyTurboJPEG 2.0+ supports 12-bit and 16-bit precision JPEG encoding and decoding using libjpeg-turbo 3.0+ APIs. This feature is ideal for medical imaging, scientific photography, and other applications requiring higher bit depth.

**Requirements:**
- libjpeg-turbo 3.0 or later (12-bit and 16-bit support is built-in)

**Important Limitations:**
- **12-bit JPEG:** Supports both lossy and lossless compression
- **16-bit JPEG:** Only supports lossless compression (JPEG standard limitation)
  - The `encode_16bit()` method will raise `NotImplementedError` as lossless mode is not yet exposed in this API
  - Use `encode_12bit()` for high-precision lossy JPEG encoding

### 12-bit JPEG (Lossy)

12-bit JPEG provides higher precision than standard 8-bit JPEG while maintaining compatibility with lossy compression.

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Create 12-bit image (values range from 0 to 4095)
img_12bit = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)

# Encode to 12-bit JPEG
jpeg_data = jpeg.encode_12bit(img_12bit, quality=95)

# Decode from 12-bit JPEG
decoded_img = jpeg.decode_12bit(jpeg_data)

# Save to file
with open('output_12bit.jpg', 'wb') as f:
    f.write(jpeg_data)

# Load from file
with open('output_12bit.jpg', 'rb') as f:
    decoded_from_file = jpeg.decode_12bit(f.read())
```

### 16-bit JPEG (Lossless Only - Not Yet Supported)

**Note:** 16-bit JPEG is only supported for lossless compression in the JPEG standard. Since lossless mode is not currently exposed in this API, 16-bit encoding is not available. Use 12-bit encoding for high-precision lossy JPEG compression.

The JPEG standard does not support 16-bit lossy compression. If you need lossless 16-bit JPEG support, please open an issue on the GitHub repository.

### Medical and Scientific Imaging

For medical and scientific applications, 12-bit JPEG provides excellent precision while maintaining file size efficiency:

```python
import numpy as np
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

jpeg = TurboJPEG()

# Create 12-bit medical image (e.g., DICOM format)
# Medical images typically use 0-4095 range
medical_img = np.random.randint(0, 4096, (512, 512, 1), dtype=np.uint16)

# Encode with highest quality for medical applications
jpeg_medical = jpeg.encode_12bit(
    medical_img,
    pixel_format=TJPF_GRAY,
    jpeg_subsample=TJSAMP_GRAY,
    quality=100
)

# Decode for analysis
decoded_medical = jpeg.decode_12bit(jpeg_medical, pixel_format=TJPF_GRAY)

# Verify value range preservation
print(f"Original range: [{medical_img.min()}, {medical_img.max()}]")
print(f"Decoded range: [{decoded_medical.min()}, {decoded_medical.max()}]")
```

## License

See the LICENSE file for details.
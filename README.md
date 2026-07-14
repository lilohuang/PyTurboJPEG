# PyTurboJPEG

A Python wrapper for libjpeg-turbo that enables efficient JPEG image decoding and encoding.

[![PyPI Version](https://img.shields.io/pypi/v/pyturbojpeg.svg?style=flat-square&color=blue)](https://pypi.org/project/pyturbojpeg/)
![Python Version](https://img.shields.io/badge/python-3.8+-blue?logo=python&logoColor=white)
[![Downloads](https://img.shields.io/pypi/dm/pyturbojpeg.svg?style=flat-square&color=orange)](https://pypistats.org/packages/pyturbojpeg)
[![License](https://img.shields.io/github/license/lilohuang/PyTurboJPEG.svg?style=flat-square)](https://github.com/lilohuang/PyTurboJPEG/blob/master/LICENSE)

## Prerequisites

- [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases) **3.0 or later** (required for PyTurboJPEG 2.0+)
- [numpy](https://github.com/numpy/numpy)
- Python 3.8 or later

**Important:** PyTurboJPEG 2.0+ requires libjpeg-turbo 3.0 or later as it uses the new function-based TurboJPEG 3 API. For libjpeg-turbo 2.x compatibility, please use PyTurboJPEG 1.x.

## Installation

### macOS
```bash
brew install jpeg-turbo
pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
```

### Windows
1. Download the [official libjpeg-turbo installer](https://github.com/libjpeg-turbo/libjpeg-turbo/releases)
2. Install PyTurboJPEG:
   ```bash
   pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
   ```

### Linux
1. Download the [official libjpeg-turbo installer](https://github.com/libjpeg-turbo/libjpeg-turbo/releases)
2. Install PyTurboJPEG:
   ```bash
   pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
   ```

## Basic Usage

### Initialization

```python
from turbojpeg import TurboJPEG

# Use the default library installation
jpeg = TurboJPEG()

# Or specify the library path explicitly
# jpeg = TurboJPEG(r'D:\turbojpeg.dll')  # Windows
# jpeg = TurboJPEG('/usr/lib64/libturbojpeg.so')  # Linux
# jpeg = TurboJPEG('/usr/local/lib/libturbojpeg.dylib')  # macOS
```

Resource limits are disabled by default (`max_pixels=0`, `max_memory=0`, and
`scan_limit=0`) so trusted large-image workflows can use libjpeg-turbo's full
supported range.  Configure finite limits when processing untrusted JPEGs:

```python
jpeg = TurboJPEG(
    max_pixels=100_000_000,
    max_memory=512,  # megabytes
    scan_limit=500,
)
```

Native `max_memory` enforcement requires libjpeg-turbo 3.0.2 or later.  With
3.0.0 or 3.0.1, requesting a non-zero `max_memory` emits a warning;
PyTurboJPEG still enforces a configured `max_pixels` before Python output
allocation as well as a configured progressive scan limit.

### Decoding

```python
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT

jpeg = TurboJPEG()

# Basic decoding to a BGR array
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

# Get the available scaling factors
scaling_factors = jpeg.scaling_factors

# Decode to grayscale
with open('input.jpg', 'rb') as f:
    gray_array = jpeg.decode(f.read(), pixel_format=TJPF_GRAY)
```

### Decoding Header Information

```python
# Get image properties without full decoding (backward compatible)
with open('input.jpg', 'rb') as f:
    width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(f.read())

# Get the precision to select the appropriate decode function
with open('input.jpg', 'rb') as f:
    jpeg_data = f.read()
    width, height, jpeg_subsample, jpeg_colorspace, precision = jpeg.decode_header(jpeg_data, return_precision=True)
    
    # Use the precision to select the appropriate decode function
    if precision == 8:
        img = jpeg.decode(jpeg_data)
    elif precision == 12:
        img = jpeg.decode_12bit(jpeg_data)
    elif precision == 16:
        img = jpeg.decode_16bit(jpeg_data)
```

### YUV Decoding

```python
# Decode to a YUV buffer
with open('input.jpg', 'rb') as f:
    buffer_array, plane_sizes = jpeg.decode_to_yuv(f.read())

# Request explicit offsets, strides, and valid plane dimensions
with open('input.jpg', 'rb') as f:
    buffer_array, plane_info = jpeg.decode_to_yuv(
        f.read(), pad=8, return_metadata=True)

# Decode to YUV planes
with open('input.jpg', 'rb') as f:
    planes = jpeg.decode_to_yuv_planes(f.read())

# A unified YUV buffer must be re-encoded with the same row alignment
with open('input.jpg', 'rb') as f:
    jpeg_data = f.read()
width, height, jpeg_subsample, _ = jpeg.decode_header(jpeg_data)
yuv_buffer, _ = jpeg.decode_to_yuv(jpeg_data, pad=8)
encoded = jpeg.encode_from_yuv(
    yuv_buffer,
    height=height,
    width=width,
    jpeg_subsample=jpeg_subsample,
    align=8,
)
```

`plane_sizes` contains the native `(height, width)` of every YUV plane. These
dimensions can be larger than the visible JPEG dimensions when chroma
subsampling rounds an odd width or height upward. Each unified-buffer row
stride is its plane's width rounded up to `pad`; plane offsets are the
cumulative `stride * height` values. `return_metadata=True` returns those
values directly as `YUVPlaneInfo(offset, stride, width, height)`. All row and
plane padding is zero-initialized.

Legacy `TJFLAG_*` values are mapped to TurboJPEG 3 parameters and every native
parameter-set result is checked. Flags that have no meaning for an operation
raise `ValueError` instead of being silently ignored. For example,
`TJFLAG_BOTTOMUP` applies to packed-pixel `encode()`/`decode()` calls, whereas
planar YUV calls reject it. Progressive and DCT flags are rejected for lossless
JPEG because those modes have no corresponding lossless semantics.

### Encoding

```python
from turbojpeg import TJSAMP_GRAY, TJFLAG_PROGRESSIVE

# Basic encoding with default settings
with open('output.jpg', 'wb') as f:
    f.write(jpeg.encode(bgr_array))

# Encode with grayscale subsampling
with open('output_gray.jpg', 'wb') as f:
    f.write(jpeg.encode(bgr_array, jpeg_subsample=TJSAMP_GRAY))

# Encode with custom quality
with open('output_quality_50.jpg', 'wb') as f:
    f.write(jpeg.encode(bgr_array, quality=50))

# Encode with progressive entropy coding
with open('output_progressive.jpg', 'wb') as f:
    f.write(jpeg.encode(bgr_array, quality=100, flags=TJFLAG_PROGRESSIVE))

# Encode with lossless JPEG compression
with open('output_gray.jpg', 'wb') as f:
    f.write(jpeg.encode(bgr_array, lossless=True))
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

# Lossless Huffman table optimization (re-encodes with optimal tables,
# identical pixels; typically smaller unless already optimized)
with open('input.jpg', 'rb') as f:
    optimized_data = jpeg.optimize(f.read())
with open('optimized_output.jpg', 'wb') as f:
    f.write(optimized_data)
```

Lossless crop origins must be aligned to the JPEG iMCU grid. The default
`preserve=False` rounds the origin down and expands the result to retain the
whole requested area. `preserve=True` rounds it up so the result stays inside
the requested area. In both modes, valid partial iMCUs at the right and bottom
image edges are retained.

### In-Place Operations

```python
import numpy as np

# In-place decoding (reuse an existing array)
img_array = np.empty((640, 480, 3), dtype=np.uint8)
with open('input.jpg', 'rb') as f:
    result = jpeg.decode(f.read(), dst=img_array)
# result is the same as img_array: id(result) == id(img_array)

# In-place encoding (reuse an existing buffer)
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
    """Transpose an image based on the EXIF Orientation tag.
    
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
    
    # Decode the image
    f.seek(0)
    image = jpeg.decode(f.read())
    
    # Apply the orientation transformation
    transposed_image = transpose_image(image, orientation)

cv2.imshow('transposed_image', transposed_image)
cv2.waitKey(0)
```

### ICC Color Management Workflow

```python
import io
import numpy as np
from PIL import Image, ImageCms
from turbojpeg import TurboJPEG, TJPF_BGR

def decode_jpeg_with_color_management(jpeg_path):
    """
    Decodes a JPEG and applies color management by converting the image from
    its embedded ICC profile to sRGB.
    
    Args:
        jpeg_path (str): Path to the input JPEG file.
        
    Returns:
        PIL.Image: The color-corrected sRGB Image object.
    """
    # 1. Initialize TurboJPEG
    jpeg = TurboJPEG()
    
    with open(jpeg_path, 'rb') as f:
        jpeg_data = f.read()
    
    # 2. Get image headers and decode pixels
    # Use the TJPF_BGR format (OpenCV standard) for the raw buffer
    width, height, _, _ = jpeg.decode_header(jpeg_data)
    pixels = jpeg.decode(jpeg_data, pixel_format=TJPF_BGR)
    
    # 3. Encapsulate the pixels in a Pillow Image object
    # Key: Use the 'raw' decoder with 'BGR' to map BGR bytes to an RGB Image object
    img = Image.frombytes('RGB', (width, height), pixels, 'raw', 'BGR')
    
    # 4. Handle the ICC profile transformation
    try:
        # Extract the embedded ICC profile
        icc_profile = jpeg.get_icc_profile(jpeg_data)
        
        if icc_profile:
            # Create the source and destination profile objects
            src_profile = ImageCms.getOpenProfile(io.BytesIO(icc_profile))
            dst_profile = ImageCms.createProfile("sRGB")
            
            # Perform the color transformation (similar to "Convert to Profile" in Photoshop)
            # This step recalculates pixel values to align with the sRGB standard
            img = ImageCms.profileToProfile(
                img, 
                src_profile, 
                dst_profile, 
                outputMode='RGB'
            )
            print(f"Successfully applied the ICC profile from {jpeg_path}")
        else:
            print("No ICC profile found; assuming sRGB.")
            
    except Exception as e:
        print(f"Color Management Error: {e}. Returning the original raw image.")
        
    return img

# --- Example Usage ---
if __name__ == "__main__":
    result_img = decode_jpeg_with_color_management('icc_profile.jpg')
    result_img.show()
    # result_img.save('output_srgb.jpg', quality=95)
```

## High-Precision JPEG Support

PyTurboJPEG 2.0+ supports 12-bit and 16-bit JPEG encoding and decoding using libjpeg-turbo 3.0+ APIs. This feature is ideal for medical imaging, scientific photography, and other applications requiring higher bit depth.

**Requirements:**
- libjpeg-turbo 3.0 or later (12-bit and 16-bit support is built-in)

**Precision Modes:**
- **12-bit JPEG:** Supports both lossy and lossless compression
- **16-bit JPEG:** Only supports lossless compression (JPEG standard limitation)

### 12-bit JPEG (Lossy)

12-bit JPEG provides higher precision than standard 8-bit JPEG while maintaining compatibility with lossy compression.

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Create a 12-bit image (values range from 0 to 4095)
img_12bit = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)

# Encode to 12-bit lossy JPEG
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

### Lossless JPEG for 12-bit and 16-bit

Both 12-bit and 16-bit JPEG formats support lossless compression for perfect reconstruction:

#### 12-bit Lossless JPEG

To use 12-bit precision with lossless compression:

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Create a 12-bit image
img_12bit = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)

# Encode to 12-bit lossless JPEG using encode_12bit() with lossless=True
jpeg_data = jpeg.encode_12bit(img_12bit, lossless=True)

# Decode using decode_12bit()
decoded_img = jpeg.decode_12bit(jpeg_data)

# Perfect reconstruction
assert np.array_equal(img_12bit, decoded_img)  # True
```

#### 16-bit Lossless JPEG

16-bit JPEG provides the highest precision with perfect reconstruction through lossless compression. The JPEG standard only supports 16-bit precision in lossless mode.

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Create a 16-bit image (values range from 0 to 65535)
img_16bit = np.random.randint(0, 65536, (480, 640, 3), dtype=np.uint16)

# Encode to 16-bit lossless JPEG
jpeg_data = jpeg.encode_16bit(img_16bit)

# Decode from 16-bit lossless JPEG
decoded_img = jpeg.decode_16bit(jpeg_data)

# Verify perfect reconstruction (lossless)
assert np.array_equal(img_16bit, decoded_img)  # True

# Save to file
with open('output_16bit_lossless.jpg', 'wb') as f:
    f.write(jpeg_data)

# Load from file
with open('output_16bit_lossless.jpg', 'rb') as f:
    decoded_from_file = jpeg.decode_16bit(f.read())
```

### Medical and Scientific Imaging

For medical and scientific applications, 12-bit JPEG provides excellent precision while maintaining file size efficiency:

```python
import numpy as np
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

jpeg = TurboJPEG()

# Create a 12-bit medical image (e.g., DICOM format)
# Medical images typically use values in the 0-4095 range
medical_img = np.random.randint(0, 4096, (512, 512, 1), dtype=np.uint16)

# Encode with the highest quality for medical applications
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

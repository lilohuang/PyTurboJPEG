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

`decode_header()` can return `TJSAMP_UNKNOWN` (`-1`) for a valid JPEG with
unusual sampling factors. Such images can still be decoded to packed pixels
with `decode()` or `decode_12bit()`, according to their precision. Planar YUV
operations, `scale_with_quality()`, and cropping require known subsampling
and raise `ValueError` for these images.

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

With `TJFLAG_STOPONWARNING`, a native warning raises `OSError` because the
operation stops before its output is necessarily complete. Without that flag,
recoverable warnings emit `UserWarning` and return the completed result.
If decoding into `dst` fails, the destination may have been partially written
and must not be treated as a complete image.

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

### JPEG Storage Colorspace

`pixel_format` describes the input array's channel order. The optional
`jpeg_colorspace` argument selects the colorspace stored in the JPEG.
Its default, `None`, preserves automatic selection: ordinary RGB/BGR inputs
are encoded as YCbCr in lossy mode, even with `TJSAMP_444`.

`TJCS_DEFAULT` (`-1`) is an explicit alternative to `None` in `encode()`,
`encode_12bit()`, and `encode_16bit()`:

```python
from turbojpeg import TJCS_DEFAULT

encoded = jpeg.encode(bgr_array, jpeg_colorspace=TJCS_DEFAULT)
```

Although the native setter accepts `TJCS_DEFAULT` only in libjpeg-turbo 3.2
or later, PyTurboJPEG supports this option with every supported version
(3.0+). Each encoding call creates a new native handle, so automatic selection
is preserved by leaving its colorspace setting untouched.

To compress three independent channels (for example, distinct stain signals)
without an RGB-to-YCbCr conversion, explicitly select RGB storage and 4:4:4:

```python
from turbojpeg import TJPF_RGB, TJCS_RGB, TJSAMP_444

encoded = jpeg.encode(
    three_channel_array,  # uint8, shape (height, width, 3)
    pixel_format=TJPF_RGB,
    jpeg_colorspace=TJCS_RGB,
    jpeg_subsample=TJSAMP_444,
    quality=90,
)
decoded = jpeg.decode(encoded, pixel_format=TJPF_RGB)
assert jpeg.decode_header(encoded)[3] == TJCS_RGB
```

This avoids mixing the channels during color conversion; lossy JPEG still
introduces DCT/quantization error within each channel. The same arguments work
with `encode_12bit()` for `uint16` input containing values from 0 to 4095.
This feature requires libjpeg-turbo 3.0 or later, the project's existing minimum.

Explicit colorspace selection uses these rules for lossy encoding:

| `jpeg_colorspace` | Supported input pixel formats | Required `jpeg_subsample` |
| --- | --- | --- |
| `TJCS_RGB` | RGB/BGR and their four-channel variants | `TJSAMP_444` |
| `TJCS_YCbCr` | RGB/BGR and their four-channel variants | Any except `TJSAMP_GRAY` |
| `TJCS_GRAY` | RGB/BGR variants or `TJPF_GRAY` | `TJSAMP_GRAY` |
| `TJCS_CMYK` | `TJPF_CMYK` | `TJSAMP_444` |
| `TJCS_YCCK` | `TJPF_CMYK` | Any except `TJSAMP_GRAY` |

Conflicting settings raise `ValueError`. In particular, specifying
`jpeg_colorspace=TJCS_RGB` also requires explicitly selecting `TJSAMP_444`,
since the default subsampling is 4:2:2. Four-channel RGB formats retain their
existing behavior: the alpha/unused channel is not stored in the JPEG.
When reusing an output buffer, size it with
`jpeg.buffer_size(image, jpeg_subsample=TJSAMP_444)` for RGB storage.

Lossless encoding already preserves the input colorspace without conversion.
With `lossless=True`, or when using `encode_16bit()`, `None` and `TJCS_DEFAULT`
select the input's native colorspace automatically. Any other explicit
`jpeg_colorspace` must match the input: `TJCS_RGB` for RGB/BGR variants,
`TJCS_GRAY` for grayscale, or `TJCS_CMYK` for CMYK. Other selections raise
`ValueError`; quality and subsampling remain ignored in lossless mode.

RGB JPEGs use the packed-pixel `encode()`/`decode()` APIs. The native codec
does not support encoding them from, or decoding them to, planar YUV;
`encode_from_yuv()` does not accept `jpeg_colorspace`.

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

`crop()` aligns crop origins to the JPEG iMCU grid. The default
`preserve=False` rounds the origin down and expands the result to retain the
whole requested area. `preserve=True` rounds it up so the result stays inside
the requested area. In both modes, valid partial iMCUs at the right and bottom
image edges are retained.

#### Batch cropping and background extension

`crop_multiple()` accepts an iterable of `(x, y, w, h)` tuples and returns JPEG
byte strings in the same order, using one native batch transform. The iterable
may contain just one tuple. Each axis is interpreted separately:

- If `w`/`h` is no larger than the corresponding source dimension, `x`/`y` is
  a crop origin inside the source image.
- `w=0` or `h=0` keeps everything from that origin to the source's right or
  bottom edge. The output dimensions need not be multiples of the iMCU size.
- If `w`/`h` is larger than the source dimension, that axis is extended.
  `x`/`y` then places the source in the output canvas. The complete source must
  fit along that axis: `x + source_width <= w` or `y + source_height <= h`.
  Placement offsets may be larger than the source dimensions.

For a 505x731 JPEG with 4:2:0 subsampling (16x16 iMCUs), remove a 16-pixel left
black bar and retain the entire remaining image, including partial edge blocks:

```python
with open('input.jpg', 'rb') as f:
    data = f.read()

# Both forms produce the same 489x731 JPEG as
# jpeg.crop(data, 16, 0, 489, 731, preserve=True).
cropped = jpeg.crop_multiple(data, [(16, 0, 0, 0)])[0]
explicit = jpeg.crop_multiple(data, [(16, 0, 489, 731)])[0]
```

The keyword-only `align` option controls **pure crops** (operations that do not
extend either axis):

| `align` | Behavior |
| --- | --- |
| `"strict"` (default) | Require aligned origins and an in-bounds rectangle. |
| `"expand"` | Round origins down, retaining the requested area within the source; matches `crop(preserve=False)`. |
| `"shrink"` | Round origins up, keeping the result inside the requested area; matches `crop(preserve=True)`. |

For `expand` and `shrink`, the requested right/bottom edges stay fixed and are
clipped to the source image. A zero dimension is resolved to the remaining
source extent before alignment. If inward alignment leaves no pixels, the
operation raises `ValueError`.

```python
# On the same 4:2:0 image, request (x=10, y=10, w=20, h=20).
expanded = jpeg.crop_multiple(data, [(10, 10, 20, 20)], align='expand')[0]
# Actual source rectangle: (0, 0, 30, 30)

contained = jpeg.crop_multiple(data, [(10, 10, 20, 20)], align='shrink')[0]
# Actual source rectangle: (16, 16, 14, 14)
```

Operations extending either axis always require aligned `x` **and** `y`, even
with `align='expand'` or `'shrink'`. Their placement and crop offsets are never
automatically shifted. Pure crops and extensions can be mixed in the same batch.
Choosing an alignment mode does not change a requested extension into a crop.

Background extension supports grayscale, YCbCr, and RGB JPEGs, with
`background_luminance` from 0 (black) to 1 (white). RGB backgrounds use each
channel's own quantization table. CMYK/YCCK JPEGs support cropping but not
background extension. For example, center a 64x64 image in a 320x320 white canvas:

```python
import numpy as np
from turbojpeg import TJPF_RGB, TJSAMP_420

image = np.full((64, 64, 3), (30, 90, 160), dtype=np.uint8)
source = jpeg.encode(image, pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_420)
centered = jpeg.crop_multiple(
    source, [(128, 128, 320, 320)], background_luminance=1.0)[0]
```

An extension requires the source dimension on each extended axis to be a
multiple of its iMCU size; otherwise `ValueError` is raised consistently on
all supported library versions. With YCbCr-to-grayscale conversion, this source
dimension check uses the grayscale output's 8x8 blocks; origins still follow the
source JPEG's iMCU grid. Partial edges on an axis that is not extended are
preserved. Alignment modes do not remove this source-dimension requirement.
This prevents older libjpeg-turbo releases from silently discarding source
pixels during extension. These operations do not decode and recompress pixels.

Region validation errors identify the **zero-based** batch index, parameter,
source dimensions, subsampling, and iMCU size. For example, an unaligned `x` in
the third region raises (wrapped here for readability):

```text
crop_parameters[2]: x=10 is not aligned; x must be a multiple of 16;
use align='expand' (keeps the requested in-bounds pixels but may include extra area),
or align='shrink' (discards unaligned edges of the requested area and may be empty);
source=505x731, subsampling=4:2:0, iMCU=16x16
```

Geometry errors suggest remedies and explain their costs:

| Problem | Explicit remedy | Tradeoff |
|---|---|---|
| Unaligned origin in a pure crop | Choose `align='expand'` or `'shrink'`. | `expand` includes extra source area; `shrink` discards the unaligned part of the requested area and can leave no pixels. Neither recompresses the image. |
| `shrink` leaves no pixels | Increase the requested extent to include the next aligned origin, using the minimum in the error. At a partial right/bottom edge with no later aligned origin, move the origin back or use `expand`. | Enlarging the request or moving back includes pixels outside the original request. Exact unaligned pixel bounds require decoding, cropping, and re-encoding, which adds lossy compression. |
| A crop exceeds the right/bottom edge | Use the suggested maximum `w`/`h`, or `0` for the remaining extent. | The output becomes smaller than requested. An origin outside the source must also be moved inside, which selects a different source region. |
| The source does not fit at the requested canvas offset | Move it to the suggested aligned `x`/`y`, or enlarge the canvas enough to fit the whole source at an aligned offset. | Moving changes placement; enlarging changes output dimensions. An unaligned offset still requires a manual adjustment for extensions. |
| An extended source dimension has a partial iMCU | First crop the **source** dimension to the suggested multiple, or decode, pad, and re-encode. | Cropping discards the stated number of right-edge columns or bottom-edge rows without recompressing retained coefficients. Pixel padding preserves the full source area but re-encoding adds lossy compression. Changing output `w`/`h` or `align` cannot fix this source constraint. |
| CMYK/YCCK background extension | Use a pixel-padding workflow supporting the source colorspace, then re-encode. | Re-encoding adds lossy compression. Preserve color profiles explicitly; this RGB example below is not a CMYK/YCCK workflow. |

For example, extending a **100x64, 4:2:0** source to 128x64 requires a choice.
If losing its four rightmost columns is acceptable, explicitly crop the source
width to 96 (a multiple of 16) first:

```python
# data is the original 100x64, 4:2:0 JPEG.
trimmed = jpeg.crop(data, 0, 0, 96, 64)
padded = jpeg.crop_multiple(trimmed, [(0, 0, 128, 64)])[0]
# Retains the first 96 source columns; adds 32 white columns on the right.
```

If every source column is required, pad decoded pixels instead. This 8-bit RGB
example retains all 100 columns before re-encoding and adds 28 white columns:

```python
pixels = jpeg.decode(data, pixel_format=TJPF_RGB)
canvas = np.full((64, 128, 3), 255, dtype=np.uint8)
canvas[:, :100] = pixels
padded = jpeg.encode(canvas, pixel_format=TJPF_RGB,
                     jpeg_subsample=TJSAMP_420, quality=95)
```

This second approach performs an additional lossy compression pass: pixels can
change even at `quality=100`, and ICC/EXIF metadata is not copied automatically.
For 12-bit data, use `decode_12bit()`/`encode_12bit()`, `np.uint16`, and white
value 4095. A source smaller than one iMCU on the extended axis cannot be fixed
by cropping to a positive aligned dimension; use the pixel-padding approach.
Remedies address the reported axis or constraint; the other axis and remaining
batch entries must also be valid. The wrapper never silently trims the source,
moves its placement, or falls back to recompression.

All regions are validated before the native batch transform runs. Invalid region
geometry raises `ValueError`, and invalid numeric types raise `TypeError`, without
returning a partially completed list. Out-of-bounds crops or placements that
previously raised a native `OSError` now raise a contextual `ValueError`.
Errors in the JPEG itself or batch-wide options are reported for the whole call.

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

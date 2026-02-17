# PyTurboJPEG
A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image.

## Prerequisites
- [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases) **3.0 or later** (required for PyTurboJPEG 2.0+)
- [numpy](https://github.com/numpy/numpy)

**Important**: PyTurboJPEG 2.0+ requires libjpeg-turbo 3.0 or later as it uses the new function-based TurboJPEG 3 API. If you need to use libjpeg-turbo 2.x, please use PyTurboJPEG 1.x instead.

## Example

```python
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT

# specifying library path explicitly
# jpeg = TurboJPEG(r'D:\turbojpeg.dll')
# jpeg = TurboJPEG('/usr/lib64/libturbojpeg.so')
# jpeg = TurboJPEG('/usr/local/lib/libturbojpeg.dylib')

# using default library installation
jpeg = TurboJPEG()

# decoding input.jpg to BGR array
in_file = open('input.jpg', 'rb')
bgr_array = jpeg.decode(in_file.read())
in_file.close()
cv2.imshow('bgr_array', bgr_array)
cv2.waitKey(0)

# decoding input.jpg to BGR array with fast upsample and fast DCT. (i.e. fastest speed but lower accuracy)
in_file = open('input.jpg', 'rb')
bgr_array = jpeg.decode(in_file.read(), flags=TJFLAG_FASTUPSAMPLE|TJFLAG_FASTDCT)
in_file.close()
cv2.imshow('bgr_array', bgr_array)
cv2.waitKey(0)

# direct rescaling 1/2 while decoding input.jpg to BGR array
in_file = open('input.jpg', 'rb')
bgr_array_half = jpeg.decode(in_file.read(), scaling_factor=(1, 2))
in_file.close()
cv2.imshow('bgr_array_half', bgr_array_half)
cv2.waitKey(0)

# getting possible scaling factors for direct rescaling
scaling_factors = jpeg.scaling_factors

# decoding JPEG image properties
in_file = open('input.jpg', 'rb')
width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(in_file.read())
in_file.close()

# decoding input.jpg to YUV array
in_file = open('input.jpg', 'rb')
buffer_array, plane_sizes = jpeg.decode_to_yuv(in_file.read())
in_file.close()

# decoding input.jpg to YUV planes
in_file = open('input.jpg', 'rb')
planes = jpeg.decode_to_yuv_planes(in_file.read())
in_file.close()

# encoding BGR array to output.jpg with default settings.
out_file = open('output.jpg', 'wb')
out_file.write(jpeg.encode(bgr_array))
out_file.close()

# encoding BGR array to output.jpg with TJSAMP_GRAY subsample.
out_file = open('output_gray.jpg', 'wb')
out_file.write(jpeg.encode(bgr_array, jpeg_subsample=TJSAMP_GRAY))
out_file.close()

# encoding BGR array to output.jpg with quality level 50. 
out_file = open('output_quality_50.jpg', 'wb')
out_file.write(jpeg.encode(bgr_array, quality=50))
out_file.close()

# encoding BGR array to output.jpg with quality level 100 and progressive entropy coding.
out_file = open('output_quality_100_progressive.jpg', 'wb')
out_file.write(jpeg.encode(bgr_array, quality=100, flags=TJFLAG_PROGRESSIVE))
out_file.close()

# decoding input.jpg to grayscale array
in_file = open('input.jpg', 'rb')
gray_array = jpeg.decode(in_file.read(), pixel_format=TJPF_GRAY)
in_file.close()
cv2.imshow('gray_array', gray_array)
cv2.waitKey(0)

# scale with quality but leaves out the color conversion step
in_file = open('input.jpg', 'rb')
out_file = open('scaled_output.jpg', 'wb')
out_file.write(jpeg.scale_with_quality(in_file.read(), scaling_factor=(1, 4), quality=70))
out_file.close()
in_file.close()

# lossless crop image
out_file = open('lossless_cropped_output.jpg', 'wb')
out_file.write(jpeg.crop(open('input.jpg', 'rb').read(), 8, 8, 320, 240))
out_file.close()

# in-place decoding input.jpg to BGR array
# here I use a 640x480 example (in practise, read the dimensions)
in_file = open('input.jpg', 'rb')
img_array = np.empty((640, 480, 3), dtype=np.uint8)
result = jpeg.decode(in_file.read(), dst=img_array)
in_file.close()

# return value is the img_array argument value
id(result) == id(img_array)
# True

# Optional: display the in-place array
# cv2.imshow('img_array', img_array)
# cv2.waitKey(0)

# in-place encoding with default settings.
buffer_size = jpeg.buffer_size(img_array)
dest_buf = bytearray(buffer_size)
result, n_byte = jpeg.encode(img_array, dst=dest_buf)

# return value is the dest_buf argument value
id(result) == id(dest_buf)

out_file = open('output.jpg', 'wb')
out_file.write(dest_buf[:n_byte])
out_file.close()
```

```python
# using PyTurboJPEG with ExifRead to transpose an image if the image has an EXIF Orientation tag.
#
# pip install PyTurboJPEG -U
# pip install exifread -U

import cv2
import numpy as np
import exifread
from turbojpeg import TurboJPEG

def transposeImage(image, orientation):
    """See Orientation in https://www.exif.org/Exif2-2.PDF for details."""
    if orientation == None: return image
    val = orientation.values[0]
    if val == 1: return image
    elif val == 2: return np.fliplr(image)
    elif val == 3: return np.rot90(image, 2)
    elif val == 4: return np.flipud(image)
    elif val == 5: return np.rot90(np.flipud(image), -1)
    elif val == 6: return np.rot90(image, -1)
    elif val == 7: return np.rot90(np.flipud(image))
    elif val == 8: return np.rot90(image)

# using default library installation
turbo_jpeg = TurboJPEG()
# open jpeg file
in_file = open('foobar.jpg', 'rb')
# parse orientation
orientation = exifread.process_file(in_file).get('Image Orientation', None)
# seek file position back to 0 before decoding JPEG image
in_file.seek(0)
# start to decode the JPEG file
image = turbo_jpeg.decode(in_file.read())
# transpose image based on EXIF Orientation tag
transposed_image = transposeImage(image, orientation)
# close the file since it's no longer needed.
in_file.close()

cv2.imshow('transposed_image', transposed_image)
cv2.waitKey(0)
```

## High-Precision JPEG Support (12-bit and 16-bit)

PyTurboJPEG 2.0+ supports 12-bit and 16-bit precision JPEG encoding and decoding using TurboJPEG 3.0+ APIs. This is useful for medical imaging, scientific photography, and other applications requiring higher bit depth.

**Note**: 12-bit support is available in standard TurboJPEG 3.0+ builds. 16-bit support requires a special build with `-DWITH_12BIT=1 -DWITH_16BIT=1` flags.

### Basic 12-bit Encoding and Decoding

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Create a 12-bit image (uint16 array with values 0-4095)
img_12bit = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)

# Encode 12-bit image to JPEG
jpeg_buf = jpeg.encode_12bit(img_12bit, quality=95)

# Decode JPEG back to 12-bit image
decoded_12bit = jpeg.decode_12bit(jpeg_buf)

# Save to file
with open('output_12bit.jpg', 'wb') as f:
    f.write(jpeg_buf)

# Load and decode from file
with open('output_12bit.jpg', 'rb') as f:
    decoded_from_file = jpeg.decode_12bit(f.read())
```

### Basic 16-bit Encoding and Decoding

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Create a 16-bit image (uint16 array with full range 0-65535)
img_16bit = np.random.randint(0, 65536, (480, 640, 3), dtype=np.uint16)

# Encode 16-bit image to JPEG
jpeg_buf = jpeg.encode_16bit(img_16bit, quality=95)

# Decode JPEG back to 16-bit image
decoded_16bit = jpeg.decode_16bit(jpeg_buf)

# Save to file
with open('output_16bit.jpg', 'wb') as f:
    f.write(jpeg_buf)

# Load and decode from file
with open('output_16bit.jpg', 'rb') as f:
    decoded_from_file = jpeg.decode_16bit(f.read())
```

### Using Explicit Precision Parameter

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Create a 12-bit image
img_12bit = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)

# Encode with explicit precision parameter
jpeg_buf = jpeg.encode(img_12bit, quality=95, precision=12)

# Decode with explicit precision parameter
decoded = jpeg.decode(jpeg_buf, precision=12)
```

### 12-bit Grayscale Images

```python
import numpy as np
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

jpeg = TurboJPEG()

# Create a 12-bit grayscale image (single channel)
img_12bit_gray = np.random.randint(0, 4096, (480, 640, 1), dtype=np.uint16)

# Encode 12-bit grayscale with appropriate subsampling
jpeg_buf = jpeg.encode_12bit(
    img_12bit_gray, 
    pixel_format=TJPF_GRAY,
    jpeg_subsample=TJSAMP_GRAY,
    quality=95
)

# Decode back to 12-bit grayscale
decoded_gray = jpeg.decode_12bit(jpeg_buf, pixel_format=TJPF_GRAY)
```

### High-Precision with Different Quality Levels

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()
img_12bit = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)

# Encode with different quality levels
jpeg_high_quality = jpeg.encode_12bit(img_12bit, quality=100)
jpeg_medium_quality = jpeg.encode_12bit(img_12bit, quality=85)
jpeg_low_quality = jpeg.encode_12bit(img_12bit, quality=50)

print(f"High quality size: {len(jpeg_high_quality)} bytes")
print(f"Medium quality size: {len(jpeg_medium_quality)} bytes")
print(f"Low quality size: {len(jpeg_low_quality)} bytes")
```

### High-Precision with Different Subsampling

```python
import numpy as np
from turbojpeg import TurboJPEG, TJSAMP_444, TJSAMP_422, TJSAMP_420

jpeg = TurboJPEG()
img_12bit = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)

# 4:4:4 subsampling (no chroma subsampling, highest quality)
jpeg_444 = jpeg.encode_12bit(img_12bit, jpeg_subsample=TJSAMP_444, quality=95)

# 4:2:2 subsampling (default, good balance)
jpeg_422 = jpeg.encode_12bit(img_12bit, jpeg_subsample=TJSAMP_422, quality=95)

# 4:2:0 subsampling (most compression, lower quality)
jpeg_420 = jpeg.encode_12bit(img_12bit, jpeg_subsample=TJSAMP_420, quality=95)

print(f"4:4:4 size: {len(jpeg_444)} bytes")
print(f"4:2:2 size: {len(jpeg_422)} bytes")
print(f"4:2:0 size: {len(jpeg_420)} bytes")
```

### High-Precision with Progressive Encoding

```python
import numpy as np
from turbojpeg import TurboJPEG, TJFLAG_PROGRESSIVE

jpeg = TurboJPEG()
img_12bit = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)

# Encode with progressive mode for better compression
jpeg_progressive = jpeg.encode_12bit(
    img_12bit,
    quality=95,
    flags=TJFLAG_PROGRESSIVE
)

# Decode progressive JPEG
decoded = jpeg.decode_12bit(jpeg_progressive)
```

### Converting Between Precisions

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Start with an 8-bit image
img_8bit = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

# Upscale to 12-bit range (multiply by 16 to map 0-255 to 0-4080)
img_12bit = (img_8bit.astype(np.uint16) * 16)

# Encode as 12-bit JPEG
jpeg_12bit = jpeg.encode_12bit(img_12bit, quality=95)

# Decode back to 12-bit
decoded_12bit = jpeg.decode_12bit(jpeg_12bit)

# Downscale back to 8-bit if needed (divide by 16)
img_8bit_converted = (decoded_12bit / 16).astype(np.uint8)
```

### Checking JPEG Header Information

```python
import numpy as np
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

# Encode a 12-bit image
img_12bit = np.random.randint(0, 4096, (480, 640, 3), dtype=np.uint16)
jpeg_buf = jpeg.encode_12bit(img_12bit)

# Get image information from JPEG header
width, height, jpeg_subsample, jpeg_colorspace = jpeg.decode_header(jpeg_buf)

print(f"Image dimensions: {width}x{height}")
print(f"Subsampling: {jpeg_subsample}")
print(f"Colorspace: {jpeg_colorspace}")
```

### Working with Medical or Scientific Images

```python
import numpy as np
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

jpeg = TurboJPEG()

# Simulate a 12-bit medical image (e.g., from DICOM)
# Values typically range from 0 to 4095 in medical imaging
medical_image = np.random.randint(0, 4096, (512, 512, 1), dtype=np.uint16)

# Encode with lossless-like quality settings
jpeg_medical = jpeg.encode_12bit(
    medical_image,
    pixel_format=TJPF_GRAY,
    jpeg_subsample=TJSAMP_GRAY,
    quality=100  # Highest quality for medical images
)

# Decode for analysis
decoded_medical = jpeg.decode_12bit(jpeg_medical, pixel_format=TJPF_GRAY)

# Verify value ranges are preserved
print(f"Original range: [{medical_image.min()}, {medical_image.max()}]")
print(f"Decoded range: [{decoded_medical.min()}, {decoded_medical.max()}]")
```

## Installation

### macOS
- brew install jpeg-turbo
- pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git

### Windows 
- Download [libjpeg-turbo official installer](https://github.com/libjpeg-turbo/libjpeg-turbo/releases) 
- pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git

### Linux
- Download [libjpeg-turbo official installer](https://github.com/libjpeg-turbo/libjpeg-turbo/releases) 
- pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git


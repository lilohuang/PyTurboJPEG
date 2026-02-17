import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set TurboJPEG library path for testing if available
# This checks for common TurboJPEG 3.0+ installation locations
# Users can override by setting TURBOJPEG_LIB_PATH environment variable before running tests
if 'TURBOJPEG_LIB_PATH' not in os.environ:
    common_paths = [
        '/opt/libjpeg-turbo/lib64/libturbojpeg.so',  # Official TurboJPEG package
        '/usr/local/lib/libturbojpeg.so.0',          # Common Linux installation
        '/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0',  # Debian/Ubuntu
    ]
    for path in common_paths:
        if os.path.exists(path):
            os.environ['TURBOJPEG_LIB_PATH'] = path
            break

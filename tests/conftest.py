import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set TurboJPEG library path for testing if available
if os.path.exists('/opt/libjpeg-turbo/lib64/libturbojpeg.so'):
    os.environ['TURBOJPEG_LIB_PATH'] = '/opt/libjpeg-turbo/lib64/libturbojpeg.so'

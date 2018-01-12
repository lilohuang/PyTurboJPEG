from setuptools import setup, find_packages
setup(
    name='PyTurboJPEG',
    version='1.0.3',
    description='A Python wrapper of TurboJPEG for decoding and encoding JPEG image.',
    author='Lilo Huang',
    author_email='kuso.cc@gmail.com',
    url='https://github.com/lilohuang/PyTurboJPEG',
    license='MIT',
    install_requires=['numpy'],
    py_modules=['turbojpeg'],
    packages=find_packages(),
)

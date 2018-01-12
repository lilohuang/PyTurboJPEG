from setuptools import setup, find_packages
setup(
    name='PyTurboJPEG',
    author='Lilo Huang',
    author_email='kuso.cc@gmail.com',
    url='https://github.com/lilohuang/PyTurboJPEG',
    version='0.1',
    description='An experimental Python wrapper of TurboJPEG for decoding and encoding JPEG image.',
    license='MIT',
    install_requires=['numpy'],
    py_modules=['turbojpeg'],
    packages=find_packages(),
)

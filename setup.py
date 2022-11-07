import io
from setuptools import setup, find_packages
setup(
    name='PyTurboJPEG',
    version='1.7.0',
    description='A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image.',
    author='Lilo Huang',
    author_email='kuso.cc@gmail.com',
    url='https://github.com/lilohuang/PyTurboJPEG',
    license='MIT',
    install_requires=['numpy'],
    py_modules=['turbojpeg'],
    packages=find_packages(),
    long_description_content_type='text/markdown',
    long_description=io.open('README.md', encoding='utf-8').read()
)

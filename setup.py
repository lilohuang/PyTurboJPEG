import io
from setuptools import setup


setup(
    name='PyTurboJPEG',
    version='2.5.0',
    description='A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image.',
    author='Lilo Huang',
    author_email='kuso.cc@gmail.com',
    url='https://github.com/lilohuang/PyTurboJPEG',
    license='MIT',
    python_requires='>=3.8',
    install_requires=['numpy'],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.1.0',
            'pytest-memray>=1.7.0; platform_system != "Windows"',
        ],
    },
    py_modules=['turbojpeg'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
    ],
    long_description_content_type='text/markdown',
    long_description=io.open('README.md', encoding='utf-8').read()
)

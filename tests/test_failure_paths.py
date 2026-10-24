"""Regression tests for incomplete output, buffer aliasing, and ICC ownership."""

from ctypes import POINTER, c_size_t, c_void_p, cast
import os
import warnings

import numpy as np
import pytest

from turbojpeg import (
    TurboJPEG, TJCS_RGB, TJFLAG_PROGRESSIVE, TJFLAG_STOPONWARNING,
    TJINIT_COMPRESS, TJERR_FATAL, TJERR_WARNING, TJSAMP_444, TJSAMP_422,
)


@pytest.fixture(scope='module')
def jpeg():
    return TurboJPEG(os.environ.get('TURBOJPEG_LIB_PATH'))


@pytest.fixture(scope='module')
def image():
    return np.random.default_rng(42).integers(
        0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture(scope='module')
def encoded_images(jpeg):
    images = {}
    for bits, progressive in [(8, False), (8, True), (12, False),
                              (12, True), (16, False)]:
        source = np.random.default_rng(42).integers(
            0, 1 << bits, (64, 64, 3),
            dtype=np.uint8 if bits == 8 else np.uint16)
        encode = jpeg.encode if bits == 8 else getattr(jpeg, f'encode_{bits}bit')
        images[bits, progressive] = encode(
            source, flags=TJFLAG_PROGRESSIVE if progressive else 0)
    return images


def run_operation(jpeg, operation, data, flags=0):
    if operation == 'crop':
        return jpeg.crop(data, 0, 0, 32, 32)
    if operation == 'crop_multiple':
        return jpeg.crop_multiple(data, [(0, 0, 32, 32), (32, 32, 32, 32)])
    if operation in ('optimize', 'decode_header', 'get_icc_profile'):
        return getattr(jpeg, operation)(data)
    return getattr(jpeg, operation)(data, flags=flags)


OPERATIONS = {
    8: ('decode', 'decode_to_yuv', 'decode_to_yuv_planes', 'scale_with_quality',
        'crop', 'crop_multiple', 'optimize'),
    12: ('decode_12bit', 'crop', 'crop_multiple', 'optimize'),
    16: ('decode_16bit',),
}
DECODE_CASES = [
    (bits, progressive, operation)
    for bits in OPERATIONS
    for progressive in ([False, True] if bits != 16 else [False])
    for operation in OPERATIONS[bits]
]
BAD_MARKERS = [
    pytest.param(b'\xff\x02', 'Unsupported marker type', id='unknown-marker'),
    pytest.param(b'\xff\xd8', 'two SOI markers', id='duplicate-soi'),
    pytest.param(b'\xff\xdb\x00\x01', 'Bogus marker length', id='invalid-dqt'),
]


def damage_scan(data, marker):
    sos = data.index(b'\xff\xda')
    entropy = sos + 2 + int.from_bytes(data[sos + 2:sos + 4], 'big')
    # The early marker first produces JWRN_HIT_MARKER, then a fatal error.
    return data[:entropy + 5] + marker + data[entropy + 5:]


@pytest.mark.parametrize('bits,progressive,operation', DECODE_CASES)
@pytest.mark.parametrize('marker,error', BAD_MARKERS)
def test_warning_followed_by_fatal_raises_and_destroys_handle(
        jpeg, encoded_images, bits, progressive, operation, marker, error,
        monkeypatch):
    data = damage_scan(encoded_images[bits, progressive], marker)
    native_init = jpeg._TurboJPEG__init
    native_destroy = jpeg._TurboJPEG__destroy
    created, destroyed = [], []

    def create(mode):
        handle = native_init(mode)
        created.append(handle)
        return handle

    def destroy(handle):
        destroyed.append(handle)
        native_destroy(handle)

    monkeypatch.setattr(jpeg, '_TurboJPEG__init', create)
    monkeypatch.setattr(jpeg, '_TurboJPEG__destroy', destroy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(OSError, match=error):
            run_operation(jpeg, operation, data)
    assert not caught  # A fatal error must not be downgraded to UserWarning.
    assert created and sorted(created) == sorted(destroyed)


@pytest.mark.parametrize('bits,progressive', [
    (8, False), (8, True), (12, False), (12, True), (16, False)])
@pytest.mark.parametrize('marker,error', BAD_MARKERS)
def test_stop_on_warning_still_aborts_at_first_warning(
        jpeg, encoded_images, bits, progressive, marker, error):
    operation = 'decode' if bits == 8 else f'decode_{bits}bit'
    data = damage_scan(encoded_images[bits, progressive], marker)
    with pytest.raises(OSError, match='premature end of data segment'):
        run_operation(jpeg, operation, data, flags=TJFLAG_STOPONWARNING)


@pytest.mark.parametrize('bits,progressive,operation', DECODE_CASES)
@pytest.mark.parametrize('damage', ['missing_eoi', 'header_junk'])
def test_recoverable_warning_still_returns_complete_output(
        jpeg, encoded_images, bits, progressive, operation, damage):
    data = encoded_images[bits, progressive]
    expected = run_operation(jpeg, operation, data)
    if damage == 'missing_eoi':
        damaged = data[:-2]
        message = 'Premature end of JPEG file'
    else:
        damaged = data[:2] + b'junk' + data[2:]
        message = 'extraneous bytes before marker'
    with pytest.warns(UserWarning, match=message):
        actual = run_operation(jpeg, operation, damaged)
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize('operation', ('decode_header',) + OPERATIONS[8])
def test_header_warning_then_fatal_raises_before_image_allocation(
        jpeg, encoded_images, operation, monkeypatch):
    data = encoded_images[8, False]
    damaged = data[:2] + b'junk\xff\x02' + data[2:]

    def fail_allocation(*args, **kwargs):
        pytest.fail('An invalid header must be rejected before image allocation')

    monkeypatch.setattr(np, 'empty', fail_allocation)
    monkeypatch.setattr(np, 'zeros', fail_allocation)
    with pytest.raises(OSError, match='Unsupported marker type'):
        run_operation(jpeg, operation, damaged)


@pytest.mark.parametrize('progressive', [False, True])
@pytest.mark.parametrize('flags', [0, TJFLAG_STOPONWARNING])
def test_warning_then_fatal_never_returns_caller_destination(
        jpeg, encoded_images, progressive, flags):
    data = damage_scan(encoded_images[8, progressive], b'\xff\x02')
    destination = np.full((64, 64, 3), 0xA5, dtype=np.uint8)
    with pytest.raises(OSError):
        jpeg.decode(data, dst=destination, flags=flags)


@pytest.mark.parametrize('message', [
    'Unknown native warning',
    'Corrupt JPEG data: an unknown failure',
    'Premature end of JPEG file\nUnsupported marker type 0x02',
])
def test_unknown_warning_never_returns_unfinished_output(
        jpeg, encoded_images, monkeypatch, message):
    monkeypatch.setattr(jpeg, '_TurboJPEG__decompress', lambda *args: -1)
    monkeypatch.setattr(jpeg, '_TurboJPEG__get_error_code', lambda handle: TJERR_WARNING)
    monkeypatch.setattr(jpeg, '_TurboJPEG__get_error_string', lambda handle: message)
    with pytest.raises(OSError) as caught:
        jpeg.decode(encoded_images[8, False])
    assert str(caught.value) == message


@pytest.mark.parametrize('layout', [
    'same', 'destination_inside_source', 'source_inside_destination',
    'one_byte_overlap', 'adjacent', 'adjacent_reversed', 'separate',
])
@pytest.mark.parametrize('container', ['memoryview', 'numpy'])
@pytest.mark.parametrize('operation', ['encode', 'encode_lossless', 'decode'])
def test_buffer_overlap_rejected_but_disjoint_views_work(
        jpeg, image, layout, container, operation):
    lossless = operation == 'encode_lossless'
    encoded = jpeg.encode(image, lossless=lossless)
    if operation == 'decode':
        source_size, destination_size = len(encoded), image.nbytes
    else:
        source_size = image.nbytes
        destination_size = jpeg.buffer_size(image, TJSAMP_444) if lossless \
            else jpeg.buffer_size(image)
    source_offset, destination_offset = {
        'same': (0, 0),
        'destination_inside_source': (0, 32),
        'source_inside_destination': (32, 0),
        'one_byte_overlap': (0, source_size - 1),
        'adjacent': (0, source_size),
        'adjacent_reversed': (destination_size, 0),
        'separate': (0, source_size + 16),
    }[layout]
    storage = bytearray(max(source_offset + source_size,
                            destination_offset + destination_size))
    if operation == 'decode':
        storage[source_offset:source_offset + source_size] = encoded
        source = memoryview(storage)[source_offset:source_offset + source_size]
        if container == 'numpy':
            source = np.frombuffer(source, dtype=np.uint8)
        destination = np.ndarray(image.shape, dtype=np.uint8, buffer=storage,
                                 offset=destination_offset)

        def call():
            return jpeg.decode(source, dst=destination)
    else:
        source = np.ndarray(image.shape, dtype=np.uint8, buffer=storage,
                            offset=source_offset)
        source[:] = image
        destination = memoryview(storage)[
            destination_offset:destination_offset + destination_size]
        if container == 'numpy':
            destination = np.frombuffer(destination, dtype=np.uint8)

        def call():
            return jpeg.encode(source, dst=destination, lossless=lossless)

    before = bytes(storage)
    if layout in ('adjacent', 'adjacent_reversed', 'separate'):
        if operation == 'decode':
            actual = call()
            assert actual is destination
            np.testing.assert_array_equal(actual, jpeg.decode(encoded))
        else:
            actual, written = call()
            assert actual is destination
            assert bytes(memoryview(actual)[:written]) == encoded
        assert storage[source_offset:source_offset + source_size] == \
            before[source_offset:source_offset + source_size]
    else:
        with pytest.raises(ValueError, match='must not overlap.*separate'):
            call()
        assert bytes(storage) == before


def test_encode_rejects_overlapping_bytearray(jpeg, image):
    storage = bytearray(max(image.nbytes, jpeg.buffer_size(image)))
    source = np.ndarray(image.shape, dtype=np.uint8, buffer=storage)
    source[:] = image
    before = bytes(storage)
    with pytest.raises(ValueError, match='must not overlap'):
        jpeg.encode(source, dst=storage)
    assert bytes(storage) == before


def test_decode_rejects_overlapping_readonly_input_view(jpeg, image):
    encoded = jpeg.encode(image)
    storage = bytearray(max(image.nbytes, len(encoded)))
    storage[:len(encoded)] = encoded
    source = memoryview(storage)[:len(encoded)].toreadonly()
    destination = np.ndarray(image.shape, dtype=np.uint8, buffer=storage)
    before = bytes(storage)
    with pytest.raises(ValueError, match='must not overlap'):
        jpeg.decode(source, dst=destination)
    assert bytes(storage) == before


def test_encode_can_copy_noncontiguous_source_before_writing(jpeg, image):
    storage = bytearray(max(image.nbytes, jpeg.buffer_size(image)))
    source = np.ndarray(image.shape, dtype=np.uint8, buffer=storage)
    source[:] = image
    source = source[:, ::-1]
    expected = jpeg.encode(source.copy())
    _, written = jpeg.encode(source, dst=storage)
    assert bytes(storage[:written]) == expected


ICC_CHUNK = 65519
MAX_ICC_SIZE = 255 * ICC_CHUNK


@pytest.fixture
def icc_jpeg(jpeg):
    if jpeg._TurboJPEG__get_icc_profile is None:
        pytest.skip('ICC API requires libjpeg-turbo 3.1 or later')
    return jpeg


@pytest.mark.parametrize('size', [
    0, 1, ICC_CHUNK - 1, ICC_CHUNK, ICC_CHUNK + 1,
    2 * ICC_CHUNK, 2 * ICC_CHUNK + 1, 8 * 1024 * 1024, MAX_ICC_SIZE,
])
@pytest.mark.parametrize('mode', ['default', 'rgb', 'lossless'])
def test_icc_segment_boundaries_roundtrip_in_exact_sized_destination(
        icc_jpeg, size, mode):
    jpeg = icc_jpeg
    # The embedding API treats ICC payloads as opaque bytes. A changing byte
    # pattern also checks reassembly across segment boundaries.
    profile = np.arange(size, dtype=np.uint8).tobytes()
    source = np.zeros((1, 1, 3), dtype=np.uint8)
    kwargs = {'icc_profile': profile}
    if mode == 'rgb':
        kwargs['jpeg_colorspace'] = TJCS_RGB
    elif mode == 'lossless':
        kwargs['lossless'] = True
    subsample = TJSAMP_444 if mode == 'lossless' else TJSAMP_422
    capacity = jpeg.buffer_size(source, subsample, icc_profile=profile)
    expected_overhead = size + 18 * ((size + ICC_CHUNK - 1) // ICC_CHUNK)
    assert capacity == jpeg.buffer_size(source, subsample) + expected_overhead
    storage = bytearray([0xA5]) * (capacity + 32)
    destination = memoryview(storage)[:capacity]
    returned, written = jpeg.encode(source, dst=destination, **kwargs)
    assert returned is destination
    assert 0 < written <= capacity
    assert storage[capacity:] == b'\xa5' * 32
    data = bytes(destination[:written])
    assert data == jpeg.encode(source, **kwargs)
    assert jpeg.get_icc_profile(data) == (profile or None)


@pytest.mark.parametrize('capacity_kind', ['no_markers', 'one_byte_short'])
def test_icc_insufficient_capacity_rejected_before_writing(icc_jpeg, capacity_kind):
    jpeg = icc_jpeg
    source = np.zeros((1, 1, 3), dtype=np.uint8)
    profile = bytes(8 * 1024 * 1024)
    if capacity_kind == 'no_markers':
        capacity = jpeg.buffer_size(source) + len(profile)
    else:
        capacity = jpeg.buffer_size(source, icc_profile=profile) - 1
    destination = bytearray([0xA5]) * capacity
    with pytest.raises(ValueError, match='buffer is too small'):
        jpeg.encode(source, dst=destination, icc_profile=profile)
    assert destination == b'\xa5' * capacity


@pytest.mark.parametrize('operation', ['encode', 'set_icc_profile', 'buffer_size'])
@pytest.mark.parametrize('container', ['bytes', 'memoryview', 'uint16'])
def test_icc_over_255_segments_rejected_before_native_set(
        icc_jpeg, operation, container, monkeypatch):
    jpeg = icc_jpeg
    profile = bytes(MAX_ICC_SIZE + 1)
    if container == 'memoryview':
        profile = memoryview(profile)
    elif container == 'uint16':
        profile = np.frombuffer(profile, dtype=np.uint16)

    def fail_native_set(*args):
        pytest.fail('Oversized ICC data must be rejected before native allocation')

    monkeypatch.setattr(jpeg, '_TurboJPEG__set_icc_profile', fail_native_set)
    source = np.zeros((1, 1, 3), dtype=np.uint8)
    if operation == 'set_icc_profile':
        handle = jpeg._TurboJPEG__init(TJINIT_COMPRESS)
        try:
            with pytest.raises(ValueError, match='16707346.*16707345.*255'):
                jpeg.set_icc_profile(handle, profile)
        finally:
            jpeg._TurboJPEG__destroy(handle)
    else:
        with pytest.raises(ValueError, match='16707346.*16707345.*255'):
            getattr(jpeg, operation)(source, icc_profile=profile)


@pytest.mark.parametrize('failure', ['memory_error', 'copy_error', 'native_error', 'empty'])
def test_icc_output_always_freed(icc_jpeg, image, monkeypatch, failure):
    jpeg = icc_jpeg
    data = jpeg.encode(image, icc_profile=bytes(132))
    native_get = jpeg._TurboJPEG__get_icc_profile
    native_free = jpeg._TurboJPEG__free
    native_destroy = jpeg._TurboJPEG__destroy
    allocated, freed, destroyed = [], [], []

    def get(handle, buffer, size):
        status = native_get(handle, buffer, size)
        allocated.append(cast(buffer, POINTER(c_void_p))[0])
        if failure == 'native_error':
            return -1
        if failure == 'empty':
            cast(size, POINTER(c_size_t))[0] = 0
        return status

    def free(buffer):
        freed.append(buffer.value)
        native_free(buffer)

    def destroy(handle):
        destroyed.append(handle)
        native_destroy(handle)

    def fail_copy(*args):
        if failure == 'memory_error':
            raise MemoryError('injected copy failure')
        raise RuntimeError('injected copy failure')

    monkeypatch.setattr(jpeg, '_TurboJPEG__get_icc_profile', get)
    monkeypatch.setattr(jpeg, '_TurboJPEG__free', free)
    monkeypatch.setattr(jpeg, '_TurboJPEG__destroy', destroy)
    monkeypatch.setattr(jpeg, '_TurboJPEG__copy_from_buffer', fail_copy)
    if failure == 'native_error':
        monkeypatch.setattr(jpeg, '_TurboJPEG__get_error_code', lambda h: TJERR_FATAL)
        monkeypatch.setattr(jpeg, '_TurboJPEG__get_error_string', lambda h: 'injected failure')
    error_type = {'memory_error': MemoryError, 'copy_error': RuntimeError,
                  'native_error': OSError}.get(failure)
    try:
        if error_type is None:
            assert jpeg.get_icc_profile(data) is None
        else:
            with pytest.raises(error_type, match='injected'):
                jpeg.get_icc_profile(data)
        assert len(allocated) == 1 and allocated == freed
        assert len(destroyed) == 1
    finally:
        # Keep a failing regression test from leaking its own native allocation.
        for address in allocated:
            if address and address not in freed:
                native_free(address)


def test_icc_header_warning_followed_by_fatal_is_not_missing_profile(
        icc_jpeg, image):
    data = icc_jpeg.encode(image, icc_profile=bytes(132))
    damaged = data[:2] + b'junk\xff\x02' + data[2:]
    with pytest.raises(OSError, match='Unsupported marker type'):
        icc_jpeg.get_icc_profile(damaged)


def test_icc_unknown_warning_is_not_missing_profile(icc_jpeg, image, monkeypatch):
    data = icc_jpeg.encode(image)
    monkeypatch.setattr(icc_jpeg, '_TurboJPEG__get_icc_profile', lambda *args: -1)
    monkeypatch.setattr(icc_jpeg, '_TurboJPEG__get_error_code', lambda h: TJERR_WARNING)
    monkeypatch.setattr(icc_jpeg, '_TurboJPEG__get_error_string', lambda h: 'unknown failure')
    with pytest.raises(OSError, match='unknown failure'):
        icc_jpeg.get_icc_profile(data)

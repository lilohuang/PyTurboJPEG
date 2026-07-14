"""Opt-in performance regression tests against a released baseline."""

import contextlib
import gc
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import types

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_BENCHMARKS = os.environ.get(
    'PYTURBOJPEG_RUN_BENCHMARKS') == '1'
BASELINE_REF_OVERRIDE = os.environ.get(
    'PYTURBOJPEG_BENCHMARK_BASELINE')
ROUNDS = int(os.environ.get(
    'PYTURBOJPEG_BENCHMARK_ROUNDS', '11'))
MAX_REGRESSION_PERCENT = float(os.environ.get(
    'PYTURBOJPEG_BENCHMARK_MAX_PERCENT', '5'))
MAX_REGRESSION_US = float(os.environ.get(
    'PYTURBOJPEG_BENCHMARK_MAX_US', '0.25'))
TARGET_BATCH_NS = 30_000_000

pytestmark = pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason=(
        'set PYTURBOJPEG_RUN_BENCHMARKS=1 to run performance tests'
    ),
)


def _load_module(name, source, filename):
    module = types.ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    exec(compile(source, filename, 'exec'), module.__dict__)
    return module


def _run_git(arguments, description):
    try:
        result = subprocess.run(
            ['git'] + arguments,
            cwd=str(REPOSITORY_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.fail(
            'Unable to {}: {}'.format(description, exc),
            pytrace=False,
        )
    return result.stdout


def _version_tuple(value):
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)', value)
    if match is None:
        pytest.fail(
            'Cannot determine release ordering from version {!r}'.format(
                value),
            pytrace=False,
        )
    return tuple(int(part) for part in match.groups())


def _find_baseline_ref(candidate_version):
    if BASELINE_REF_OVERRIDE:
        return BASELINE_REF_OVERRIDE

    candidate_tuple = _version_tuple(candidate_version)
    output = _run_git(
        ['tag', '--list', 'v[0-9]*'],
        'list release tags for benchmark baseline selection',
    )
    releases = []
    for tag in output.splitlines():
        match = re.fullmatch(r'v(\d+)\.(\d+)\.(\d+)', tag)
        if match is None:
            continue
        version = tuple(int(part) for part in match.groups())
        if version < candidate_tuple:
            releases.append((version, tag))
    if not releases:
        pytest.fail(
            'No release tag older than {} is available'.format(
                candidate_version),
            pytrace=False,
        )
    return max(releases)[1]


def _read_baseline_source(baseline_ref):
    return _run_git(
        ['show', '{}:turbojpeg.py'.format(baseline_ref)],
        'read benchmark baseline {!r}'.format(baseline_ref),
    )


def _calibrate(function):
    iterations = 1
    while True:
        started = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        elapsed = time.perf_counter_ns() - started
        if elapsed >= TARGET_BATCH_NS or iterations >= 32_768:
            return iterations
        iterations *= 2


def _measure_pair(baseline_function, candidate_function):
    iterations = _calibrate(baseline_function)
    warmup_iterations = max(3, min(100, iterations // 4))
    for _ in range(warmup_iterations):
        baseline_function()
        candidate_function()

    samples = {'baseline': [], 'candidate': []}
    functions = {
        'baseline': baseline_function,
        'candidate': candidate_function,
    }
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for round_index in range(ROUNDS):
            order = ('baseline', 'candidate')
            if round_index % 2:
                order = tuple(reversed(order))
            for name in order:
                function = functions[name]
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    function()
                elapsed = time.perf_counter_ns() - started
                samples[name].append(elapsed / iterations / 1_000)
    finally:
        if gc_was_enabled:
            gc.enable()

    return (
        statistics.median(samples['baseline']),
        statistics.median(samples['candidate']),
        iterations,
    )


@contextlib.contextmanager
def _single_cpu():
    original_affinity = None
    if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
        try:
            original_affinity = os.sched_getaffinity(0)
            if original_affinity:
                os.sched_setaffinity(0, {min(original_affinity)})
        except OSError:
            original_affinity = None
    try:
        yield
    finally:
        if original_affinity is not None:
            try:
                os.sched_setaffinity(0, original_affinity)
            except OSError:
                pass


def _operation(jpeg, operation, image, encoded):
    if operation == 'encode':
        return lambda: jpeg.encode(image)
    if operation == 'decode':
        return lambda: jpeg.decode(encoded)
    if operation == 'decode_header':
        return lambda: jpeg.decode_header(encoded)
    if operation == 'decode_to_yuv':
        return lambda: jpeg.decode_to_yuv(encoded)
    raise AssertionError('Unknown benchmark operation: {}'.format(operation))


def _check_operation(baseline_function, candidate_function, operation):
    baseline_result = baseline_function()
    candidate_result = candidate_function()
    if operation == 'encode':
        assert candidate_result == baseline_result
    elif operation == 'decode':
        np.testing.assert_array_equal(candidate_result, baseline_result)
    elif operation == 'decode_header':
        assert candidate_result == baseline_result
    else:
        np.testing.assert_array_equal(
            candidate_result[0], baseline_result[0])
        assert candidate_result[1] == baseline_result[1]


def test_default_operations_do_not_regress_against_previous_release():
    """Compare common default operations with the previous release."""
    assert ROUNDS >= 3
    assert MAX_REGRESSION_PERCENT >= 0
    assert MAX_REGRESSION_US >= 0

    candidate_module = _load_module(
        '_pyturbojpeg_benchmark_candidate',
        (REPOSITORY_ROOT / 'turbojpeg.py').read_text(encoding='utf-8'),
        str(REPOSITORY_ROOT / 'turbojpeg.py'),
    )
    baseline_ref = _find_baseline_ref(candidate_module.__version__)
    baseline_module = _load_module(
        '_pyturbojpeg_benchmark_baseline',
        _read_baseline_source(baseline_ref),
        '{}:turbojpeg.py'.format(baseline_ref),
    )
    lib_path = os.environ.get('TURBOJPEG_LIB_PATH') or None
    baseline_jpeg = baseline_module.TurboJPEG(lib_path=lib_path)
    candidate_jpeg = candidate_module.TurboJPEG(lib_path=lib_path)

    random = np.random.RandomState(2_500)
    image_sizes = ((8, 8), (32, 32), (720, 1280))
    operations = ('encode', 'decode', 'decode_header', 'decode_to_yuv')
    failures = []

    print(
        '\nComparing PyTurboJPEG {} with {} ({})'.format(
            candidate_module.__version__,
            baseline_module.__version__,
            baseline_ref,
        )
    )
    print(
        '{:<24} {:>12} {:>12} {:>10} {:>8}'.format(
            'operation', 'baseline us', 'current us', 'change', 'result')
    )

    with _single_cpu():
        for height, width in image_sizes:
            image = random.randint(
                0, 256, (height, width, 3)).astype(np.uint8)
            encoded = baseline_jpeg.encode(image)
            for operation in operations:
                baseline_function = _operation(
                    baseline_jpeg, operation, image, encoded)
                candidate_function = _operation(
                    candidate_jpeg, operation, image, encoded)
                _check_operation(
                    baseline_function, candidate_function, operation)
                baseline_us, candidate_us, iterations = _measure_pair(
                    baseline_function, candidate_function)
                change_percent = (
                    candidate_us / baseline_us - 1) * 100
                limit_us = max(
                    baseline_us * (
                        1 + MAX_REGRESSION_PERCENT / 100),
                    baseline_us + MAX_REGRESSION_US,
                )
                passed = candidate_us <= limit_us
                name = '{}_{}x{}'.format(operation, width, height)
                print(
                    '{:<24} {:>12.3f} {:>12.3f} {:>+9.1f}% {:>8}'.format(
                        name,
                        baseline_us,
                        candidate_us,
                        change_percent,
                        'PASS' if passed else 'FAIL',
                    )
                )
                if not passed:
                    failures.append(
                        '{}: {:.3f} us exceeded {:.3f} us '
                        '(baseline {:.3f} us, {} iterations)'.format(
                            name,
                            candidate_us,
                            limit_us,
                            baseline_us,
                            iterations,
                        )
                    )

    assert not failures, 'Performance regressions:\n{}'.format(
        '\n'.join(failures))

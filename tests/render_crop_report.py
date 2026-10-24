"""Compare a trusted Git revision with this checkout and render crop evidence.

Run from the repository root; Pillow is needed only for this optional report.
The ordinary unit tests do not require Pillow, Git history, or saved images.
"""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crop_test_data import BATCH_CASES, CROP_CASES, SAMPLING, coordinate_image  # noqa: E402
from turbojpeg import (  # noqa: E402
    TurboJPEG, TJPF_RGB, TJPF_GRAY, TJCS_RGB, TJSAMP_444, TJSAMP_420,
    TJFLAG_FASTUPSAMPLE,
)


def load_baseline(ref):
    revision = subprocess.check_output(
        ['git', 'rev-parse', '--verify', ref + '^{commit}'], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(
        ['git', 'show', revision + ':turbojpeg.py'], cwd=ROOT, text=True)
    module = types.ModuleType('_crop_baseline')
    sys.modules[module.__name__] = module
    exec(compile(source, '<baseline turbojpeg.py>', 'exec'), module.__dict__)
    return revision, module


def attempt(function, *args, **kwargs):
    try:
        return function(*args, **kwargs), None
    except (ValueError, OSError) as error:
        return None, '{}: {}'.format(type(error).__name__, error)


def crop_pixels(pixels, rectangle):
    x, y, width, height = rectangle
    return pixels[y:y + height, x:x + width]


def compare_pair(before, after, expected, decode, pixel_format):
    old_data, old_error = before
    data, error = after
    if expected is None:
        assert error and error.startswith('ValueError:')
        assert old_error and old_error.startswith('ValueError:'), (old_error, error)
        return {'same_rejection': 'ValueError', 'before_error': old_error, 'after_error': error}
    assert old_error is None and error is None, (old_error, error)
    assert old_data == data, 'Previously valid crop changed its JPEG bytes'
    actual = decode(data, pixel_format=pixel_format, flags=TJFLAG_FASTUPSAMPLE)
    np.testing.assert_array_equal(actual, expected)
    return {'jpeg_bytes_identical': True, 'pixels_match_expected': True}


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def render(output, name, description, source, requested, rectangle, expected, before, after):
    """Four panels with independent expected pixels and enlarged source context."""
    # Keep actionable errors readable even when they explain several remedies.
    error_lines = max(len(textwrap.wrap(content, 32))
                      for content in (before, after, '') if isinstance(content, str))
    panel_bottom = max(466, 182 + error_lines * 25)
    canvas = Image.new('RGB', (1600, panel_bottom + 74), '#101827')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), name.replace('_', ' '), font=font(24), fill='white')
    draw.text((20, 48), description, font=font(17), fill='#ced9ea')
    draw.text((20, 78), 'Requested: {}    Expected source rectangle: {}'.format(
        requested, rectangle), font=font(17), fill='#ced9ea')
    source_image = Image.fromarray(source)
    left = top = 0
    if rectangle is not None:
        x, y, w, h = rectangle
        rx, ry, rw, rh = requested
        left, top = max(0, min(x, rx) - 8), max(0, min(y, ry) - 8)
        right = min(source.shape[1], max(x + w, rx + rw) + 8)
        bottom = min(source.shape[0], max(y + h, ry + rh) + 8)
        source_image = source_image.crop((left, top, right, bottom))
    elif expected is None:
        rx, ry, rw, rh = requested
        left, top = max(0, rx - 8), max(0, ry - 8)
        source_image = source_image.crop((left, top,
                                         min(source.shape[1], rx + rw + 8),
                                         min(source.shape[0], ry + rh + 8)))
    source_panel = ('Source context ({}, {})'.format(left, top), source_image)
    panels = [source_panel, ('Expected pixels', expected),
              ('Before', before), ('After', after)]
    for index, (label, content) in enumerate(panels):
        px = index * 400 + 16
        draw.text((px, 118), label, font=font(20), fill='white')
        draw.rectangle((px, 152, px + 368, panel_bottom), fill='#253149')
        if isinstance(content, str) or content is None:
            message = content or 'ValueError: alignment leaves no pixels'
            draw.multiline_text((px + 12, 170), textwrap.fill(message, 32),
                                font=font(17), fill='#ffb9ae', spacing=8)
            continue
        picture = content if isinstance(content, Image.Image) else Image.fromarray(content)
        scale = min(350 / picture.width, 296 / picture.height, 20)
        width, height = round(picture.width * scale), round(picture.height * scale)
        resized = picture.resize((width, height), Image.Resampling.NEAREST)
        if index == 0:
            annotation = ImageDraw.Draw(resized)
            for rect, color, line_width in ((rectangle, '#00edff', 3),
                                             (requested, '#ffba00', 2)):
                if rect is not None:
                    x, y, w, h = rect
                    annotation.rectangle(((x - left) * scale, (y - top) * scale,
                                          (x + w - left) * scale - 1,
                                          (y + h - top) * scale - 1),
                                         outline=color, width=line_width)
        canvas.paste(resized, (px + 8, 160))
        draw.text((px, panel_bottom + 10), '{} x {} pixels; {:.1f}x view'.format(
            picture.width, picture.height, scale), font=font(16), fill='#ced9ea')
    draw.text((20, panel_bottom + 47), 'ORANGE = requested bounds    CYAN = expected source bounds'
              '    All previews use nearest-neighbor zoom.', font=font(15), fill='white')
    filename = name + '.png'
    canvas.save(output / filename)
    return filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', required=True, help='Trusted Git revision to execute')
    parser.add_argument('--library', default=os.environ.get('TURBOJPEG_LIB_PATH'))
    parser.add_argument('--output', type=Path, default=ROOT / 'test_output/crop_report')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision, baseline_module = load_baseline(args.baseline_ref)
    before = baseline_module.TurboJPEG(args.library)
    after = TurboJPEG(args.library)
    records, figures, changes = [], [], []
    shown = {'tiny_expand', 'tiny_preserve_empty', 'cross_mcu_preserve',
             'interior_expand', 'interior_preserve', 'clamp_edges_expand',
             'clamp_edges_preserve'}
    for bits in (8, 12):
        encode = after.encode if bits == 8 else after.encode_12bit
        decode = after.decode if bits == 8 else after.decode_12bit
        for name, subsample, mcu in SAMPLING + [('rgb', TJSAMP_444, (8, 8))]:
            options = {'jpeg_colorspace': TJCS_RGB} if name == 'rgb' else {}
            encoded = encode(coordinate_image(bits=bits), quality=100,
                             pixel_format=TJPF_RGB, jpeg_subsample=subsample, **options)
            for gray in ((False,) if name == 'rgb' else (False, True)):
                pixel_format = TJPF_GRAY if gray else TJPF_RGB
                pixels = decode(encoded, pixel_format=pixel_format, flags=TJFLAG_FASTUPSAMPLE)
                prefix = '{}bit-{}-gray{}'.format(bits, name, gray)
                for case in CROP_CASES:
                    rect = case.expected[mcu]
                    expected = None if rect is None else crop_pixels(pixels, rect)
                    old = attempt(before.crop, encoded, *case.requested,
                                  preserve=case.preserve, gray=gray)
                    new = attempt(after.crop, encoded, *case.requested,
                                  preserve=case.preserve, gray=gray)
                    record = compare_pair(old, new, expected, decode, pixel_format)
                    records.append(dict(case=prefix + '-' + case.name, **record))
                    if bits == 8 and name == '420' and not gray and case.name in shown:
                        previews = [error if error else decode(data, pixel_format=TJPF_RGB,
                                    flags=TJFLAG_FASTUPSAMPLE) for data, error in (old, new)]
                        figures.append(render(args.output, 'crop_' + case.name,
                            '4:2:0 / 16x16 iMCU; preserve={}; {}'.format(
                                case.preserve, 'same rejection' if rect is None else 'JPEG bytes identical'),
                            pixels, case.requested, rect, expected, *previews))
                parameters = [item[0] for item in BATCH_CASES]
                old_batch = before.crop_multiple(encoded, parameters, gray=gray)
                new_batch = after.crop_multiple(encoded, parameters, gray=gray)
                assert len(old_batch) == len(new_batch) == len(BATCH_CASES)
                for index, (old, new, (_, rect)) in enumerate(zip(old_batch, new_batch, BATCH_CASES)):
                    record = compare_pair((old, None), (new, None),
                                          crop_pixels(pixels, rect), decode, pixel_format)
                    records.append(dict(case=prefix + '-batch-' + str(index), **record))
                for x, y in ((1, 0), (0, 1), (1, 1)):
                    old = attempt(before.crop_multiple, encoded, [(x, y, 2, 2)], gray=gray)
                    new = attempt(after.crop_multiple, encoded, [(x, y, 2, 2)], gray=gray)
                    record = compare_pair(old, new, None, decode, pixel_format)
                    records.append(dict(case=prefix + '-unaligned-{},{}'.format(x, y), **record))

    # These are intentional behavior changes, separately from compatibility.
    for name, colorspace, width, height, region in (
        ('rgb_white_background', TJCS_RGB, 256, 192, (0, 0, 320, 256)),
        ('rgb_offset_background', TJCS_RGB, 256, 192, (32, 32, 320, 256)),
        ('ycbcr_offset_background', None, 256, 192, (32, 32, 320, 256)),
        ('large_centered_placement', TJCS_RGB, 64, 64, (128, 128, 320, 320)),
        ('partial_edge_extension_rejected', None, 259, 197, (0, 0, 320, 256)),
    ):
        encoded = after.encode(coordinate_image(width, height), quality=100,
                               pixel_format=TJPF_RGB, jpeg_colorspace=colorspace,
                               jpeg_subsample=TJSAMP_444 if colorspace == TJCS_RGB else TJSAMP_420)
        pixels = after.decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
        if width == 259:
            Image.fromarray(coordinate_image()).save(args.output / 'source_original.png')
            Image.fromarray(pixels).save(args.output / 'source_decoded.png')
            (args.output / 'source.jpg').write_bytes(encoded)
        x, y, out_width, out_height = region
        expected = np.full((out_height, out_width, 3), 255, dtype=np.uint8)
        expected[y:y + height, x:x + width] = pixels
        previews, errors, mismatches = [], [], []
        for implementation in (before, after):
            data, error = attempt(implementation.crop_multiple, encoded, [region])
            errors.append(error)
            if error:
                previews.append(error)
                mismatches.append(None)
            else:
                result = after.decode(data[0], pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
                previews.append(result)
                mismatches.append(int(np.count_nonzero(np.any(result != expected, axis=2))))
        if width == 259:
            assert errors[1] and errors[1].startswith('ValueError:') and 'partial iMCU' in errors[1]
            description = 'Partial source iMCUs: requested canvas shown for reference; new code rejects this extension.'
        else:
            assert errors[1] is None and mismatches[1] == 0
            description = 'Source placement and white canvas; mismatched pixels before={}, after={}'.format(*mismatches)
        changes.append(dict(case=name, errors=errors, mismatched_pixels=mismatches))
        figures.append(render(args.output, name, description, pixels, region,
                              None, expected, *previews))

    # New batch alignment modes have explicit, independent expected rectangles.
    encoded = after.encode(coordinate_image(64, 64), quality=100,
                           pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_420)
    pixels = after.decode(encoded, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
    requested = (10, 10, 20, 20)
    for align, rectangle in (('expand', (0, 0, 30, 30)), ('shrink', (16, 16, 14, 14))):
        old_data, old_error = attempt(before.crop_multiple, encoded, [requested])
        result = after.crop_multiple(encoded, [requested], align=align)[0]
        actual = after.decode(result, pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
        expected = crop_pixels(pixels, rectangle)
        np.testing.assert_array_equal(actual, expected)
        old_preview = old_error if old_error else after.decode(
            old_data[0], pixel_format=TJPF_RGB, flags=TJFLAG_FASTUPSAMPLE)
        name = 'batch_alignment_' + align
        changes.append(dict(case=name, align=align, expected_rectangle=rectangle,
                            before_error=old_error, pixels_match_expected=True))
        figures.append(render(args.output, name,
            "Before: default batch behavior. After: align={!r}, matching crop(preserve={}).".format(
                align, align == 'shrink'),
            pixels, requested, rectangle, expected, old_preview, actual))

    summary = {
        'baseline_commit': revision,
        'candidate_sha256': hashlib.sha256((ROOT / 'turbojpeg.py').read_bytes()).hexdigest(),
        'native_library': args.library or 'auto-discovered',
        'comparison_count': len(records),
        'identical_jpeg_count': sum('jpeg_bytes_identical' in record for record in records),
        'identical_rejection_count': sum('same_rejection' in record for record in records),
        'intentional_changes': changes,
        'comparisons': records,
    }
    (args.output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    page = ['<!doctype html><meta charset="utf-8"><title>Crop comparison</title>',
            '<style>body{font:18px sans-serif;background:#101827;color:#eee;margin:24px}'
            'img{max-width:100%;height:auto}a{color:#80d8ff}p{max-width:1000px}</style>',
            '<h1>Crop comparison: before / expected / after</h1>',
            '<p>Baseline: <code>{}</code><br>Native library: <code>{}</code></p>'.format(
                revision, html.escape(summary['native_library'])),
            '<p>{} normal comparisons: {} JPEG outputs byte-identical; {} requests rejected '
            'with ValueError in both versions (messages may differ). '
            'Every successful crop also matches the expected source pixels.</p>'.format(
                len(records), summary['identical_jpeg_count'], summary['identical_rejection_count']),
            '<p>The test grid labels absolute x,y coordinates every 32 pixels. Orange marks '
            'the requested rectangle; cyan marks the expected source rectangle. Tiny crops '
            'are enlarged. Expected pixels come from the decoded source JPEG, with fast '
            'chroma upsampling to avoid edge interpolation differences.</p>',
            '<p>Background extension examples show intentional fixes. For the rejected '
            'partial-edge extension, the expected panel shows the requested canvas only; '
            'that operation is unsupported. Batch alignment examples compare the prior '
            'default with explicitly selecting the new expand/shrink modes.</p>',
            '<p><a href="source_decoded.png">Decoded test grid</a> | '
            '<a href="source.jpg">Source JPEG</a> | <a href="results.json">Full results</a></p>']
    page.extend('<p><a href="{0}"><img src="{0}" alt="{1}"></a></p>'.format(
        filename, html.escape(filename)) for filename in figures)
    (args.output / 'index.html').write_text('\n'.join(page) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'comparisons'}, indent=2))
    print('Report:', args.output / 'index.html')


if __name__ == '__main__':
    main()

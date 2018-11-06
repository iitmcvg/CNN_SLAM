"""
The numpngw module defines two functions and a class:

* write_png(...) writes a numpy array to a PNG file.
* write_apng(...) writes a sequence of arrays to an animated PNG file.
* AnimatedPNGWriter is a class that can be used with Matplotlib animations.

-----
Copyright (c) 2015, Warren Weckesser
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import (division as _division,
                        print_function as _print_function)

import sys as _sys
import contextlib as _contextlib
from io import BytesIO as _BytesIO
import time as _time
import struct as _struct
import zlib as _zlib
from fractions import Fraction as _Fraction
import numpy as _np


__all__ = ['write_png', 'write_apng', 'AnimatedPNGWriter']

__version__ = "0.0.8.dev1"

_PY3 = _sys.version_info > (3,)
if _PY3:
    def _bord(c):
        return c
else:
    _bord = ord


def _software_text():
    software = ("numpngw (version %s), "
                "https://github.com/WarrenWeckesser/numpngw" % __version__)
    return software


def _filter0(row, prev_row):
    return row


def _filter0inv(frow, prev_row):
    return frow


def _filter1(row, prev_row):
    d = _np.zeros_like(row)
    d[1:] = _np.diff(row, axis=0)
    d[0] = row[0]
    return d


def _filter1inv(frow, prev_row):
    return frow.cumsum(axis=0, dtype=_np.uint64).astype(_np.uint8)


def _filter2(row, prev_row):
    d = row - prev_row
    return d


def _filter2inv(frow, prev_row):
    return frow + prev_row


def _filter3(row, prev_row):
    a = _np.zeros_like(row, dtype=_np.int64)
    a[1:] = row[:-1]
    c = ((a + prev_row) // 2).astype(row.dtype)
    d = row - c
    return d


def _filter3inv(frow, prev_row):
    # Slow python loop, but currently this is only used for testing.
    row = _np.empty_like(frow)
    for k in range(len(frow)):
        if k == 0:
            row[k] = frow[k] + (prev_row[k] // 2)
        else:
            row[k] = frow[k] + (row[k-1].astype(int) +
                                prev_row[k].astype(int)) // 2
    return row


def _filter4(row, prev_row):
    """Paeth filter."""
    # Create a, b and c.
    a = _np.zeros_like(row, dtype=_np.int64)
    a[1:] = row[:-1]
    b = prev_row.astype(_np.int64)
    c = _np.zeros_like(b)
    c[1:] = b[:-1]

    p = a + b - c
    pa = _np.abs(p - a)
    pb = _np.abs(p - b)
    pc = _np.abs(p - c)
    y = _np.where((pa <= pb) & (pa <= pc), a, _np.where(pb <= pc, b, c))
    pr = y.astype(_np.uint8)
    d = row - pr
    return d


def _filter4inv(frow, prev_row):
    # Slow python loop, but currently this is only used for testing.
    row = _np.empty_like(frow)
    for k in range(len(frow)):
        if k == 0:
            ra = _np.zeros_like(frow[k])
            rc = _np.zeros_like(frow[k])
        else:
            ra = row[k-1].astype(int)
            rc = prev_row[k-1].astype(int)
        rb = prev_row[k].astype(int)
        p = ra + rb - rc
        pa = _np.abs(p - ra)
        pb = _np.abs(p - rb)
        pc = _np.abs(p - rc)
        y = _np.where((pa <= pb) & (pa <= pc), ra, _np.where(pb <= pc, rb, rc))
        row[k] = frow[k] + y
    return row


def _interlace_passes(img):
    """
    Return the subimages of img that make up the seven Adam7 interlace passes.
    """
    pass1 = img[::8, ::8]
    pass2 = img[::8, 4::8]
    pass3 = img[4::8, ::4]
    pass4 = img[::4, 2::4]
    pass5 = img[2::4, ::2]
    pass6 = img[::2, 1::2]
    pass7 = img[1::2, :]
    return (pass1, pass2, pass3, pass4, pass5, pass6, pass7)


def _create_stream(a, filter_type=None):
    """
    Convert the data in `a` into a python string.

    `a` is must to be a 2D or 3D array of unsigned 8- or 16-bit
    integers.

    The string is formatted as the "scan lines" of the array.
    """
    filters = [_filter0, _filter1, _filter2, _filter3, _filter4]

    if filter_type is None:
        filter_type = "heuristic"
    allowed_filter_types = [0, 1, 2, 3, 4, "heuristic"]
    if filter_type not in allowed_filter_types:
        raise ValueError('filter_type must be one of %r' %
                         (allowed_filter_types,))

    if a.ndim == 2:
        # Gray scale.  Add a trivial third dimension.
        a = a[:, :, _np.newaxis]
    lines = []
    prev_row = _np.zeros_like(a[0]).view(_np.uint8)
    for row in a:
        # Convert the row to big-endian (i.e. network byte order).
        row_be = row.astype('>' + row.dtype.str[1:]).view(_np.uint8)
        if filter_type == "heuristic":
            filtered_rows = [filt(row_be, prev_row) for filt in filters]
            values = _np.array([_np.abs(fr.view(_np.int8).astype(_np.int)).sum()
                                for fr in filtered_rows])
            ftype = values.argmin()
            # Create the string, with the filter type prepended.
            lines.append(chr(ftype).encode('ascii') +
                         filtered_rows[ftype].tostring())
        else:
            filtered_row = filters[filter_type](row_be, prev_row)
            lines.append(chr(filter_type).encode('ascii') +
                         filtered_row.tostring())
        prev_row = row_be
    stream = b''.join(lines)
    return stream


def _write_chunk(f, chunk_type, chunk_data):
    """
    Write a chunk to the file `f`.  This function wraps the chunk_type and
    chunk_data with the length and CRC field, and writes the result to `f`.
    """
    content = chunk_type + chunk_data
    length = _struct.pack("!I", len(chunk_data))
    crc = _struct.pack("!I", _zlib.crc32(content) & 0xFFFFFFFF)
    f.write(length + content + crc)


def _write_ihdr(f, width, height, nbits, color_type, interlace):
    """Write an IHDR chunk to `f`."""
    fmt = "!IIBBBBB"
    chunk_data = _struct.pack(fmt, width, height, nbits, color_type, 0, 0,
                              interlace)
    _write_chunk(f, b"IHDR", chunk_data)


def _write_text(f, keyword, text_string):
    """Write a tEXt chunk to `f`.

    keyword and test_string are expected to be bytes (not unicode).
    They must already be validated.
    """
    data = keyword + b'\0' + text_string
    _write_chunk(f, b'tEXt', data)


def _write_time(f, timestamp):
    """Write a tIME chunk to `f`."""
    chunk_data = _struct.pack('!HBBBBB', *timestamp)
    _write_chunk(f, b'tIME', chunk_data)


def _write_gama(f, gamma):
    """Write a gAMA chunk to `f`."""
    gama = int(gamma*100000 + 0.5)
    chunk_data = _struct.pack('!I', gama)
    _write_chunk(f, b'gAMA', chunk_data)


def _write_plte(f, palette):
    _write_chunk(f, b"PLTE", palette.tostring())


def _write_trns(f, trans):
    trans_be = trans.astype('>' + trans.dtype.str[1:])
    _write_chunk(f, b"tRNS", trans_be.tostring())


def _write_bkgd(f, color, color_type):
    """
    Write bKGD chunk to `f`.

    * If `color_type` is 0 or 4, `color` must be an integer.
    * If `color_type` is 2 or 6, `color` must be a sequence of three
      integers (RGB values).
    * If `color_type` is 3, `color` must be an integer that is less than
      the number of colors in the palette.
    """
    if color_type == 0 or color_type == 4:
        chunk_data = _struct.pack("!H", color)
    elif color_type == 2 or color_type == 6:
        chunk_data = _struct.pack("!HHH", *color)
    elif color_type == 3:
        chunk_data = _struct.pack("B", color)
    else:
        raise ValueError("invalid chunk_type %r" % (color_type,))
    _write_chunk(f, b"bKGD", chunk_data)


def _write_phys(f, phys):
    """Write a pHYs chunk to `f`."""
    chunk_data = _struct.pack("!IIB", *phys)
    _write_chunk(f, b"pHYs", chunk_data)


def _write_idat(f, data):
    """Write an IDAT chunk to `f`."""
    _write_chunk(f, b"IDAT", data)


def _write_iend(f):
    """Write an IEND chunk to `f`."""
    _write_chunk(f, b"IEND", b"")


def _write_actl(f, num_frames, num_plays):
    """Write an acTL chunk to `f`."""
    if num_frames < 1:
        raise ValueError("Attempt to create acTL chunk with num_frames (%i) "
                         "less than 1." % (num_frames,))
    chunk_data = _struct.pack("!II", num_frames, num_plays)
    _write_chunk(f, b"acTL", chunk_data)


def _write_fctl(f, sequence_number, width, height, x_offset, y_offset,
                delay_num, delay_den, dispose_op=0, blend_op=0):
    """Write an fcTL chunk to `f`."""
    if width < 1:
        raise ValueError("width must be greater than 0")
    if height < 1:
        raise ValueError("heigt must be greater than 0")
    if x_offset < 0:
        raise ValueError("x_offset must be nonnegative")
    if y_offset < 0:
        raise ValueError("y_offset must be nonnegative")

    fmt = "!IIIIIHHBB"
    chunk_data = _struct.pack(fmt, sequence_number, width, height,
                              x_offset, y_offset, delay_num, delay_den,
                              dispose_op, blend_op)
    _write_chunk(f, b"fcTL", chunk_data)


def _write_fdat(f, sequence_number, data):
    """Write an fdAT chunk to `f`."""
    seq = _struct.pack("!I", sequence_number)
    _write_chunk(f, b"fdAT", seq + data)


def _write_data(f, a, bitdepth, max_chunk_len=None, sequence_number=None,
                filter_type=None, interlace=0):
    """
    Write the image data in the array `a` to the file, using IDAT chunks
    if sequence_number is None and fdAT chunks otherwise.

    `f` must be a writable file object.
    `a` must be a numpy array to be written to the file `f`.
    If `sequence_number` is None, 'IDAT' chunks are written.
    If `sequence_number` is not None, `fdAT` chunks are written.

    `interlace` must be 0 or 1.  It determines the interlace method.
    0 mean no interlacing; 1 means Adam7 interlacing.

    Returns the number of chunks written to the file.

    `filter_type` is passed on to _create_stream().
    """
    if interlace == 1:
        passes = _interlace_passes(a)
    else:
        passes = [a]

    if bitdepth is not None and bitdepth < 8:
        passes = [_pack(aa, bitdepth) for aa in passes]

    if filter_type == "auto":
        filter_types = [0, 1, 2, 3, 4, "heuristic"]
    else:
        filter_types = [filter_type]

    zstream = None
    for filter_type in filter_types:
        stream = b''
        for a in passes:
            if a.size > 0:
                stream += _create_stream(a, filter_type=filter_type)
        z = _zlib.compress(stream)
        if zstream is None or len(z) < len(zstream):
            zstream = z

    # zstream is a string containing the packed, compressed version of the
    # data from the array `a`.  This will be written to the file in one or
    # more IDAT or fdAT chunks.

    if max_chunk_len is None:
        # Put the whole thing in one chunk.
        max_chunk_len = len(zstream)
    elif max_chunk_len < 1:
        raise ValueError("max_chunk_len must be at least 1.")

    num_data_chunks = (len(zstream) + max_chunk_len - 1) // max_chunk_len
    for k in range(num_data_chunks):
        start = k * max_chunk_len
        end = min(start + max_chunk_len, len(zstream))
        data = zstream[start:end]
        if sequence_number is None:
            _write_idat(f, data)
        else:
            _write_fdat(f, sequence_number, data)
            sequence_number += 1

    return num_data_chunks


def _encode_latin1(s):
    if _PY3:
        unicode_type = str
    else:
        unicode_type = unicode
    if isinstance(s, unicode_type):
        s = s.encode('latin-1')
    return s


def _validate_text(text_list):
    if text_list is None:
        text_list = []

    creation_time = _encode_latin1(_time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  _time.gmtime()))
    software = _encode_latin1(_software_text())

    text_list = [(_encode_latin1(keyword), _encode_latin1(text))
                 for keyword, text in text_list]

    keywords = [keyword for keyword, text in text_list]
    if b"Creation Time" not in keywords:
        text_list.append((b"Creation Time", creation_time))
    if b"Software" not in keywords:
        text_list.append((b"Software", software))

    validated_text_list = []
    for keyword, text_string in text_list:
        if text_string is None:
            # Drop elements where the text string is None.
            continue

        # Validate the keyword.
        if not (0 < len(keyword) < 80):
            raise ValueError("length of keyword must greater than 0 and less "
                             "than 80.")
        kw_check = all([(31 < _bord(c) < 127) or (160 < _bord(c) < 256)
                        for c in keyword])
        if not kw_check:
            raise ValueError("keyword %r contains non-printable characters." %
                             (keyword,))
        if keyword.startswith(b' '):
            raise ValueError("keyword %r begins with a space." % (keyword,))
        if keyword.endswith(b' '):
            raise ValueError("keyword %r ends with a space." % (keyword,))
        if b'  ' in keyword:
            raise ValueError("keyword %r contains consecutive spaces." %
                             (keyword,))

        # Validate the text string.
        if b'\0' in text_string:
            raise ValueError("text_string contains a null character.")

        validated_text_list.append((keyword, text_string))

    return validated_text_list


def _palettize(a):
    # `a` must be a numpy array with dtype `np.uint8` and shape (m, n, 3) or
    # (m, n, 4).
    a = _np.ascontiguousarray(a)
    depth = a.shape[-1]
    dt = ','.join(['u1'] * depth)
    b = a.view(dt).reshape(a.shape[:-1])
    colors, inv = _np.unique(b, return_inverse=True)
    index = inv.astype(_np.uint8).reshape(a.shape[:-1])
    # palette is the RGB values of the unique RGBA colors.
    palette = colors.view(_np.uint8).reshape(-1, depth)[:, :3]
    if depth == 3:
        trans = None
    else:
        # trans is the 1-d array of alpha values of the unique RGBA colors.
        # trans is the same length as `palette`.
        trans = colors['f3']
    return index, palette, trans


def _palettize_seq(seq):
    """"
    seq must be a sequence of 3-d numpy arrays with dtype np.uint8,
    all with the same depth (i.e. the same length of the third dimension).
    """
    # Call np.unique for each array in seq.  Each array is viewed as a
    # 2-d structured array of colors.
    depth = seq[0].shape[-1]
    dt = ','.join(['u1'] * depth)
    result = [_np.unique(a.view(dt).reshape(a.shape[:-1]), return_inverse=True)
              for a in seq]

    # `sizes` is the number of unique colors found in each array.
    sizes = [len(r[0]) for r in result]

    # Combine all the colors found in each array to get the overall
    # set of unique colors.
    combined = _np.concatenate([r[0] for r in result])
    colors, inv = _np.unique(combined, return_inverse=True)

    offsets = _np.cumsum(_np.r_[0, sizes[:-1]])
    invs = [r[1].reshape(a.shape[:2]) for r, a in zip(result, seq)]

    # The sequence index_seq holds the converted arrays.  The values
    # in these arrays are indices into `palette`.  Note that if
    # len(palette) > 256, the conversion to np.uint8 will cause
    # some values in the arrays in `index_seq` to wrap around.
    # The caller must check the len(palette) to determine if this
    # has happened.
    index_seq = [inv[o:o+s][i].astype(_np.uint8)
                 for i, o, s in zip(invs, offsets, sizes)]
    palette = colors.view(_np.uint8).reshape(-1, depth)[:, :3]
    if depth == 3:
        trans = None
    else:
        # trans is the 1-d array of alpha values of the unique RGBA colors.
        # trans is the same length as `palette`.
        trans = colors['f3']
    return index_seq, palette, trans


def _pack(a, bitdepth):
    """
    Pack the values in `a` into bitfields of a smaller array.

    `a` must be a 2-d numpy array with dtype `np.uint8`
    bitdepth must be either 1, 2, 4 or 8.
    (bitdepth=8 is a trivial case, for which the return value is simply `a`.)
    """
    if a.dtype != _np.uint8:
        raise ValueError('Input array must have dtype uint8')
    if a.ndim != 2:
        raise ValueError('Input array must be two dimensional')

    if bitdepth == 8:
        return a

    ncols, rembits = divmod(a.shape[1]*bitdepth, 8)
    if rembits > 0:
        ncols += 1
    b = _np.zeros((a.shape[0], ncols), dtype=_np.uint8)
    for row in range(a.shape[0]):
        bcol = 0
        pos = 8
        for col in range(a.shape[1]):
            val = (2**bitdepth - 1) & a[row, col]
            pos -= bitdepth
            if pos < 0:
                bcol += 1
                pos = 8 - bitdepth
            b[row, bcol] |= (val << pos)

    return b


def _unpack(p, bitdepth, width):
    powers = _np.arange(bitdepth-1, -1, -1)
    up = _np.unpackbits(p).reshape(p.shape[0], -1, bitdepth).dot(2**powers)
    a = up[:, :width]
    return a


def _validate_array(a):
    if a.ndim != 2:
        if a.ndim != 3 or a.shape[2] > 4 or a.shape[2] == 0:
            raise ValueError("array must be 2D, or 3D with shape "
                             "(m, n, d) with 1 <= d <= 4.")
    itemsize = a.dtype.itemsize
    if not _np.issubdtype(a.dtype, _np.unsignedinteger) or itemsize > 2:
        raise ValueError("array must be an array of 8- or 16-bit "
                         "unsigned integers")


def _get_color_type(a, use_palette):
    if a.ndim == 2:
        color_type = 0
    else:
        depth = a.shape[2]
        if depth == 1:
            # Grayscale
            color_type = 0
        elif depth == 2:
            # Grayscale and alpha
            color_type = 4
        elif depth == 3:
            # RGB
            if a.dtype == _np.uint8 and use_palette:
                # Indexed color (create a palette)
                color_type = 3
            else:
                # RGB colors
                color_type = 2
        elif depth == 4:
            # RGB and alpha
            if a.dtype == _np.uint8 and use_palette:
                color_type = 3
            else:
                color_type = 6
    return color_type


def _validate_bitdepth(bitdepth, a, color_type):
    if bitdepth not in [None, 1, 2, 4, 8, 16]:
        raise ValueError('bitdepth %i is not valid.  Valid values are '
                         '1, 2, 4, 8 or 16' % (bitdepth,))

    if bitdepth is not None:
        if color_type in [2, 4, 6]:
            if 8*a.dtype.itemsize != bitdepth:
                raise ValueError("For the given input, the bit depth must "
                                 "match the data type of the array.")
        elif color_type == 3:
            if bitdepth == 16:
                raise ValueError("Bit depth 16 not allowed when use_palette "
                                 "is True.")


def _validate_timestamp(timestamp):
    if timestamp is None:
        return None
    if len(timestamp) != 6:
        raise ValueError("timestamp must have length 6")
    return timestamp


def _validate_phys(phys):
    if phys is not None:
        if len(phys) == 2:
            phys = tuple(phys) + (0,)
        elif phys[2] not in [0, 1]:
            raise ValueError('Third element of `phys` must be 0 or 1.')
        phys = [int(x) for x in phys]
        if phys[0] <= 0 or phys[1] <= 0:
            raise ValueError('The pixels per unit in `phys` must be positive.')
    return phys


def _add_background_color(background, palette, trans, bitdepth):
    if len(background) != 3:
        raise ValueError("background must have length 3 when "
                         "use_palette is True.")
    index = _np.where((palette == background).all(axis=-1))[0]
    if index.size > 0:
        # The given background color is in the palette.
        background = index[0]
    else:
        # The given background color is *not* in the palette.  Is there
        # room for one more color?
        if bitdepth is None:
            bitdepth = 8
        if len(palette) == 2**bitdepth:
            msg = ("The array already has the maximum of %i colors, and a "
                   "background color that is not in the array has been given. "
                   "With a bitdepth of %i, no more than %i colors are allowed "
                   "when using a palette." % (2**bitdepth, bitdepth,
                                              2**bitdepth))
            raise ValueError(msg)
        else:
            index = len(palette)
            palette = _np.append(palette,
                                 _np.array([background],
                                           dtype=_np.uint8),
                                 axis=0)
            if trans is not None:
                trans = _np.append(trans, [_np.uint8(255)])
            background = index

    return background, palette, trans


def write_png(fileobj, a, text_list=None, use_palette=False,
              transparent=None,  bitdepth=None, max_chunk_len=None,
              timestamp=None, gamma=None, background=None,
              filter_type=None, interlace=0, phys=None):
    """
    Write a numpy array to a PNG file.

    Parameters
    ----------
    fileobj : string or file object
        If fileobj is a string, it is the name of the PNG file to be created.
        Otherwise fileobj must be a file opened for writing.
    a : numpy array
        Must be an array of 8- or 16-bit unsigned integers.  The shape of `a`
        must be (m, n) or (m, n, d) with 1 <= d <= 4.
    text_list : list of (keyword, text) tuples, optional
        Each tuple is written to the file as a 'tEXt' chunk.  See the Notes
        for more information about text in PNG files.
    use_palette : bool, optional
        If True, *and* the data type of `a` is `numpy.uint8`, *and* the size
        of `a` is (m, n, 3), then a PLTE chunk is created and an indexed color
        image is created.  (If the conditions on `a` are not true, this
        argument is ignored and a palette is not created.)  There must not be
        more than 2**bitdepth distinct colors in `a`.  If the conditions on `a`
        are true but the array has more than 256 colors, a ValueError exception
        is raised.
    transparent : integer or 3-tuple of integers (r, g, b), optional
        If the colors in `a` do not include an alpha channel (i.e. the shape
        of `a` is (m, n), (m, n, 1) or (m, n, 3)), the `transparent` argument
        can be used to specify a single color that is to be considered the
        transparent color.  This argument is ignored if `a` includes an
        alpha channel, or if `use_palette` is True and the `transparent`
        color is not in `a`.  Otherwise, a 'tRNS' chunk is included in the
        PNG file.
    bitdepth : integer, optional
        Bit depth of the output image.  Valid values are 1, 2, 4 and 8.
        Only valid for grayscale images with no alpha channel with an input
        array having dtype numpy.uint8.  If not given, the bit depth is
        inferred from the data type of the input array `a`.
    max_chunk_len : integer, optional
        The data in a PNG file is stored in records called IDAT chunks.
        `max_chunk_len` sets the maximum number of data bytes to stored in
        each IDAT chunk.  The default is None, which means that all the data
        is written to a single IDAT chunk.
    timestamp : tuple with length 6, optional
        If this argument is not None, a 'tIME' chunk is included in the
        PNG file.  The value must be a tuple of six integers: (year, month,
        day, hour, minute, second).
    gamma : float, optional
        If this argument is not None, a 'gAMA' chunk is included in the
        PNG file.  The argument is expected to be a floating point value.
        The value written in the 'gAMA' chunk is int(gamma*100000 + 0.5).
    background : int (for grayscale) or sequence of three ints (for RGB)
        Set the default background color.  When this option is used, a
        'bKGD' chunk is included in the PNG file.  When `use_palette`
        is True, and `background` is not one of the colors in `a`, the
        `background` color is included in the palette, and so it counts
        towards the maximum number of 256 colors allowed in a palette.
    filter_type : one of 0, 1, 2, 3, 4, "heuristic" or "auto", optional
        Controls the filter type that is used per scanline in the IDAT
        chunks.  The default is "auto", which means the output data is
        generated six time, once for each of the other possible filter
        types, and the filter that generates the smallest output is used.
    interlace : either 0 or 1
        Interlace method to use.  0 means no interlace; 1 means Adam7.
    phys : tuple with length 2 or 3, optional
        If given, a `pHYs` chunk is written to the PNG file.
        If `phys` is given, it must be a tuple of integers with length 2
        or 3.  The first two integers are the pixels per unit of the X
        and Y axes, respectively.  The third value, if given, must be 0
        or 1.  If the value is 1, the units of the first two values are
        pixels per *meter*.  If the third value is 0 (or not given),
        the units of the first two values are undefined.  In that case,
        the values define the pixel aspect ratio only.

    Notes
    -----
    If `a` is three dimensional (i.e. `a.ndim == 3`), the size of the last
    dimension determines how the values in the last dimension are interpreted,
    as follows:

        a.shape[2]     Interpretation
        ----------     --------------------
            1          grayscale
            2          grayscale and alpha
            3          RGB
            4          RGB and alpha

    The `text_list` argument accepts a list of tuples of two strings argument.
    The first item in each tuple is the *keyword*, and the second is the text
    string.  This argument allows `'tEXt'` chunks to be created.  The
    following is from the PNG specification::

        The keyword indicates the type of information represented by the
        text string. The following keywords are predefined and should be
        used where appropriate:

            Title          Short (one line) title or caption for image
            Author         Name of image's creator
            Description    Description of image (possibly long)
            Copyright      Copyright notice
            Creation Time  Time of original image creation
            Software       Software used to create the image
            Disclaimer     Legal disclaimer
            Warning        Warning of nature of content
            Source         Device used to create the image
            Comment        Miscellaneous comment; conversion from GIF comment

        Both keyword and text are interpreted according to the ISO 8859-1
        (Latin-1) character set [ISO-8859]. The text string can contain any
        Latin-1 character. Newlines in the text string should be represented
        by a single linefeed character (decimal 10); use of other control
        characters in the text is discouraged.

        Keywords must contain only printable Latin-1 characters and spaces;
        that is, only character codes 32-126 and 161-255 decimal are allowed.
        To reduce the chances for human misreading of a keyword, leading and
        trailing spaces are forbidden, as are consecutive spaces. Note also
        that the non-breaking space (code 160) is not permitted in keywords,
        since it is visually indistinguishable from an ordinary space.
    """

    if interlace not in [0, 1]:
        raise ValueError('interlace must be 0 or 1.')

    if filter_type is None:
        filter_type = "auto"

    phys = _validate_phys(phys)

    a = _np.ascontiguousarray(a)
    _validate_array(a)

    text_list = _validate_text(text_list)

    timestamp = _validate_timestamp(timestamp)

    # Determine color_type:
    #
    #  color_type   meaning                    tRNS chunk contents (optional)
    #  ----------   ------------------------   --------------------------------
    #      0        grayscale                  Single gray level value, 2 bytes
    #      2        RGB                        Single RGB, 2 bytes per channel
    #      3        8 bit indexed RGB or RGBA  Series of 1 byte alpha values
    #      4        Grayscale and alpha
    #      6        RGBA
    #
    #
    # from http://www.w3.org/TR/PNG/:
    # Table 11.1 - Allowed combinations of colour type and bit depth
    #
    #                      Color  Allowed
    # PNG image type       type  bit depths      Interpretation
    # Greyscale              0   1, 2, 4, 8, 16  Each pixel is a greyscale
    #                                            sample
    # Truecolour             2   8, 16           Each pixel is an RGB triple
    # Indexed-colour         3   1, 2, 4, 8      Each pixel is a palette index;
    #                                            a PLTE chunk shall appear.
    # Greyscale with alpha   4   8, 16           Each pixel is a greyscale
    #                                            sample followed by an alpha
    #                                            sample.
    # Truecolour with alpha  6   8, 16           Each pixel is an RGB triple
    #                                            followed by an alpha sample.

    color_type = _get_color_type(a, use_palette)

    trans = None
    if color_type == 3:
        # The array is 8 bit RGB or RGBA, and a palette is to be created.

        # Note that this replaces `a` with the index array.
        a, palette, trans = _palettize(a)
        # `a` has the same shape as before, but now it is an array of indices
        # into the array `palette`, which contains the colors.  `trans` is
        # either None (if there was no alpha channel), or an array the same
        # length as `palette` containing the alpha values of the colors.

        bd = bitdepth if bitdepth is not None else 8
        max_num_colors = 2**bd
        if len(palette) > max_num_colors:
            raise ValueError("The array has %i colors.  With bit depth %i, "
                             "no more than %i colors are allowed when using "
                             "a palette." %
                             (len(palette), bd, max_num_colors))

        if background is not None:
            # A default background color has been given, and we're creating
            # an indexed palette (use_palette is True).  Convert the given
            # background color to an index.  If the color is not in the
            # palette, extend the palette with the new color (or raise an
            # error if there are already 2**bitdepth colors).
            background, palette, trans = _add_background_color(background,
                                                               palette, trans,
                                                               bitdepth)

        if trans is None and transparent is not None:
            # The array does not have an alpha channel.  The caller has given
            # a color value that should be considered to be transparent.
            # We construct an array `trans` of alpha values, and set the
            # alpha of the color that is to be transparent to 0.  All other
            # alpha values are set to 255 (fully opaque).
            # `trans` only has entries for colors in the palette up to the
            # given `transparent` color, so `trans` is not the same length as
            # `palette` (unless the transparent color happens to be the last
            # color in the palette).
            pal_index = _np.nonzero((palette == transparent).all(axis=1))[0]
            if pal_index.size > 0:
                if pal_index.size > 1:
                    raise ValueError("Only one transparent color may "
                                     "be given.")
                trans = _np.zeros(pal_index[0]+1, dtype=_np.uint8)
                trans[:-1] = 255

    elif (color_type == 0 or color_type == 2) and transparent is not None:
        # XXX Should do some validation of `transparent`...
        trans = _np.asarray(transparent, dtype=_np.uint16)

    _validate_bitdepth(bitdepth, a, color_type)

    if hasattr(fileobj, 'write'):
        # Assume it is a file-like object with a write method.
        f = fileobj
    else:
        # Assume it is a filename.
        f = open(fileobj, "wb")

    # Write the PNG header.
    png_header = b"\x89PNG\x0D\x0A\x1A\x0A"
    f.write(png_header)

    # Write the chunks...

    # IHDR chunk
    if bitdepth is not None:
        nbits = bitdepth
    else:
        nbits = a.dtype.itemsize*8
    _write_ihdr(f, a.shape[1], a.shape[0], nbits, color_type, interlace)

    # tEXt chunks, if any.
    if text_list is not None:
        for keyword, text_string in text_list:
            _write_text(f, keyword, text_string)

    if timestamp is not None:
        _write_time(f, timestamp)

    if gamma is not None:
        _write_gama(f, gamma)

    # PLTE chunk, if requested.
    if color_type == 3:
        _write_plte(f, palette)

    # tRNS chunk, if there is one.
    if trans is not None:
        _write_trns(f, trans)

    # bKGD chunk, if there is one.
    if background is not None:
        _write_bkgd(f, background, color_type)

    # pHYs chunk, if `phys` was given.
    if phys is not None:
        _write_phys(f, phys)

    # _write_data(...) writes the IDAT chunk(s).
    _write_data(f, a, bitdepth, max_chunk_len=max_chunk_len,
                filter_type=filter_type, interlace=interlace)

    # IEND chunk
    _write_iend(f)

    if f != fileobj:
        f.close()


def _msec_to_numden(delay):
    """
    delay is the time delay in milliseconds.

    Return value is the tuple (delay_num, delay_den) representing
    the delay in seconds as the fraction delay_num/delay_den.
    Each value in the tuple is an integer less than 65536.
    """
    if delay == 0:
        return (0, 1)
    # Convert delay to seconds.
    delay_sec = delay/1000.0
    if delay_sec > 1:
        f = _Fraction.from_float(1.0/delay_sec).limit_denominator(65535)
        num = f.denominator
        den = f.numerator
    else:
        f = _Fraction.from_float(delay_sec).limit_denominator(65535)
        num = f.numerator
        den = f.denominator
    if (num, den) == (1, 0):
        raise ValueError("delay=%r is too large to convert to "
                         "delay_num/delay_den" % (delay,))
    if (num, den) == (0, 1):
        raise ValueError("delay=%r is too small to convert to "
                         "delay_num/delay_den" % (delay,))
    return num, den


def write_apng(fileobj, seq, delay=None, num_plays=0, default_image=None,
               offset=None,
               text_list=None, use_palette=False,
               transparent=None, bitdepth=None,
               max_chunk_len=None, timestamp=None, gamma=None,
               background=None, filter_type=None, interlace=0, phys=None):
    """
    Write an APNG file from a sequence of numpy arrays.

    Warning:
    * This API is experimental, and will likely change.
    * The function has not been thoroughly tested.

    Parameters
    ----------
    seq : sequence of numpy arrays
        All the arrays must have the same shape and dtype.
    delay : scalar or sequence of scalars, optional
        The time display the frames, in milliseconds.
        If `delay` is None (the default) or 0, the frames are played as
        fast as possible.  If `delay` is a sequence, it must have the same
        length as `seq.
    num_plays : int
        The number of times to repeat the animation.  If 0, the animation
        is repeated indefinitely.
    default_image : numpy array
        If this image is given, it is the image that is displayed by renderers
        that do not support animated PNG files.  If the renderer does support
        animation, this image is not shown.  If this argument is not given,
        the image shown by renderers that do not support animation will be
        `seq[0]`.
    offset : sequence of tuples each with length 2, optional
        If given, this must be a sequence of the form
            [(row_offset0, col_offset0), (row_offset1, col_offset1), ...]
        The length of the sequence must be the same as `seq`.  It defines
        the location of the image within the PNG output buffer.
    text_list : list of (keyword, text) tuples, optional
        Each tuple is written to the file as a 'tEXt' chunk.
    use_palette : bool, optional
        If True, *and* the data type of the arrays in `seq` is `numpy.uint8`,
        *and* the size of each array is (m, n, 3), then a PLTE chunk is
        created and an indexed color image is created.  (If the conditions
        on the arrays are not true, this argument is ignored and a palette
        is not created.)  There must not be more than 256 distinct colors in
        the arrays.  If the above conditions are true but the arrays have
        more than 256 colors, a ValueError exception is raised.
    transparent : integer or 3-tuple of integers (r, g, b), optional
        If the colors in the input arrays do not include an alpha channel
        (i.e. the shape of each array is (m, n), (m, n, 1) or (m, n, 3)),
        the `transparent` argument can be used to specify a single color that
        is to be considered the transparent color.  This argument is ignored
        if the arrays have an alpha channel.
    bitdepth : integer, optional
        Bit depth of the output image.  Valid values are 1, 2, 4 and 8.
        Only valid for grayscale images with no alpha channel with an input
        array having dtype numpy.uint8.  If not given, the bit depth is
        inferred from the data type of the input arrays.
    max_chunk_len : integer, optional
        The data in a APNG file is stored in records called IDAT and fdAT
        chunks.  `max_chunk_len` sets the maximum number of data bytes to
        stored in each chunk.  The default is None, which means that all the
        data from a frame is written to a single IDAT or fdAT chunk.
    timestamp : tuple with length 6, optional
        If this argument is not None, a 'tIME' chunk is included in the
        PNG file.  The value must be a tuple of six integers: (year, month,
        day, hour, minute, second).
    gamma : float, optional
        If this argument is not None, a 'gAMA' chunk is included in the
        PNG file.  The argument is expected to be a floating point value.
        The value written in the 'gAMA' chunk is int(gamma*100000 + 0.5).
    background : int (for grayscale) or sequence of three ints (for RGB)
        Set the default background color.  When this option is used, a
        'bKGD' chunk is included in the PNG file.  When `use_palette`
        is True, and `background` is not one of the colors in `a`, the
        `background` color is included in the palette, and so it counts
        towards the maximum number of 256 colors allowed in a palette.
    filter_type : one of 0, 1, 2, 3, 4, "heuristic" or "auto", optional
        Controls the filter type that is used per scanline in the IDAT
        chunks.  The default is "auto", which means the output data for
        each frame is generated six time, once for each of the other
        possible filter types, and the filter that generates the smallest
        output is used.
    interlace : either 0 or 1
        Interlace method to use.  0 means no interlace; 1 means Adam7.
    phys : tuple with length 2 or 3, optional
        If given, a `pHYs` chunk is written to the PNG file.
        If `phys` is given, it must be a tuple of integers with length 2
        or 3.  The first two integers are the pixels per unit of the X
        and Y axes, respectively.  The third value, if given, must be 0
        or 1.  If the value is 1, the units of the first two values are
        pixels per *meter*.  If the third value is 0 (or not given),
        the units of the first two values are undefined.  In that case,
        the values define the pixel aspect ratio only.

    Notes
    -----

    See the `write_png` docstring for additional details about some
    of the arguments.
    """
    if interlace not in [0, 1]:
        raise ValueError('interlace must be 0 or 1.')

    if filter_type is None:
        filter_type = "auto"

    num_frames = len(seq)
    if num_frames == 0:
        raise ValueError("no frames given in `seq`")

    if delay is None:
        delay = [0] * num_frames
    else:
        try:
            ndelay = len(delay)
        except TypeError:
            delay = [delay] * num_frames
        if len(delay) != num_frames:
            raise ValueError('len(delay) must be the same as len(seq)')

    # Validate seq
    if type(seq) == _np.ndarray:
        # seq is a single numpy array containing the frames.
        seq = _np.ascontiguousarray(seq)
        _validate_array(seq[0])
    else:
        # seq is not a numpy array, so it must be a sequence of numpy arrays,
        # all with the same dtype.
        for a in seq:
            _validate_array(a)
        if any(a.dtype != seq[0].dtype for a in seq[1:]):
            raise ValueError("all arrays in `seq` must have the same dtype.")
        seq = [_np.ascontiguousarray(a) for a in seq]

    if offset is not None:
        if len(offset) != len(seq):
            raise ValueError('length of offset sequence must equal len(seq)')
    else:
        offset = [(0, 0)] * num_frames

    # Overall width and height.
    width = max(a.shape[1] + offset[k][1] for k, a in enumerate(seq))
    height = max(a.shape[0] + offset[k][0] for k, a in enumerate(seq))

    # Validate default_image
    if default_image is not None:
        _validate_array(default_image)
        if default_image.dtype != seq[0].dtype:
            raise ValueError('default_image must have the same data type as '
                             'the arrays in seq')
        if default_image.shape[0] > height or default_image.shape[1] > width:
            raise ValueError("The default image has shape (%i, %i), which "
                             "exceeds the overall image size implied by `seq` "
                             "and `offset`, which is (%i,  %i)" %
                             (default_image.shape[:2] + (height, width)))

    text_list = _validate_text(text_list)

    phys = _validate_phys(phys)

    timestamp = _validate_timestamp(timestamp)

    color_type = _get_color_type(seq[0], use_palette)

    trans = None
    if color_type == 3:
        # The arrays are 8 bit RGB or RGBA, and a palette is to be created.

        if default_image is None:
            seq, palette, trans = _palettize_seq(seq)
        else:
            tmp = [default_image] + [a for a in seq]
            index_tmp, palette, trans = _palettize_seq(tmp)
            default_image = index_tmp[0]
            seq = index_tmp[1:]
        # seq and default_image have the same shapes as before, but now the
        # the arrays hold indices into the array `palette`, which contains
        # the colors.  `trans` is either None (if there was no alpha channel),
        # or an array containing the alpha values of the colors.

        bd = bitdepth if bitdepth is not None else 8
        max_num_colors = 2**bd
        if len(palette) > max_num_colors:
            raise ValueError("The arrays have a total of %i colors. "
                             "With bit depth %i, no more than %i colors are "
                             "allowed when using a palette." %
                             (len(palette), bd, max_num_colors))


        if background is not None:
            # A default background color has been given, and we're creating
            # an indexed palette (use_palette is True).  Convert the given
            # background color to an index.  If the color is not in the
            # palette, extend the palette with the new color (or raise an
            # error if there are already 256 colors).
            background, palette, trans = _add_background_color(background,
                                                               palette, trans,
                                                               bitdepth)

        if trans is None and transparent is not None:
            # The array does not have an alpha channel.  The caller has given
            # a color value that should be considered to be transparent.
            pal_index = _np.nonzero((palette == transparent).all(axis=1))[0]
            if pal_index.size > 0:
                if pal_index.size > 1:
                    raise ValueError("Only one transparent color may "
                                     "be given.")
                trans = _np.zeros(pal_index[0]+1, dtype=_np.uint8)
                trans[:-1] = 255

    elif (color_type == 0 or color_type == 2) and transparent is not None:
        # XXX Should do some validation of `transparent`...
        trans = _np.asarray(transparent, dtype=_np.uint16)

    if bitdepth == 8 and seq[0].dtype == _np.uint8:
        bitdepth = None

    _validate_bitdepth(bitdepth, seq[0], color_type)

    # --- Open and write the file ---------

    if hasattr(fileobj, 'write'):
        # Assume it is a file-like object with a write method.
        f = fileobj
    else:
        # Assume it is a filename.
        f = open(fileobj, "wb")

    # Write the PNG header.
    png_header = b"\x89PNG\x0D\x0A\x1A\x0A"
    f.write(png_header)

    # Write the chunks...

    # IHDR chunk
    if bitdepth is not None:
        nbits = bitdepth
    else:
        nbits = seq[0].dtype.itemsize*8
    _write_ihdr(f, width, height, nbits, color_type, interlace)

    # tEXt chunks, if any.
    if text_list is not None:
        for keyword, text_string in text_list:
            _write_text(f, keyword, text_string)

    if timestamp is not None:
        _write_time(f, timestamp)

    if gamma is not None:
        _write_gama(f, gamma)

    # PLTE chunk, if requested.
    if color_type == 3:
        _write_plte(f, palette)

    # tRNS chunk, if there is one.
    if trans is not None:
        _write_trns(f, trans)

    # bKGD chunk, if there is one.
    if background is not None:
        _write_bkgd(f, background, color_type)

    # pHYs chunk, if `phys` was given.
    if phys is not None:
        _write_phys(f, phys)

    # acTL chunk
    _write_actl(f, num_frames, num_plays)

    # Convert delays (which are in  milliseconds) to the number of
    # seconds expressed as the fraction delay_num/delay_den.
    delay_num, delay_den = zip(*[_msec_to_numden(d) for d in delay])

    sequence_number = 0
    frame_number = 0

    if default_image is None:
        # fcTL chunk for the first frame
        _write_fctl(f, sequence_number=sequence_number,
                    width=seq[0].shape[1], height=seq[0].shape[0],
                    x_offset=0, y_offset=0,
                    delay_num=delay_num[frame_number],
                    delay_den=delay_den[frame_number],
                    dispose_op=0, blend_op=0)
        sequence_number += 1
        frame_number += 1
        # IDAT chunk(s) for the first frame (no sequence_number)
        _write_data(f, seq[0], bitdepth, max_chunk_len=max_chunk_len,
                    filter_type=filter_type, interlace=interlace)
        seq = seq[1:]
    else:
        # IDAT chunk(s) for the default image
        _write_data(f, default_image, bitdepth, max_chunk_len=max_chunk_len,
                    filter_type=filter_type, interlace=interlace)

    for frame in seq:
        # fcTL chunk for the next frame
        _write_fctl(f, sequence_number=sequence_number,
                    width=frame.shape[1], height=frame.shape[0],
                    x_offset=offset[frame_number][1],
                    y_offset=offset[frame_number][0],
                    delay_num=delay_num[frame_number],
                    delay_den=delay_den[frame_number],
                    dispose_op=0, blend_op=0)
        sequence_number += 1
        frame_number += 1

        # fdAT chunk(s) for the next frame
        num_chunks = _write_data(f, frame, bitdepth,
                                 max_chunk_len=max_chunk_len,
                                 sequence_number=sequence_number,
                                 filter_type=filter_type, interlace=interlace)
        sequence_number += num_chunks

    # IEND chunk
    _write_iend(f)

    if f != fileobj:
        f.close()


def _finddiff(img1, img2):
    """
    Finds the bounds of the region where img1 and img2 differ.

    img1 and img2 must be 2D or 3D numpy arrays with the same shape.
    """
    if img1.shape != img2.shape:
        raise ValueError('img1 and img2 must have the same shape')

    if img1.ndim == 2:
        mask = img1 != img2
    else:
        mask = _np.any(img1 != img2, axis=-1)
    if _np.any(mask):
        rows, cols = _np.where(mask)
        row_range = rows.min(), rows.max() + 1
        col_range = cols.min(), cols.max() + 1
    else:
        row_range = None
        col_range = None
    return row_range, col_range


class AnimatedPNGWriter(object):
    """
    This class implements the interface required by the matplotlib class
    `matplotlib.animation.MovieWriter`.  An instance of this class may be
    used as the `writer` argument of `matplotlib.animation.Animation.save()`.

    This class is experimental.  It may change without warning in the next
    release.
    """

    # I haven't tried to determine all the additional arguments that
    # should be given in __init__, and I haven't checked what should
    # be pulled from rcParams if a corresponding argument is not given.
    def __init__(self, fps, filter_type=None):
        self.fps = fps
        # Convert frames-per-second to delay between frames in milliseconds.
        self._delay = 1000/fps
        self._filter_type = filter_type

    def setup(self, fig, outfile, dpi, *args):
        self.fig = fig
        self.outfile = outfile
        self.dpi = dpi
        self._frames = []
        self._prev_frame = None

    def grab_frame(self, **savefig_kwargs):
        img_io = _BytesIO()
        self.fig.savefig(img_io, format='rgba', dpi=self.dpi, **savefig_kwargs)
        raw = img_io.getvalue()

        # A bit of experimentation suggested that taking the integer part of
        # the following products is the correct conversion, but I haven't
        # verified it in the matplotlib code.  If this is not the correct
        # conversion, the call of the reshape method after calling frombuffer
        # will likely raise an exception.
        height = int(self.fig.get_figheight() * self.dpi)
        width = int(self.fig.get_figwidth() * self.dpi)
        a = _np.frombuffer(raw, dtype=_np.uint8).reshape(height, width, 4)

        if self._prev_frame is None:
            self._frames.append((a, (0, 0), self._delay))
        else:
            rows, cols = _finddiff(a, self._prev_frame)
            if rows is None:
                # No difference, so just increment the delay of the previous
                # frame.
                img, offset, delay = self._frames[-1]
                self._frames[-1] = (img, offset, delay + self._delay)
            else:
                # b is the rectangular region that contains the part
                # of the image that changed.
                b = a[rows[0]:rows[1], cols[0]:cols[1]]
                offset = (rows[0], cols[0])
                self._frames.append((b, offset, self._delay))
        self._prev_frame = a

    def finish(self):
        for img, offset, delay in self._frames:
            if not _np.all(img[:, :, 3] == 255):
                break
        else:
            # All the alpha values are 255, so drop the alpha channel.
            self._frames = [(img[:, :, :3], offset, delay)
                            for img, offset, delay in self._frames]

        imgs, offsets, delays = zip(*self._frames)
        write_apng(self.outfile, imgs, offset=offsets, delay=delays,
                   filter_type=self._filter_type)

    @_contextlib.contextmanager
    def saving(self, fig, outfile, dpi, *args):
        """
        Context manager to facilitate writing the movie file.

        All arguments are passed to `setup()`.
        """
        self.setup(fig, outfile, dpi, *args)
        yield
        self.finish()

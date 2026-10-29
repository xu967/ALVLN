#!/usr/bin/env python3
"""Save a `gz topic -e --json-output` gz.msgs.Image message as a PNG.

Pure stdlib (zlib), no extra dependencies. Supports 8-bit RGB/BGR (3ch) and
RGBA/BGRA (4ch), including rows with stride padding (`step > width*nch`).
Other formats (grayscale, 16-bit, float) are rejected without creating dst.

Usage:
  gz topic -e -t /uav/camera -n 1 --json-output | save_frame.py out.png
  save_frame.py frame.json out.png
"""
import base64, binascii, json, struct, sys, zlib

# pixelFormatType/format -> (channels, swap R/B)
_FORMATS = {
    "RGB_INT8": (3, False), "R8G8B8": (3, False),
    "BGR_INT8": (3, True), "B8G8R8": (3, True),
    "RGBA_INT8": (4, False), "R8G8B8A8": (4, False),
    "BGRA_INT8": (4, True), "B8G8R8A8": (4, True),
}

def _chunk(tag, data):
    c = struct.pack('>I', len(data)) + tag + data
    return c + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

def _png(w, h, nch, buf):
    """buf 是紧凑行主序（w*h*nch），调用方负责去掉 stride padding。"""
    color_type = {3: 2, 4: 6}[nch]  # 2=RGB, 6=RGBA
    if len(buf) < w * h * nch:
        raise ValueError(f"pixel buffer too short: {len(buf)} < {w * h * nch}")
    rows = b''.join(b'\x00' + buf[y*w*nch:(y+1)*w*nch] for y in range(h))
    return (b'\x89PNG\r\n\x1a\n'
            + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, color_type, 0, 0, 0))
            + _chunk(b'IDAT', zlib.compress(rows, 6))
            + _chunk(b'IEND', b''))

def decode(msg):
    """gz.msgs.Image dict -> (w, h, nch, 紧凑 RGB/RGBA 字节)。失败抛 ValueError。"""
    w, h, step = int(msg['width']), int(msg['height']), int(msg['step'])
    if w <= 0 or h <= 0 or step <= 0:
        raise ValueError(f"invalid image size {w}x{h} step={step}")
    fmt = str(msg.get('pixelFormatType') or msg.get('format') or '').upper()
    if fmt in _FORMATS:
        nch, swap = _FORMATS[fmt]
    elif not fmt and step % w == 0 and step // w in (3, 4):
        # 兼容没有 pixelFormatType 的旧消息；不靠 step 猜未知格式。
        nch, swap = step // w, False
    else:
        raise ValueError(f"unsupported pixel format {fmt or '?'!r} "
                         f"(need 8-bit RGB/BGR/RGBA/BGRA)")
    if step < w * nch:
        raise ValueError(f"step {step} < width*channels {w * nch}")
    raw = base64.b64decode(msg.get('data', ''))
    need = (h - 1) * step + w * nch
    if len(raw) < need:
        raise ValueError(f"payload too short: {len(raw)} < {need}")
    rows = bytearray(w * h * nch)
    for y in range(h):
        row = bytearray(raw[y*step:y*step + w*nch])
        if swap:
            for i in range(0, len(row), nch):
                row[i], row[i+2] = row[i+2], row[i]
        rows[y*w*nch:(y+1)*w*nch] = row
    return w, h, nch, bytes(rows)

def png_bytes(msg):
    w, h, nch, rows = decode(msg)
    return w, h, nch, _png(w, h, nch, rows)

def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    src = sys.argv[1] if len(sys.argv) == 3 else '-'
    dst = sys.argv[2] if len(sys.argv) == 3 else sys.argv[1]
    text = sys.stdin.buffer.read() if src == '-' else open(src, 'rb').read()
    try:
        msg = json.loads(text)
        w, h, nch, png = png_bytes(msg)
    except (ValueError, KeyError, TypeError, binascii.Error) as e:
        print(f"save_frame: {e}", file=sys.stderr)
        return 1
    with open(dst, 'wb') as f:
        f.write(png)
    stamp = msg.get('header', {}).get('stamp', {})
    print(f"saved {dst} {w}x{h} {nch}ch sim_time={stamp.get('sec','?')}.{stamp.get('nsec','?'):>09}", file=sys.stderr)
    return 0

if __name__ == '__main__':
    sys.exit(main())

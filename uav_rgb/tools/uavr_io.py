#!/usr/bin/env python3
"""UAVR v1 数据集二进制头、meta JSON 与标签区的共用读取辅助（tools/DATASET_FORMAT.md §1）。

仅依赖 Python 标准库（不引入 numpy / PIL），便于 UI、闭环闸门、打包与评测工具统一复用。
"""
import json
import os
import struct
from typing import NamedTuple

UAVR_MAGIC = b"UAVR"
UAVR_VERSION = 1
UAVR_HEADER = struct.Struct("<4sHHIHHBBBBQQQI")  # 48 B


class UavrHeader(NamedTuple):
    magic: bytes
    version: int
    header_size: int
    n: int
    h: int
    w: int
    c: int
    img_dtype: int
    label_dtype: int
    reserved: int
    images_off: int
    labels_off: int
    meta_off: int
    meta_len: int


def read_uavr_header(f_or_bytes, *, strict_rgb=False) -> UavrHeader:
    """从文件对象、48+ 字节缓冲或文件路径读取并校验 UAVR v1 头。"""
    if isinstance(f_or_bytes, (bytes, bytearray, memoryview)):
        raw = bytes(f_or_bytes[:UAVR_HEADER.size])
    elif isinstance(f_or_bytes, (str, os.PathLike)):
        with open(f_or_bytes, "rb") as f:
            raw = f.read(UAVR_HEADER.size)
    else:
        raw = f_or_bytes.read(UAVR_HEADER.size)
    if len(raw) != UAVR_HEADER.size:
        raise ValueError(f"不是 UAVR 数据集（文件头不足 {UAVR_HEADER.size} 字节）")
    hdr = UavrHeader(*UAVR_HEADER.unpack(raw))
    if hdr.magic != UAVR_MAGIC or hdr.version != UAVR_VERSION or hdr.header_size != UAVR_HEADER.size:
        raise ValueError(f"UAVR 头不匹配（magic={hdr.magic!r} version={hdr.version} header_size={hdr.header_size}）")
    if strict_rgb and (hdr.c, hdr.img_dtype, hdr.label_dtype) != (3, 1, 1):
        raise ValueError(f"不支持 c={hdr.c} img_dtype={hdr.img_dtype} label_dtype={hdr.label_dtype}（需要 3/uint8/uint8）")
    return hdr


def _validate_offsets(hdr: UavrHeader, file_size: int, *, strict_offsets: bool) -> None:
    if strict_offsets:
        img_bytes = hdr.n * hdr.c * hdr.h * hdr.w
        if hdr.images_off != UAVR_HEADER.size or hdr.labels_off != hdr.images_off + img_bytes:
            raise ValueError("UAVR 头里的 images/labels 偏移不自洽")
        if hdr.meta_off != hdr.labels_off + hdr.n or file_size != hdr.meta_off + hdr.meta_len:
            raise ValueError("UAVR 头里的 meta 偏移与文件长度不自洽")
    elif hdr.meta_off + hdr.meta_len > file_size:
        raise ValueError("UAVR meta 偏移超出文件长度（文件被截断？）")


def read_uavr_meta_and_header(path, *, strict_rgb=False, strict_offsets=False):
    """读取并返回 (UavrHeader, meta dict)。"""
    with open(path, "rb") as f:
        hdr = read_uavr_header(f, strict_rgb=strict_rgb)
        file_size = os.fstat(f.fileno()).st_size
        _validate_offsets(hdr, file_size, strict_offsets=strict_offsets)
        f.seek(hdr.meta_off)
        raw = f.read(hdr.meta_len)
    if len(raw) != hdr.meta_len:
        raise ValueError(f"{path}: UAVR meta 读取不完整")
    try:
        meta = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: UAVR meta 不是合法 JSON：{exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: UAVR meta 顶层必须是对象")
    return hdr, meta


def read_uavr_meta(path, *, strict_rgb=False, strict_offsets=False) -> dict:
    """只读 UAVR 文件尾部的 meta JSON 对象。"""
    _, meta = read_uavr_meta_and_header(path, strict_rgb=strict_rgb, strict_offsets=strict_offsets)
    return meta


def read_uavr_labels(path, *, strict_rgb=True, strict_offsets=True) -> bytes:
    """只读 UAVR 文件的标签区原始字节（长度 = N，每个样本 1 字节 uint8）。"""
    with open(path, "rb") as f:
        hdr = read_uavr_header(f, strict_rgb=strict_rgb)
        file_size = os.fstat(f.fileno()).st_size
        _validate_offsets(hdr, file_size, strict_offsets=strict_offsets)
        f.seek(hdr.labels_off)
        raw = f.read(hdr.n)
    if len(raw) != hdr.n:
        raise ValueError(f"{path}: UAVR 标签区被截断")
    return raw

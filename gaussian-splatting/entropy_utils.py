"""Entropy coding utilities for TurboSplat compression.

Provides Morton-curve sorting, sub-byte bit-packing, and zstd-based
binary serialization to replace np.savez_compressed. Together these
techniques yield ~1.5-2x additional compression on top of TurboQuant.

Requires: pyzstd  (pip install pyzstd)
"""

import struct
import numpy as np

try:
    import pyzstd
except ImportError:
    pyzstd = None


# ---------------------------------------------------------------------------
# 1. Morton code (Z-order curve) sorting
# ---------------------------------------------------------------------------

def _expand_bits(v):
    """Expand 21-bit integers to 63 bits with 2 zero-bits between each.

    Works on numpy int64 arrays (vectorized).
    """
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return v


def morton_encode_3d(x, y, z):
    """Encode 3D integer coordinates to 63-bit Morton / Z-order codes.

    Args:
        x, y, z: int64 arrays with values in [0, 2^21-1].

    Returns:
        uint64 array of Morton codes.
    """
    return (_expand_bits(x)
            | (_expand_bits(y) << np.uint64(1))
            | (_expand_bits(z) << np.uint64(2)))


def morton_sort_gaussians(attrs):
    """Sort all Gaussian attributes by Morton / Z-order curve of positions.

    Clusters spatially nearby Gaussians together in memory, improving
    compressibility of every attribute array.

    Args:
        attrs: dict with keys xyz, sh_dc, sh_rest, opacity, scales, rotations.

    Returns:
        sorted_attrs: same keys, arrays reordered by Morton code.
    """
    xyz = attrs["xyz"]
    # Normalise to [0, 2^21-1] integer grid
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-6)
    grid = ((xyz - bbox_min) / bbox_size * (2**21 - 1)).astype(np.int64)
    grid = np.clip(grid, 0, 2**21 - 1)

    codes = morton_encode_3d(grid[:, 0], grid[:, 1], grid[:, 2])
    order = np.argsort(codes)

    sorted_attrs = {}
    for key, val in attrs.items():
        sorted_attrs[key] = val[order]
    return sorted_attrs


# ---------------------------------------------------------------------------
# 2. Sub-byte bit-packing
# ---------------------------------------------------------------------------

def pack_2bit(indices):
    """Pack 2-bit indices (values 0-3) into bytes, 4 values per byte.

    Args:
        indices: uint8 array with values in [0, 3], arbitrary shape.

    Returns:
        packed: uint8 array (~4x smaller).
        original_length: int, needed for unpacking.
    """
    flat = indices.flatten().astype(np.uint8)
    original_length = len(flat)
    pad = (4 - len(flat) % 4) % 4
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    packed = ((flat[0::4] << 6)
              | (flat[1::4] << 4)
              | (flat[2::4] << 2)
              | flat[3::4])
    return packed.astype(np.uint8), original_length


def unpack_2bit(packed, original_length):
    """Unpack 2-bit packed bytes back to individual uint8 indices."""
    unpacked = np.empty(len(packed) * 4, dtype=np.uint8)
    unpacked[0::4] = (packed >> 6) & 0x3
    unpacked[1::4] = (packed >> 4) & 0x3
    unpacked[2::4] = (packed >> 2) & 0x3
    unpacked[3::4] = packed & 0x3
    return unpacked[:original_length]


def pack_3bit(indices):
    """Pack 3-bit indices (values 0-7) into bytes, 8 values per 3 bytes.

    Uses a simple scheme: groups of 8 values -> 24 bits -> 3 bytes.

    Args:
        indices: uint8 array with values in [0, 7].

    Returns:
        packed: uint8 array (~2.67x smaller).
        original_length: int.
    """
    flat = indices.flatten().astype(np.uint8)
    original_length = len(flat)
    pad = (8 - len(flat) % 8) % 8
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])

    # Reshape to groups of 8 values
    groups = flat.reshape(-1, 8).astype(np.uint32)
    # Pack 8 x 3-bit values into 24 bits (3 bytes)
    packed24 = np.zeros((len(groups), 3), dtype=np.uint8)
    # Byte 0: v0[2:0] v1[2:0] v2[1:0]  (bits: v0<<5 | v1<<2 | v2>>1)
    packed24[:, 0] = ((groups[:, 0] << 5)
                      | (groups[:, 1] << 2)
                      | (groups[:, 2] >> 1)).astype(np.uint8)
    # Byte 1: v2[0] v3[2:0] v4[2:0] v5[1:0]
    packed24[:, 1] = (((groups[:, 2] & 1) << 7)
                      | (groups[:, 3] << 4)
                      | (groups[:, 4] << 1)
                      | (groups[:, 5] >> 2)).astype(np.uint8)
    # Byte 2: v5[1:0] v6[2:0] v7[2:0]
    packed24[:, 2] = (((groups[:, 5] & 3) << 6)
                      | (groups[:, 6] << 3)
                      | groups[:, 7]).astype(np.uint8)

    return packed24.flatten(), original_length


def unpack_3bit(packed, original_length):
    """Unpack 3-bit packed bytes back to individual uint8 indices."""
    groups = packed.reshape(-1, 3).astype(np.uint32)
    unpacked = np.empty(len(groups) * 8, dtype=np.uint8)

    unpacked[0::8] = ((groups[:, 0] >> 5) & 0x7).astype(np.uint8)
    unpacked[1::8] = ((groups[:, 0] >> 2) & 0x7).astype(np.uint8)
    unpacked[2::8] = (((groups[:, 0] & 0x3) << 1) | ((groups[:, 1] >> 7) & 0x1)).astype(np.uint8)
    unpacked[3::8] = ((groups[:, 1] >> 4) & 0x7).astype(np.uint8)
    unpacked[4::8] = ((groups[:, 1] >> 1) & 0x7).astype(np.uint8)
    unpacked[5::8] = (((groups[:, 1] & 0x1) << 2) | ((groups[:, 2] >> 6) & 0x3)).astype(np.uint8)
    unpacked[6::8] = ((groups[:, 2] >> 3) & 0x7).astype(np.uint8)
    unpacked[7::8] = (groups[:, 2] & 0x7).astype(np.uint8)

    return unpacked[:original_length]


def pack_4bit(indices):
    """Pack 4-bit indices (values 0-15) into bytes, 2 values per byte.

    Args:
        indices: uint8 array with values in [0, 15].

    Returns:
        packed: uint8 array (~2x smaller).
        original_length: int.
    """
    flat = indices.flatten().astype(np.uint8)
    original_length = len(flat)
    pad = len(flat) % 2
    if pad:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    packed = (flat[0::2] << 4) | flat[1::2]
    return packed.astype(np.uint8), original_length


def unpack_4bit(packed, original_length):
    """Unpack 4-bit packed bytes back to individual uint8 indices."""
    unpacked = np.empty(len(packed) * 2, dtype=np.uint8)
    unpacked[0::2] = (packed >> 4) & 0xF
    unpacked[1::2] = packed & 0xF
    return unpacked[:original_length]


def pack_indices(indices, bits):
    """Pack quantisation indices at the given bit-width.

    For 8-bit indices no packing is performed (already 1 byte each).
    For sub-byte widths (2, 3, 4) the appropriate bit-packer is used.

    Args:
        indices: uint8 array.
        bits: quantisation bit-width (2, 3, 4, or 8).

    Returns:
        packed: uint8 array.
        original_length: int (number of original elements).
    """
    if bits == 2:
        return pack_2bit(indices)
    elif bits == 3:
        return pack_3bit(indices)
    elif bits == 4:
        return pack_4bit(indices)
    else:
        # 8-bit or larger: no packing needed
        return indices.flatten().astype(np.uint8), len(indices.flatten())


def unpack_indices(packed, original_length, bits):
    """Unpack quantisation indices at the given bit-width.

    Args:
        packed: uint8 array from pack_indices.
        original_length: int from pack_indices.
        bits: quantisation bit-width.

    Returns:
        uint8 array of length original_length.
    """
    if bits == 2:
        return unpack_2bit(packed, original_length)
    elif bits == 3:
        return unpack_3bit(packed, original_length)
    elif bits == 4:
        return unpack_4bit(packed, original_length)
    else:
        return packed[:original_length]


# ---------------------------------------------------------------------------
# 3. zstd binary format  (magic: TSv4)
# ---------------------------------------------------------------------------

def save_compressed(output_path, arrays_dict, compression_level=19):
    """Save dict of numpy arrays with zstd compression.

    Binary format::

        4 bytes: magic  b"TSv4"
        4 bytes: number of arrays  (uint32 LE)
        For each array:
            4 bytes: name length  (uint32 LE)
            N bytes: name  (utf-8)
            4 bytes: dtype string length  (uint32 LE)
            N bytes: dtype string  (utf-8)
            4 bytes: ndim  (uint32 LE)
            ndim * 8 bytes: shape dims  (uint64 LE each)
            4 bytes: compressed data length  (uint32 LE)
            N bytes: zstd-compressed raw array bytes

    Args:
        output_path: Destination file path.
        arrays_dict: {name: np.ndarray, ...}.
        compression_level: zstd level 1-22 (default 19 = high).
    """
    if pyzstd is None:
        raise ImportError("pyzstd is required for zstd format: pip install pyzstd")

    with open(output_path, "wb") as f:
        f.write(b"TSv4")
        f.write(struct.pack("<I", len(arrays_dict)))

        for name, arr in arrays_dict.items():
            arr = np.ascontiguousarray(arr)
            # Name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            # Dtype
            dtype_str = str(arr.dtype).encode("utf-8")
            f.write(struct.pack("<I", len(dtype_str)))
            f.write(dtype_str)
            # Shape
            f.write(struct.pack("<I", arr.ndim))
            for s in arr.shape:
                f.write(struct.pack("<Q", s))
            # Compressed data
            raw = arr.tobytes()
            compressed = pyzstd.compress(raw, level_or_option=compression_level)
            f.write(struct.pack("<I", len(compressed)))
            f.write(compressed)


def load_compressed(input_path):
    """Load arrays from custom zstd-compressed TSv4 format.

    Args:
        input_path: Path to a .tsv4 file written by save_compressed.

    Returns:
        Dict of {name: np.ndarray}.
    """
    if pyzstd is None:
        raise ImportError("pyzstd is required for zstd format: pip install pyzstd")

    arrays = {}
    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != b"TSv4":
            raise ValueError(f"Bad magic bytes: {magic!r} (expected b'TSv4')")
        n_arrays = struct.unpack("<I", f.read(4))[0]

        for _ in range(n_arrays):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            dtype_len = struct.unpack("<I", f.read(4))[0]
            dtype_str = f.read(dtype_len).decode("utf-8")
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(ndim))
            comp_len = struct.unpack("<I", f.read(4))[0]
            compressed = f.read(comp_len)
            raw = pyzstd.decompress(compressed)
            arrays[name] = np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)

    return arrays

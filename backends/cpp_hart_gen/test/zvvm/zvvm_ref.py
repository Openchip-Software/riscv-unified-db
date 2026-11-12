# Copyright (c) Openchip and Software Technologies, S.L.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Independent reference for the Zvvm FP/MX multiply-accumulate instructions.

Follows the specification pseudocode (integrated-matrix.adoc, v0.9.0-on-main):
fp_gemm, fp_scaled_gemm, int_scaled_gemm, read_block_scales, with
psm=0 and any G / rnd. Arithmetic is exact (Fraction); rounding follows
IEEE 754 with tininess after rounding. Decisions shared with the UDB IDL:
E5M2 S.11111.01 is a signaling NaN; exact-zero sums follow IEEE 754 6.3
applied in order from +0; scales follow the specification text.
"""

from collections import Counter
from fractions import Fraction

# Events seen while computing expected values, for the coverage summary of gen_tests.py.
COV = Counter()

FORMATS = {  # name: (ebits, mbits, ieee specials)
    "E2M1": (2, 1, False),
    "E4M3": (4, 3, False),
    "E5M2": (5, 2, True),
    "FP16": (5, 10, True),
    "BF16": (8, 7, True),
    "FP32": (8, 23, True),
    "FP64": (11, 52, True),
}
RMS = ["RNE", "RTZ", "RDN", "RUP", "RMM"]
NX, UF, OF, NV = 1, 2, 4, 16
FIN, INF, NAN = "fin", "inf", "nan"


def fmt_of(eew, alt):
    return {
        4: "E2M1",
        8: "E5M2" if alt else "E4M3",
        16: "BF16" if alt else "FP16",
        32: "FP32",
        64: "FP64",
    }[eew]


class Flags:
    def __init__(self):
        self.v = 0


def pack(f, s, e, m):
    eb, mb, _ = FORMATS[f]
    return (s << (eb + mb)) | (e << mb) | m


def default_nan(f):
    eb, mb, _ = FORMATS[f]
    return 0x7F if f == "E4M3" else pack(f, 0, (1 << eb) - 1, 1 << (mb - 1))


def decode(f, bits):
    """(kind, sign, Fraction magnitude, is_snan)"""
    eb, mb, ieee = FORMATS[f]
    bias = (1 << (eb - 1)) - 1
    m = bits & ((1 << mb) - 1)
    e = (bits >> mb) & ((1 << eb) - 1)
    s = (bits >> (eb + mb)) & 1
    if f == "E4M3" and e == 15 and m == 7:
        return (NAN, 0, Fraction(0), False)
    if ieee and e == (1 << eb) - 1:
        if m == 0:
            return (INF, s, Fraction(0), False)
        return (NAN, 0, Fraction(0), (m >> (mb - 1)) == 0)
    if e == 0:
        return (FIN, s, Fraction(m) * Fraction(2) ** (1 - bias - mb), False)
    return (FIN, s, Fraction((1 << mb) | m) * Fraction(2) ** (e - bias - mb), False)


def operand(f, bits, fl):
    k, s, v, snan = decode(f, bits)
    if snan:
        fl.v |= NV
        COV[f"signaling NaN operand {f}"] += 1
    elif k == INF:
        COV["infinite operand"] += 1
    return (k, s, v)


def exponent_of(x):
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    return e


def round_int(s, x, rm, rto):
    n = x.numerator // x.denominator
    rem = x - n
    if rem == 0:
        return n, False
    if rto:
        return n | 1, True
    half = Fraction(1, 2)
    up = {
        "RNE": rem > half or (rem == half and n % 2 == 1),
        "RTZ": False,
        "RDN": s == 1,
        "RUP": s == 0,
        "RMM": rem >= half,
    }[rm]
    return n + (1 if up else 0), True


def round_to(x, f, rm, fl, rto=False):
    k, s, v = x
    eb, mb, _ = FORMATS[f]
    assert f not in ("E2M1", "E4M3")
    bias = (1 << (eb - 1)) - 1
    emax_field = (1 << eb) - 1
    emin = 1 - bias
    if k == NAN:
        return default_nan(f)
    if k == INF:
        return pack(f, s, emax_field, 0)
    if v == 0:
        return pack(f, s, 0, 0)
    E = exponent_of(v)
    n1, _ = round_int(s, v / Fraction(2) ** (E - mb), rm, rto)
    tiny = (E + (1 if n1 == 1 << (mb + 1) else 0)) < emin
    q = max(E, emin) - mb
    n, inexact = round_int(s, v / Fraction(2) ** q, rm, rto)
    if n == 1 << (mb + 1):
        n >>= 1
        q += 1
    normal = n >= 1 << mb
    biased = q + mb + bias if normal else 0
    if biased >= emax_field:
        COV["overflow" + (" (round-to-odd)" if rto else "")] += 1
        fl.v |= OF | NX
        to_inf = not rto and (
            rm in ("RNE", "RMM") or (rm == "RUP" and s == 0) or (rm == "RDN" and s == 1)
        )
        return pack(f, s, emax_field, 0) if to_inf else pack(f, s, emax_field - 1, (1 << mb) - 1)
    if inexact:
        fl.v |= NX | (UF if tiny else 0)
        if tiny:
            COV["underflow"] += 1
    return pack(f, s, biased, n - (1 << mb) if normal else n)


def x_add(a, b, rm, fl):
    ka, sa, va = a
    kb, sb, vb = b
    if ka == NAN or kb == NAN:
        return (NAN, 0, Fraction(0))
    if ka == INF and kb == INF:
        if sa != sb:
            fl.v |= NV
            COV["inf - inf"] += 1
            return (NAN, 0, Fraction(0))
        return a
    if ka == INF:
        return a
    if kb == INF:
        return b
    t = (-va if sa else va) + (-vb if sb else vb)
    if t == 0:
        if va == 0 and vb == 0 and sa == sb:
            return (FIN, sa, Fraction(0))
        if va != 0:
            COV[f"exact cancellation to zero, {rm}"] += 1
        return (FIN, 1 if rm == "RDN" else 0, Fraction(0))
    return (FIN, 1 if t < 0 else 0, abs(t))


def x_mul(a, b, fl):
    ka, sa, va = a
    kb, sb, vb = b
    s = sa ^ sb
    if ka == NAN or kb == NAN:
        return (NAN, 0, Fraction(0))
    if ka == INF or kb == INF:
        if (ka == FIN and va == 0) or (kb == FIN and vb == 0):
            fl.v |= NV
            COV["inf x 0"] += 1
            return (NAN, 0, Fraction(0))
        return (INF, s, Fraction(0))
    return (FIN, s, va * vb)


ZERO = (FIN, 0, Fraction(0))


def fp_add(a, b, f, rm, fl):
    return round_to(x_add(operand(f, a, fl), operand(f, b, fl), rm, fl), f, rm, fl)


def fp_mul(a, b, f, rm, fl):
    return round_to(x_mul(operand(f, a, fl), operand(f, b, fl), fl), f, rm, fl)


def decode_scale(e, f, rm, fl):
    if e == 0xFF:
        COV["NaN block scale (0xFF)"] += 1
        return default_nan(f)
    return round_to((FIN, 0, Fraction(2) ** (e - 127)), f, rm, fl)


def is_nan(bits, f):
    return decode(f, bits)[0] == NAN


# ─── Register file and GEMM ──────────────────────────────────────────────


class VRegs:
    def __init__(self, vlen):
        self.vlenb = vlen // 8
        self.b = bytearray(32 * self.vlenb)

    def read(self, eew, flat, base):
        bitoff = base * self.vlenb * 8 + flat * eew
        if eew == 4:
            byte = self.b[bitoff // 8]
            return (byte >> (bitoff % 8)) & 0xF
        n = eew // 8
        return int.from_bytes(self.b[bitoff // 8 : bitoff // 8 + n], "little")

    def write(self, eew, flat, base, val):
        bitoff = base * self.vlenb * 8 + flat * eew
        n = eew // 8
        self.b[bitoff // 8 : bitoff // 8 + n] = (val & ((1 << eew) - 1)).to_bytes(n, "little")


def tile_reg_idx(i, group_regs, row_elems_per_reg, epr):
    linesize = row_elems_per_reg * group_regs
    line, e = divmod(i, linesize)
    return (e // row_elems_per_reg) * epr + line * row_elems_per_reg + e % row_elems_per_reg


class Geom:
    def __init__(self, vlen, sew, w, lam, lmul, vl):
        self.eew_c, self.eew_a, self.w, self.lmul, self.lam = sew, sew // w, w, lmul, lam
        self.k_eff = lam * w * lmul
        self.n = vl // (lam * lmul)
        self.m = (lmul * vlen // self.eew_a) // self.k_eff
        self.emul_c = self.m // lam
        self.epr_a, self.epr_c = vlen // self.eew_a, vlen // sew
        assert vl % (lam * lmul) == 0 and self.n <= self.m and self.emul_c in (1, 2, 4, 8, 16)


def c_idx(g, i, j):
    return tile_reg_idx(i * g.m + j, g.emul_c, g.lam, g.epr_c)


def ab_idx(g, row, k):
    return tile_reg_idx(row * g.k_eff + k, g.lmul, g.lam * g.w, g.epr_a)


def group_sum(rf, g, i, j, k_lo, k_hi, fa, fb, rm, vs1, vs2, fl):
    s = ZERO
    for k in range(k_lo, k_hi + 1):
        a = operand(fa, rf.read(g.eew_a, ab_idx(g, i, k), vs1), fl)
        b = operand(fb, rf.read(g.eew_a, ab_idx(g, j, k), vs2), fl)
        s = x_add(s, x_mul(a, b, fl), rm, fl)
    return s


def accumulate(acc, s, fc, rnd, rm, scale, fl):
    COV[f"group rnd={rnd}{' scaled' if scale is not None else ''}"] += 1
    if rnd == "xct":
        if scale is not None:
            s = x_mul(operand(fc, scale, fl), s, fl)
        return round_to(x_add(operand(fc, acc, fl), s, rm, fl), fc, rm, fl)
    sb = round_to(s, fc, rm, fl, rto=(rnd == "rto"))
    if scale is not None:
        sb = fp_mul(scale, sb, fc, rm, fl)
    return fp_add(acc, sb, fc, rm, fl)


def fp_gemm(rf, g, fa, fb, fc, rm, G, rnd, vs1, vs2, vd, fl):
    for j in range(g.n):
        for i in range(g.m):
            acc = rf.read(g.eew_c, c_idx(g, i, j), vd)
            for step in range(g.lmul):
                for g0 in range(0, g.lam, G):
                    k_lo = step * g.lam * g.w + g0 * g.w
                    s = group_sum(rf, g, i, j, k_lo, k_lo + G * g.w - 1, fa, fb, rm, vs1, vs2, fl)
                    acc = accumulate(acc, s, fc, rnd, rm, None, fl)
            rf.write(g.eew_c, c_idx(g, i, j), vd, acc)


def block_scales(rf, i, j, s, r, fc, rm, fl):
    pa = rf.read(16, i * r + s, 0)
    pb = rf.read(16, j * r + s, 0)
    sa = decode_scale(pa & 0xFF, fc, rm, fl)
    sb = decode_scale(pb >> 8, fc, rm, fl)
    return fp_mul(sa, sb, fc, rm, fl)


def fp_scaled_gemm(rf, g, fa, fb, fc, rm, G, rnd, bs, vs1, vs2, vd, fl):
    bsize = 16 if bs else 32
    nblk = (g.k_eff + bsize - 1) // bsize
    r = g.lam * g.eew_c // 16
    step_len = g.lam * g.w
    for j in range(g.n):
        for i in range(g.m):
            acc = rf.read(g.eew_c, c_idx(g, i, j), vd)
            nan_out = False
            for s in range(nblk):
                scale = block_scales(rf, i, j, s, r, fc, rm, fl)
                if is_nan(scale, fc):
                    COV[f"NaN combined block scale in block {'0' if s == 0 else '> 0'}"] += 1
                    nan_out = True
                    break
                blo, bhi = s * bsize, min(s * bsize + bsize, g.k_eff) - 1
                for step in range(g.lmul):
                    slo = step * step_len
                    lo, hi = max(slo, blo), min(slo + step_len - 1, bhi)
                    if lo > hi:
                        continue
                    sd_lo, sd_hi = (lo - slo) // g.w, (hi - slo) // g.w
                    for g0 in range(sd_lo, sd_hi + 1, G):
                        g_len = min(G, sd_hi - g0 + 1)
                        if g_len < G:
                            COV["shortened group"] += 1
                            if hi == bhi and bhi < slo + step_len - 1:
                                COV["shortened group at a block boundary"] += 1
                        k_lo = slo + g0 * g.w
                        S = group_sum(
                            rf, g, i, j, k_lo, k_lo + g_len * g.w - 1, fa, fb, rm, vs1, vs2, fl
                        )
                        acc = accumulate(acc, S, fc, rnd, rm, scale, fl)
            rf.write(g.eew_c, c_idx(g, i, j), vd, default_nan(fc) if nan_out else acc)


def int_scaled_gemm(rf, g, fc, rm, bs, vs1, vs2, vd, fl):
    bsize = 16 if bs else 32
    nblk = (g.k_eff + bsize - 1) // bsize
    r = g.lam * g.eew_c // 16

    def sx(v):
        return v - (1 << g.eew_a) if v >> (g.eew_a - 1) else v

    for j in range(g.n):
        for i in range(g.m):
            acc = rf.read(g.eew_c, c_idx(g, i, j), vd)
            nan_out = False
            for s in range(nblk):
                scale = block_scales(rf, i, j, s, r, fc, rm, fl)
                if is_nan(scale, fc):
                    COV[f"NaN combined block scale in block {'0' if s == 0 else '> 0'}"] += 1
                    nan_out = True
                    break
                dot = 0
                for k in range(s * bsize, min(s * bsize + bsize, g.k_eff)):
                    dot += sx(rf.read(g.eew_a, ab_idx(g, i, k), vs1)) * sx(
                        rf.read(g.eew_a, ab_idx(g, j, k), vs2)
                    )
                f = round_to((FIN, 1 if dot < 0 else 0, Fraction(abs(dot))), fc, rm, fl)
                acc = fp_add(acc, fp_mul(scale, f, fc, rm, fl), fc, rm, fl)
            rf.write(g.eew_c, c_idx(g, i, j), vd, default_nan(fc) if nan_out else acc)

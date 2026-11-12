#!/usr/bin/env python3
# Copyright (c) Openchip and Software Technologies, S.L.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Generate self-checking riscv-tests (p environment) programs for the Zvvm
FP and MX multiply-accumulate instructions, with expected values from
zvvm_ref.py.

Usage: gen_tests.py <out_dir>
Writes one .S file per (configuration, suite), and coverage.txt, which counts
the grouping, rounding and special-value events each suite exercises.
Output is deterministic. The configurations are read from cfgs/.

A failing program reports test number case * 10^7 + byte offset * 256 + byte
value for a wrong C byte, and case * 10^7 + 9 * 10^6 + fflags for wrong flags.
Run with: rake test:zvvm_fp CONFIG=<configuration>
"""

import os
import random
import sys

import zvvm_ref as R

FUNCT6 = {
    "vfmmacc": 0b010100,
    "vfwmmacc": 0b010101,
    "vfqmmacc": 0b010110,
    "vf8wmmacc": 0b010111,
    "vfwimmacc": 0b111001,
    "vfqimmacc": 0b111010,
    "vf8wimmacc": 0b111011,
}
W_OF = {
    "vfmmacc": 1,
    "vfwmmacc": 2,
    "vfqmmacc": 4,
    "vf8wmmacc": 8,
    "vfwimmacc": 2,
    "vfqimmacc": 4,
    "vf8wimmacc": 8,
}
VS1, VS2, VD = 8, 16, 24


def encode(inst, vm):
    f3 = 0b000 if "imm" in inst else 0b001
    return (
        (FUNCT6[inst] << 26)
        | (vm << 25)
        | (VS2 << 20)
        | (VS1 << 15)
        | (f3 << 12)
        | (VD << 7)
        | 0x57
    )


def vtype(sew, lmul, lam, bs, alt_a, alt_b, alt):
    sew_enc = {8: 0, 16: 1, 32: 2, 64: 3}[sew]
    lmul_enc = {1: 0, 2: 1, 4: 2, 8: 3}[lmul]
    lam_enc = lam.bit_length()
    return (
        (lam_enc << 60)
        | (bs << 59)
        | (alt_a << 58)
        | (alt_b << 57)
        | (alt << 8)
        | (1 << 7)
        | (1 << 6)
        | (sew_enc << 3)
        | lmul_enc
    )


class Config:
    def __init__(self, name, vlen, disclosure):
        self.name, self.vlen, self.disclosure = name, vlen, disclosure

    def g_rnd(self, sew, w, lam):
        return self.disclosure(sew, w, lam)


def rand_fp(rng, f, spread, special):
    """Random encoding of f: exponents near the bias +- spread, sometimes special."""
    eb, mb, ieee = R.FORMATS[f]
    bias = (1 << (eb - 1)) - 1
    width = 1 + eb + mb
    r = rng.random()
    if r < special:
        choices = [0, 1 << (width - 1), 1]  # +0, -0, smallest subnormal
        if ieee:
            choices += [
                R.pack(f, 0, (1 << eb) - 1, 0),
                R.default_nan(f),
                R.pack(f, 1, (1 << eb) - 1, 1),
            ]
        choices += [
            R.pack(f, rng.randrange(2), (1 << eb) - 2 if ieee else (1 << eb) - 1, (1 << mb) - 2)
        ]
        return rng.choice(choices)
    if f in ("E2M1", "E4M3", "E5M2"):
        v = rng.getrandbits(width)
        while R.decode(f, v)[0] != R.FIN:
            v = rng.getrandbits(width)
        return v
    e = max(1, min((1 << eb) - 2, bias + rng.randint(-spread, spread)))
    return R.pack(f, rng.randrange(2), e, rng.getrandbits(mb))


def make_case(
    rng,
    cfg,
    inst,
    vm,
    sew,
    lam,
    lmul,
    n,
    bs,
    alt_a,
    alt_b,
    alt,
    frm,
    spread=4,
    special=0.0,
    scale_mode="near",
    title="",
    setup=None,
):
    w = W_OF[inst]
    g = R.Geom(cfg.vlen, sew, w, lam, lmul, n * lam * lmul)
    rf = R.VRegs(cfg.vlen)
    vlenb = cfg.vlen // 8
    is_int = "imm" in inst
    fa = R.fmt_of(sew // w, alt_a)
    fb = R.fmt_of(sew // w, alt_b)
    fc = R.fmt_of(sew, alt)
    eew_a = sew // w
    # fill A, B: random elements over the whole groups
    for base in (VS1, VS2):
        f = fa if base == VS1 else fb
        for idx in range(lmul * cfg.vlen // eew_a):
            if is_int:
                val = rng.getrandbits(eew_a)
            else:
                val = rand_fp(rng, f, spread, special)
            if eew_a == 4:
                bitoff = base * vlenb * 8 + idx * 4
                rf.b[bitoff // 8] |= val << (bitoff % 8)
            else:
                rf.write(eew_a, idx, base, val)
    # C: active elements random; tail columns hold a signaling NaN (must not change or raise NV)
    for flat in range(g.emul_c * cfg.vlen // sew):
        rf.write(sew, flat, VD, rand_fp(rng, fc, spread + 2, special / 2))
    for i in range(g.m):
        for j in range(g.n, g.m):
            rf.write(sew, R.c_idx(g, i, j), VD, R.pack(fc, 0, (1 << R.FORMATS[fc][0]) - 1, 1))
    # v0 scales
    for p in range(cfg.vlen // 16):
        if scale_mode == "near":
            ea, eb_ = 127 + rng.randint(-3, 3), 127 + rng.randint(-3, 3)
        elif scale_mode == "wide":
            ea, eb_ = rng.randrange(256), rng.randrange(256)
        else:
            ea, eb_ = 127, 127
        rf.write(16, p, 0, ea | (eb_ << 8))
    if setup:
        setup(rf, g)
    before = bytes(rf.b)
    fl = R.Flags()
    rm = R.RMS[frm]
    G, rnd = cfg.g_rnd(sew, w, lam)
    if is_int:
        R.int_scaled_gemm(rf, g, fc, rm, bs, VS1, VS2, VD, fl)
    elif vm == 0:
        R.fp_scaled_gemm(rf, g, fa, fb, fc, rm, G, rnd, bs, VS1, VS2, VD, fl)
    else:
        R.fp_gemm(rf, g, fa, fb, fc, rm, G, rnd, VS1, VS2, VD, fl)
    return {
        "title": f"{inst} vm={vm} SEW={sew} W={w} lambda={lam} LMUL={lmul} N={n} bs={bs} "
        f"{fa}x{fb}->{fc} {rm} G={G} rnd={rnd} {title}",
        "before": before,
        "after": bytes(rf.b),
        "vlenb": vlenb,
        "vtype": vtype(sew, lmul, lam, bs, alt_a, alt_b, alt),
        "vl": n * lam * lmul,
        "frm": frm,
        "inst": encode(inst, vm),
        "flags": fl.v,
    }


def make_illegal(cfg, inst, vm, sew, lam, lmul, n, bs, alt_a, alt_b, alt, frm, title):
    return {
        "title": f"illegal: {inst} {title}",
        "illegal": True,
        "vlenb": cfg.vlen // 8,
        "vtype": vtype(sew, lmul, lam, bs, alt_a, alt_b, alt),
        "vl": n * lam * lmul,
        "frm": frm,
        "inst": encode(inst, vm),
    }


def emit(path, cases):
    out = []
    o = out.append
    o("# Generated by gen_tests.py; expected values from zvvm_ref.py.")
    o('#include "riscv_test.h"')
    o('#include "test_macros.h"')
    o("")
    o("RVTEST_RV64UV")
    o("RVTEST_CODE_BEGIN")
    for k, c in enumerate(cases, start=1):
        vlenb = c["vlenb"]
        o(f"  # case {k}: {c['title']}")
        o(f"  li TESTNUM, {k}")
        if not c.get("illegal"):
            # vle8.v/vse8.v have IDL in UDB, but write only register vd, so
            # load one register at a time
            o(f"  li t1, {vlenb}")
            o("  vsetvli t2, t1, e8, m1, ta, ma")
            o(f"  la a0, data{k}")
            for reg in (
                list(range(VS1, VS1 + 8))
                + list(range(VS2, VS2 + 8))
                + list(range(VD, VD + 8))
                + [0]
            ):
                o(f"  vle8.v v{reg}, (a0)")
                o(f"  addi a0, a0, {vlenb}")
        o(f"  li t0, 0x{c['vtype']:x}")
        o(f"  li t1, {c['vl']}")
        o("  vsetvl t2, t1, t0")
        o("  bne t1, t2, fail")
        o(f"  csrwi frm, {c['frm']}")
        o("  csrw fflags, x0")
        o("  li s1, 0")
        o(f"  .word 0x{c['inst']:08x}")
        if c.get("illegal"):
            o("  li t1, 1")
            o("  bne s1, t1, fail")
            continue
        o("  bnez s1, fail")
        o("  csrr t3, fflags")
        o(f"  li t4, {c['flags']}")
        # on a flags mismatch, the test number is case * 10^7 + 9 * 10^6 + fflags
        o(f"  li TESTNUM, {k * 10**7 + 9 * 10**6}")
        o("  add TESTNUM, TESTNUM, t3")
        o("  bne t3, t4, fail")
        o(f"  li TESTNUM, {k * 10**7}")
        o(f"  li t1, {vlenb}")
        o("  vsetvli t2, t1, e8, m1, ta, ma")
        o("  la a0, outbuf")
        for reg in range(VD, VD + 8):
            o(f"  vse8.v v{reg}, (a0)")
            o(f"  addi a0, a0, {vlenb}")
        o("  la a0, outbuf")
        o(f"  la a1, expect{k}")
        o(f"  li a2, {8 * vlenb}")
        o("  jal ra, check_bytes")
    o("  TEST_PASSFAIL")
    o("")
    # on a byte mismatch, the test number is case * 10^7 + index * 256 + byte
    o("check_bytes:")
    o("  mv a3, a0")
    o("1:")
    o("  lbu t5, 0(a0)")
    o("  lbu t6, 0(a1)")
    o("  sub t4, a0, a3")
    o("  slli t4, t4, 8")
    o("  add t4, t4, t5")
    o("  add TESTNUM, TESTNUM, t4")
    o("  bne t5, t6, fail")
    o("  sub TESTNUM, TESTNUM, t4")
    o("  addi a0, a0, 1")
    o("  addi a1, a1, 1")
    o("  addi a2, a2, -1")
    o("  bnez a2, 1b")
    o("  ret")
    o("")
    o(".align 2")
    o(".global mtvec_handler")
    o("mtvec_handler:")
    o("  csrr t5, mcause")
    o("  li t6, 2")
    o("  bne t5, t6, fail")
    o("  li s1, 1")
    o("  csrr t5, mepc")
    o("  addi t5, t5, 4")
    o("  csrw mepc, t5")
    o("  mret")
    o("")
    o("RVTEST_CODE_END")
    o("  .data")
    o("RVTEST_DATA_BEGIN")
    o(".align 4")
    o(f"outbuf: .fill {8 * max(c['vlenb'] for c in cases)}, 1, 0")
    for k, c in enumerate(cases, start=1):
        if c.get("illegal"):
            continue
        vlenb = c["vlenb"]
        b = c["before"]
        data = (
            b[VS1 * vlenb : (VS1 + 8) * vlenb]
            + b[VS2 * vlenb : (VS2 + 8) * vlenb]
            + b[VD * vlenb : (VD + 8) * vlenb]
            + b[0:vlenb]
        )
        exp = c["after"][VD * vlenb : (VD + 8) * vlenb]
        o(".align 4")
        o(f"data{k}:")
        for i in range(0, len(data), 16):
            o("  .byte " + ", ".join(str(x) for x in data[i : i + 16]))
        o(f"expect{k}:")
        for i in range(0, len(exp), 16):
            o("  .byte " + ", ".join(str(x) for x in exp[i : i + 16]))
    o("RVTEST_DATA_END")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


# ─── Suites ───────────────────────────────────────────────────────────────

# (inst, sew, alt_a, alt_b, alt) for every legal FP row; vm=0 allowed when mx
FP_ROWS = [
    ("vfmmacc", 16, 0, 0, 0, False),
    ("vfmmacc", 16, 1, 1, 1, False),
    ("vfmmacc", 16, 0, 1, 0, False),
    ("vfmmacc", 32, 0, 0, 0, False),
    ("vfmmacc", 64, 0, 0, 0, False),
    ("vfmmacc", 8, 0, 1, 1, False),
    ("vfmmacc", 8, 1, 1, 1, False),
    ("vfwmmacc", 8, 0, 0, 1, True),
    ("vfwmmacc", 16, 0, 1, 0, True),
    ("vfwmmacc", 16, 1, 1, 1, True),
    ("vfwmmacc", 16, 0, 0, 1, True),
    ("vfwmmacc", 32, 0, 0, 0, False),
    ("vfwmmacc", 32, 1, 0, 0, False),
    ("vfwmmacc", 64, 0, 0, 0, False),
    ("vfqmmacc", 16, 0, 0, 0, True),
    ("vfqmmacc", 16, 0, 0, 1, True),
    ("vfqmmacc", 32, 1, 0, 0, True),
    ("vfqmmacc", 64, 1, 1, 0, False),
    ("vfqmmacc", 64, 0, 0, 0, False),
    ("vf8wmmacc", 32, 0, 0, 0, True),
    ("vf8wmmacc", 64, 0, 1, 0, True),
]
INT_ROWS = [
    ("vfwimmacc", 16, 0),
    ("vfwimmacc", 16, 1),
    ("vfqimmacc", 16, 0),
    ("vfqimmacc", 32, 0),
    ("vf8wimmacc", 32, 0),
    ("vf8wimmacc", 64, 0),
]


def choose(rng, cfg, lams, sew, w, mx, bs):
    """Pick (lambda, LMUL) with EMUL_C <= 8 and the MX capacity rules."""
    opts = []
    for lam in lams[sew]:
        emul_c = cfg.vlen // (sew * lam * lam)
        if emul_c > 8 or emul_c < 1:
            continue
        for lmul in (1, 2, 4, 8):
            if mx and (sew * lam < 16 or (bs and w * lmul > sew)):
                continue
            opts.append((lam, lmul))
    return rng.choice(opts) if opts else None


def suite(cfg, lams, seed, per_row):
    rng = random.Random(seed)
    cases = []
    for inst, sew, aa, ab, alt, mx in FP_ROWS:
        w = W_OF[inst]
        for rep in range(per_row):
            vm = 0 if (mx and rep % 2 == 1) else 1
            bs = rng.randrange(2) if vm == 0 else 0
            pick = choose(rng, cfg, lams, sew, w, vm == 0, bs)
            if pick is None:
                continue
            lam, lmul = pick
            m = (lmul * cfg.vlen // (sew // w)) // (lam * w * lmul)
            n = rng.randint(1, m)
            special = 0.08 if rep % 3 == 2 else 0.0
            scale_mode = "wide" if (vm == 0 and rep % 4 == 3) else "near"
            cases.append(
                make_case(
                    rng,
                    cfg,
                    inst,
                    vm,
                    sew,
                    lam,
                    lmul,
                    n,
                    bs,
                    aa,
                    ab,
                    alt,
                    rng.randrange(5),
                    spread=3,
                    special=special,
                    scale_mode=scale_mode,
                )
            )
    for inst, sew, alt in INT_ROWS:
        w = W_OF[inst]
        for rep in range(per_row):
            bs = rep % 2
            pick = choose(rng, cfg, lams, sew, w, True, bs)
            if pick is None:
                continue
            lam, lmul = pick
            m = (lmul * cfg.vlen // (sew // w)) // (lam * w * lmul)
            cases.append(
                make_case(
                    rng,
                    cfg,
                    inst,
                    0,
                    sew,
                    lam,
                    lmul,
                    rng.randint(1, m),
                    bs,
                    0,
                    0,
                    alt,
                    rng.randrange(5),
                    scale_mode="wide" if rep % 3 == 2 else "near",
                )
            )
    return cases


def illegal_suite(cfg, lams):
    out = []
    # reserved frm values
    for frm in (5, 6, 7):
        out.append(
            make_illegal(cfg, "vfmmacc", 1, 32, lams[32][0], 1, 1, 0, 0, 0, 0, frm, f"frm={frm}")
        )
        out.append(
            make_illegal(cfg, "vfwimmacc", 0, 16, lams[16][0], 1, 1, 0, 0, 0, 0, frm, f"frm={frm}")
        )
    # E4M3 accumulator (OFP8 -> E4M3 and OFP4 -> E4M3)
    lam8 = lams[8][-1]
    out.append(make_illegal(cfg, "vfmmacc", 1, 8, lam8, 1, 1, 0, 0, 0, 0, 0, "E4M3 C"))
    out.append(make_illegal(cfg, "vfwmmacc", 1, 8, lam8, 1, 1, 0, 0, 0, 0, 0, "OFP4 -> E4M3 C"))
    out.append(make_illegal(cfg, "vfwmmacc", 0, 8, lam8, 1, 1, 0, 0, 0, 0, 0, "MX OFP4 -> E4M3 C"))
    return out


CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "cfgs")


def load_cfg(name):
    """Read VLEN, SUPPORTED_LAMBDA_SEW* and the ZVVM_FP disclosure from cfgs/<name>.yaml."""
    import re

    with open(os.path.join(CFG_DIR, name + ".yaml")) as fh:
        text = fh.read()
    vlen = int(re.search(r"^  VLEN: (\d+)", text, re.MULTILINE)[1])
    lams = {
        sew: [
            int(x)
            for x in re.search(rf"SUPPORTED_LAMBDA_SEW{sew}: \[([^\]]*)\]", text)[1].split(",")
        ]
        for sew in (8, 16, 32, 64)
    }
    disc = {}
    for m in re.finditer(r"ZVVM_FP_GROUPING_SEW(\d+)_W(\d+): \[([^\]]*)\]", text):
        sew, w = int(m[1]), int(m[2])
        gs = [int(x) for x in m[3].split(",")]
        rs = [
            x.strip()
            for x in re.search(rf"ZVVM_FP_RND_SEW{sew}_W{w}: \[([^\]]*)\]", text)[1].split(",")
        ]
        for k in range(7):
            disc[(sew, w, 1 << k)] = (gs[k], rs[k])
    return Config(name, vlen, lambda sw, w, l: disc[(sw, w, l)]), lams


def shortened_cases(cfg, lams, seed):
    """MX groups cut by a block boundary: G x W larger than the block size."""
    rng = random.Random(seed)
    out = []
    for inst, sew, aa, ab, alt, mx in FP_ROWS:
        if not mx:
            continue
        w = W_OF[inst]
        for lam in lams[sew]:
            emul_c = cfg.vlen // (sew * lam * lam)
            G, _ = cfg.g_rnd(sew, w, lam)
            if emul_c > 8 or G * w <= 16:
                continue
            for bs in (1, 0):
                if G * w <= (16 if bs else 32):
                    continue
                for lmul in (1, 2, 4, 8):
                    if bs and w * lmul > sew:
                        continue
                    m = (lmul * cfg.vlen // (sew // w)) // (lam * w * lmul)
                    out.append(
                        make_case(
                            rng,
                            cfg,
                            inst,
                            0,
                            sew,
                            lam,
                            lmul,
                            rng.randint(1, m),
                            bs,
                            aa,
                            ab,
                            alt,
                            rng.randrange(5),
                            spread=3,
                            title="[shortened groups]",
                        )
                    )
    return out


def long_sum_cases(cfg):
    """FP64 vfmmacc, lambda=2, G=2, rnd=xct: C = -2^1023 and the group's products
    2^1023 x 1 and 2^-1074 x 2^-1074. C + S = 2^-2148 rounds by frm."""
    out = []
    for frm in range(5):

        def setup(rf, g):
            for i in range(g.m):
                for k in range(g.k_eff):
                    rf.write(64, R.ab_idx(g, i, k), VS1, 0)
                for j in range(g.m):
                    rf.write(64, R.c_idx(g, i, j), VD, R.pack("FP64", 1, 2046, 0) if j < g.n else 0)
            for j in range(g.m):
                for k in range(g.k_eff):
                    rf.write(64, R.ab_idx(g, j, k), VS2, 0)
            for i in range(g.m):
                rf.write(64, R.ab_idx(g, i, 0), VS1, R.pack("FP64", 0, 2046, 0))  # 2^1023
                rf.write(64, R.ab_idx(g, i, 1), VS1, 1)  # 2^-1074
            for j in range(g.m):
                rf.write(64, R.ab_idx(g, j, 0), VS2, R.pack("FP64", 0, 1023, 0))  # 1.0
                rf.write(64, R.ab_idx(g, j, 1), VS2, 1)  # 2^-1074

        out.append(
            make_case(
                random.Random(frm),
                cfg,
                "vfmmacc",
                1,
                64,
                2,
                1,
                2,
                0,
                0,
                0,
                0,
                frm,
                title="[long sum: -2^1023 + 2^1023 + 2^-2148]",
                setup=setup,
            )
        )
    return out


# ─── Directed cases ──────────────────────────────────────────────────────


def geoms(cfg, lams, sew, w, mx=False, bs=0):
    """Every (lambda, LMUL, G, rnd) the configuration allows for (SEW, W)."""
    out = []
    for lam in lams[sew]:
        emul_c = cfg.vlen // (sew * lam * lam)
        if emul_c > 8 or emul_c < 1:
            continue
        G, rnd = cfg.g_rnd(sew, w, lam)
        if G == 0:
            continue
        for lmul in (1, 2, 4, 8):
            if mx and (sew * lam < 16 or (bs and w * lmul > sew)):
                continue
            out.append((lam, lmul, G, rnd))
    return out


def pattern(a_vals, b_vals, c_val, eew_a, scale=None):
    """Return a make_case setup that zeroes A and B, writes a_vals[k] into
    column k of every row of A and b_vals[k] into column k of every row of
    B^T, writes c_val into every active C element, and, when scale is given,
    sets the E8M0 scale pair of row m and block s to scale(m, s)."""

    def setup(rf, g):
        for row in range(g.m):
            for k in range(g.k_eff):
                rf.write(eew_a, R.ab_idx(g, row, k), VS1, a_vals.get(k, 0))
                rf.write(eew_a, R.ab_idx(g, row, k), VS2, b_vals.get(k, 0))
            for j in range(g.n):
                rf.write(g.eew_c, R.c_idx(g, row, j), VD, c_val)
        if scale is not None:
            r = g.lam * g.eew_c // 16
            for m in range(g.m):
                for s in range(r):
                    ea, eb = scale(m, s)
                    rf.write(16, m * r + s, 0, ea | (eb << 8))

    return setup


def directed(cfg, inst, vm, sew, geom, alt_a, alt_b, alt, frm, title, setup, bs=0):
    lam, lmul, _, _ = geom
    w = W_OF[inst]
    m = (lmul * cfg.vlen // (sew // w)) // (lam * w * lmul)
    return make_case(
        random.Random(0),
        cfg,
        inst,
        vm,
        sew,
        lam,
        lmul,
        m,
        bs,
        alt_a,
        alt_b,
        alt,
        frm,
        title=f"[{title}]",
        setup=setup,
    )


def neg(f, bits):
    return bits | (1 << (sum(R.FORMATS[f][:2])))


def fmax(f):
    eb, mb, _ = R.FORMATS[f]
    return R.pack(f, 0, (1 << eb) - 2, (1 << mb) - 1)


def one(f):
    eb = R.FORMATS[f][0]
    return R.pack(f, 0, (1 << (eb - 1)) - 1, 0)


def long_accumulator_cases(cfg, lams):
    """FP64 group of 4 products with rnd=xct: +MAX*MAX - MAX*MAX + 2^-1074*2^-1074 (+-2^-2148).
    The products reach 2^2048, beyond the FP64 range, so the sum needs the long accumulator.
    C = +0, and C + S = +-2^-2148 rounds to a zero or the smallest subnormal by frm."""
    out = []
    for geom in geoms(cfg, lams, 64, 1):
        if geom[1] != 1 or geom[2] < 3 or geom[3] != "xct":
            continue
        for sign in (0, 1):
            tiny = neg("FP64", 1) if sign else 1
            a = {0: fmax("FP64"), 1: fmax("FP64"), 2: 1}
            b = {0: fmax("FP64"), 1: neg("FP64", fmax("FP64")), 2: tiny}
            for frm in range(5):
                out.append(
                    directed(
                        cfg,
                        "vfmmacc",
                        1,
                        64,
                        geom,
                        0,
                        0,
                        0,
                        frm,
                        f"long sum: MAX^2 - MAX^2 {'-' if sign else '+'} 2^-2148",
                        pattern(a, b, 0, 64),
                    )
                )
        break
    return out


# (inst, SEW, alt_a, alt_b, alt) rows used by the directed special-value cases
SPECIAL_ROWS = [
    ("vfwmmacc", 16, 0, 0, 0, "E4M3xE4M3->FP16"),
    ("vfwmmacc", 16, 1, 1, 0, "E5M2xE5M2->FP16"),
    ("vfwmmacc", 32, 0, 0, 0, "FP16xFP16->FP32"),
    ("vfmmacc", 32, 0, 0, 0, "FP32xFP32->FP32"),
    ("vfmmacc", 64, 0, 0, 0, "FP64xFP64->FP64"),
    ("vfqmmacc", 64, 1, 1, 0, "BF16xBF16->FP64"),
]


def zero_sum_cases(cfg, lams):
    """Exact-zero sums under RNE and RDN for every (lambda, G, rnd) the configuration
    discloses: two products that cancel inside a group with C = -0, and a product
    that cancels C."""
    out = []
    for inst, sew, aa, ab, alt, name in SPECIAL_ROWS:
        w = W_OF[inst]
        fa, fc = R.fmt_of(sew // w, aa), R.fmt_of(sew, alt)
        seen = set()
        for geom in geoms(cfg, lams, sew, w):
            if geom[1] != 1 or (geom[2], geom[3]) in seen:
                continue
            seen.add((geom[2], geom[3]))
            x = R.pack(fa, 0, (1 << (R.FORMATS[fa][0] - 1)) - 1 + 2, 3)  # 1.x * 4
            for frm in (0, 2):
                if geom[2] * w >= 2:
                    out.append(
                        directed(
                            cfg,
                            inst,
                            1,
                            sew,
                            geom,
                            aa,
                            ab,
                            alt,
                            frm,
                            f"{name} x*1 - x*1 in one group, C = -0",
                            pattern(
                                {0: x, 1: x},
                                {0: one(fa), 1: neg(fa, one(fa))},
                                neg(fc, 0),
                                sew // w,
                            ),
                        )
                    )
                xc = R.round_to((R.FIN, 0, R.decode(fa, x)[2]), fc, "RNE", R.Flags())
                out.append(
                    directed(
                        cfg,
                        inst,
                        1,
                        sew,
                        geom,
                        aa,
                        ab,
                        alt,
                        frm,
                        f"{name} C = x, product -x",
                        pattern({0: x}, {0: neg(fa, one(fa))}, xc, sew // w),
                    )
                )
    return out


def special_value_cases(cfg, lams):
    """Signaling and quiet NaN operands, infinities and overflow."""
    out = []
    for inst, sew, aa, ab, alt, name in SPECIAL_ROWS:
        w = W_OF[inst]
        fa, fc = R.fmt_of(sew // w, aa), R.fmt_of(sew, alt)
        eb = R.FORMATS[fa][0]
        gs = [gm for gm in geoms(cfg, lams, sew, w) if gm[1] == 1]
        geom = max(gs, key=lambda gm: gm[2])
        inf_a = R.pack(fa, 0, (1 << eb) - 1, 0)
        snan_a = R.pack(fa, 0, (1 << eb) - 1, 1)
        qnan_a = R.default_nan(fa)
        cases = [
            ("signaling NaN in A", {0: snan_a, 1: one(fa)}, {0: one(fa), 1: one(fa)}, 0),
            ("quiet NaN in B", {0: one(fa)}, {0: qnan_a}, 0),
            ("inf x 0", {0: inf_a}, {0: 0}, 0),
            (
                "inf x 1 + C = -inf",
                {0: inf_a},
                {0: one(fa)},
                neg(fc, R.pack(fc, 0, (1 << R.FORMATS[fc][0]) - 1, 0)),
            ),
            ("inf - inf in one group", {0: inf_a, 1: inf_a}, {0: one(fa), 1: neg(fa, one(fa))}, 0),
            ("MAX x MAX + C = MAX", {0: fmax(fa)}, {0: fmax(fa)}, fmax(fc)),
            (
                "C = -MAX, -MAX x 2",
                {0: neg(fa, fmax(fa))},
                {0: R.pack(fa, 0, (1 << (eb - 1)), 0)},
                neg(fc, fmax(fc)),
            ),
        ]
        if fa == "E4M3":
            # no infinity and a single quiet NaN encoding per sign, S.1111.111; MAX is 448
            cases = [
                ("NaN 0x7F in A", {0: 0x7F, 1: one(fa)}, {0: one(fa), 1: one(fa)}, 0),
                ("NaN 0xFF in B", {0: one(fa)}, {0: 0xFF}, 0),
                ("448 x 448 + C = MAX", {0: 0x7E}, {0: 0x7E}, fmax(fc)),
            ]
        for title, a, b, c in cases:
            if "in one group" in title and geom[2] * w < 2:
                continue
            for frm in (0, 1, 2, 3):
                out.append(
                    directed(
                        cfg,
                        inst,
                        1,
                        sew,
                        geom,
                        aa,
                        ab,
                        alt,
                        frm,
                        f"{name} {title}",
                        pattern(a, b, c, sew // w),
                    )
                )
    return out


def nan_scale_cases(cfg, lams):
    """Microscaled E5M2 x E5M2 -> FP16 (vfwmmacc SEW=16) with two or more blocks.
    Row 0 of A has scale 0xFF in block 1 and a signaling NaN in block 1, which must
    not be evaluated (no NV). Row 1 of A has a signaling NaN in block 0 (NV, NaN).
    Row 2 has scales that overflow FP16 in block 0."""
    out = []
    for bs in (1, 0):
        bsize = 16 if bs else 32
        for geom in geoms(cfg, lams, 16, 2, mx=True, bs=bs):
            lam, lmul, _, _ = geom
            k_eff = lam * 2 * lmul
            if k_eff < 2 * bsize:
                continue
            m = (lmul * cfg.vlen // 8) // k_eff
            if m < 3:
                continue

            def setup(rf, g, bsize=bsize):
                pattern(
                    {k: one("E5M2") for k in range(g.k_eff)},
                    {k: one("E5M2") for k in range(g.k_eff)},
                    0,
                    8,
                    scale=lambda mm, s: (127, 127),
                )(rf, g)
                r = g.lam * g.eew_c // 16
                rf.write(8, R.ab_idx(g, 0, bsize + 1), VS1, R.pack("E5M2", 0, 31, 1))
                rf.write(16, 0 * r + 1, 0, 0xFF | (127 << 8))
                rf.write(8, R.ab_idx(g, 1, 3), VS1, R.pack("E5M2", 0, 31, 1))
                rf.write(16, 2 * r + 0, 0, 254 | (127 << 8))

            for frm in (0, 1):
                out.append(
                    directed(
                        cfg,
                        "vfwmmacc",
                        0,
                        16,
                        geom,
                        1,
                        1,
                        0,
                        frm,
                        f"NaN and overflowing block scales, bs={bs}",
                        setup,
                        bs=bs,
                    )
                )
            break
    return out


def directed_cases(cfg, lams):
    return zero_sum_cases(cfg, lams) + special_value_cases(cfg, lams) + nan_scale_cases(cfg, lams)


def coverage(name, cases, cov):
    feats = {}
    for c in cases:
        t = c["title"]
        if c.get("illegal"):
            continue
        parts = t.split()
        key = " ".join(
            p for p in parts if p.startswith(("G=", "rnd=", "bs=")) and not p.endswith("]")
        )
        feats[key] = feats.get(key, 0) + 1
    lines = [f"{name}: {len(cases)} cases"]
    lines += [f"  {k}: {v}" for k, v in sorted(feats.items())]
    lines += [f"  {k}: {v}" for k, v in sorted(cov.items())]
    return "\n".join(lines)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    report = []

    def write(name, build):
        R.COV.clear()
        cases = build()
        emit(os.path.join(out_dir, name + ".S"), cases)
        report.append(coverage(name, cases, R.COV))

    c128, l128 = load_cfg("rv64-ime")
    write("zvvm_fp_rv64-ime", lambda: suite(c128, l128, 1, 4) + directed_cases(c128, l128))
    write("zvvm_fp_illegal_rv64-ime", lambda: illegal_suite(c128, l128))
    grp, lgrp = load_cfg("rv64-ime-v256-grp")
    write(
        "zvvm_fp_rv64-ime-v256-grp",
        lambda: (
            long_sum_cases(grp)
            + suite(grp, lgrp, 2, 4)
            + zero_sum_cases(grp, lgrp)
            + special_value_cases(grp, lgrp)
            + nan_scale_cases(grp, lgrp)
        ),
    )
    c256, l256 = load_cfg("rv64-ime-v256")
    write("zvvm_fp_rv64-ime-v256", lambda: suite(c256, l256, 3, 3) + directed_cases(c256, l256))
    c1k, l1k = load_cfg("rv64-ime-v1024-grp")
    write(
        "zvvm_fp_rv64-ime-v1024-grp",
        lambda: (
            shortened_cases(c1k, l1k, 4)
            + suite(c1k, l1k, 5, 2)
            + long_accumulator_cases(c1k, l1k)
            + zero_sum_cases(c1k, l1k)
            + special_value_cases(c1k, l1k)
            + nan_scale_cases(c1k, l1k)
        ),
    )
    with open(os.path.join(out_dir, "coverage.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()

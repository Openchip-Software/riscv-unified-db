#!/usr/bin/env python3
# Copyright (c) Openchip and Software Technologies, S.L.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Generate self-checking riscv-tests (p environment) programs for the Zvvm
integer multiply-accumulates (vmmacc, vwmmacc, vqmmacc, v8wmmacc) and the tile
loads and stores (vmtl, vmttl, vmts, vmtts).

Usage: gen_int_tile.py <out_dir>
Writes zvvm_int_<configuration>.S and zvvm_tile_<configuration>.S for each
configuration that gen_tests.py covers. Output is deterministic.

Expected integer tiles are exact sums modulo 2^SEW. Expected tile layouts
follow the element offsets of the specification and tile_reg_idx from
zvvm_ref.py. A failing program reports test number case * 10^7 + byte offset
* 256 + byte value for the first wrong byte.
Run with: rake test:zvvm CONFIG=<configuration>
"""

import os
import random
import sys

import gen_tests as G
import zvvm_ref as R

VS1, VS2, VD = G.VS1, G.VS2, G.VD
INT_OPCODE = {1: 0b1110001, 2: 0b1110011, 4: 0b1110101, 8: 0b1110111}  # bits 31:25
INT_NAME = {1: "vmmacc", 2: "vwmmacc", 4: "vqmmacc", 8: "v8wmmacc"}
SEWS = (8, 16, 32, 64)
LMULS = (1, 2, 4, 8)


def int_encode(w):
    return (INT_OPCODE[w] << 25) | (VS2 << 20) | (VS1 << 15) | (VD << 7) | 0x57


def lam_enc(lam):
    return lam.bit_length()


# Integer multiply-accumulate


def int_case(rng, cfg, w, sew, lam, lmul, n, alt_a, alt_b):
    vl = n * lam * lmul
    g = R.Geom(cfg.vlen, sew, w, lam, lmul, vl)
    rf = R.VRegs(cfg.vlen)
    vlenb = cfg.vlen // 8
    for i in range(len(rf.b)):
        rf.b[i] = rng.getrandbits(8)
    before = bytes(rf.b)

    def elem(base, row, k, signed):
        idx = R.ab_idx(g, row, k)
        bitoff = base * vlenb * 8 + idx * g.eew_a
        if g.eew_a == 4:
            v = (rf.b[bitoff // 8] >> (bitoff % 8)) & 0xF
        else:
            v = rf.read(g.eew_a, idx, base)
        if signed and v >> (g.eew_a - 1):
            v -= 1 << g.eew_a
        return v

    for j in range(g.n):
        for i in range(g.m):
            acc = rf.read(sew, R.c_idx(g, i, j), VD)
            for k in range(g.k_eff):
                acc += elem(VS1, i, k, alt_a == 0) * elem(VS2, j, k, alt_b == 0)
            rf.write(sew, R.c_idx(g, i, j), VD, acc)
    return {
        "title": f"{INT_NAME[w]} SEW={sew} W={w} lambda={lam} LMUL={lmul} N={n} "
        f"altfmt_A={alt_a} altfmt_B={alt_b}",
        "before": before,
        "after": bytes(rf.b),
        "vlenb": vlenb,
        "vtype": G.vtype(sew, lmul, lam, 0, alt_a, alt_b, 0),
        "vl": vl,
        "frm": 0,
        "inst": int_encode(w),
        "flags": 0,
    }


def int_geoms(cfg, lams, sew, w):
    out = []
    for lam in lams[sew]:
        for lmul in LMULS:
            try:
                g = R.Geom(cfg.vlen, sew, w, lam, lmul, lam * lmul)
            except (AssertionError, ZeroDivisionError):
                continue
            if g.eew_a < 4 or g.m == 0:
                continue
            # N is at most M, and VL = N * lambda * LMUL is at most VLMAX
            out.append((lam, lmul, min(g.m, (cfg.vlen // sew) // lam)))
    return out


def int_suite(cfg, lams):
    rng = random.Random(7)
    cases = []
    for w in (1, 2, 4, 8):
        for sew in SEWS:
            if sew // w < 4:
                continue
            for lam, lmul, m in int_geoms(cfg, lams, sew, w):
                # VD + EMUL_C must stay in v24..v31 and the A/B groups in their 8 registers
                if R.Geom(cfg.vlen, sew, w, lam, lmul, lam * lmul).emul_c > 8:
                    continue
                for alt_a, alt_b in ((0, 0), (1, 1), (0, 1), (1, 0)):
                    n = rng.choice(sorted({1, m, rng.randint(1, m)}))
                    cases.append(int_case(rng, cfg, w, sew, lam, lmul, n, alt_a, alt_b))
    # input elements narrower than 4 bits are illegal
    for w, sew in ((4, 8), (8, 16), (8, 8)):
        cases.append(
            {
                "title": f"illegal: {INT_NAME[w]} SEW={sew} gives input elements of {sew // w} bits",
                "illegal": True,
                "vlenb": cfg.vlen // 8,
                "vtype": G.vtype(sew, 1, 1, 0, 0, 0, 0),
                "vl": 1,
                "frm": 0,
                "inst": int_encode(w),
            }
        )
    return cases


# Tile loads and stores


def tile_encode(store, transposed, lam_field, vm, vreg):
    return (
        (lam_field << 29)
        | ((0b101 if transposed else 0b100) << 26)
        | (vm << 25)
        | (11 << 20)  # xs2 = a1
        | (10 << 15)  # xs1 = a0
        | (0b111 << 12)
        | (vreg << 7)
        | (0b0100111 if store else 0b0000111)
    )


def tile_case(rng, cfg, store, transposed, sew, lmul, vt_lam, lam_field, vm, n_lines, ld_extra):
    lam = (1 << lam_field) >> 1 if lam_field else vt_lam
    vlenb = cfg.vlen // 8
    epr = cfg.vlen // sew
    linesize = lam * lmul
    vl = n_lines * linesize
    default_ld = (epr // lam) if transposed else linesize
    ld = 0 if ld_extra is None else default_ld + ld_extra
    eff_ld = default_ld if ld == 0 else ld
    eb = sew // 8

    def offset(i):
        line, e = divmod(i, linesize)
        return ((e * eff_ld) + line) if transposed else ((line * eff_ld) + e)

    mem_elems = max(offset(i) for i in range(vl)) + 1
    mem = bytearray(rng.getrandbits(8) for _ in range(mem_elems * eb))
    rf = R.VRegs(cfg.vlen)
    for i in range(len(rf.b)):
        rf.b[i] = rng.getrandbits(8)
    before_rf = bytes(rf.b)
    before_mem = bytes(mem)
    mask = [(rf.b[i // 8] >> (i % 8)) & 1 for i in range(vl)]
    for i in range(vl):
        if vm == 0 and not mask[i]:
            continue
        flat = R.tile_reg_idx(i, lmul, lam, epr)
        a = offset(i) * eb
        if store:
            mem[a : a + eb] = rf.read(sew, flat, VD).to_bytes(eb, "little")
        else:
            rf.write(sew, flat, VD, int.from_bytes(mem[a : a + eb], "little"))
    name = ("vmtts" if transposed else "vmts") if store else ("vmttl" if transposed else "vmtl")
    return {
        "title": f"{name} SEW={sew} LMUL={lmul} lambda={lam} (field {lam_field}) vm={vm} "
        f"VL={vl} LD={ld}",
        "store": store,
        "before": before_rf,
        "after": bytes(rf.b),
        "mem_before": before_mem,
        "mem_after": bytes(mem),
        "vlenb": vlenb,
        "vtype": G.vtype(sew, lmul, vt_lam, 0, 0, 0, 0),
        "vl": vl,
        "ld": ld,
        "inst": tile_encode(store, transposed, lam_field, vm, VD),
    }


def tile_illegal(cfg, title, sew, lmul, vt_lam, vl, inst):
    return {
        "title": "illegal: " + title,
        "illegal": True,
        "vlenb": cfg.vlen // 8,
        "vtype": G.vtype(sew, lmul, vt_lam, 0, 0, 0, 0),
        "vl": vl,
        "ld": 0,
        "inst": inst,
    }


def tile_suite(cfg, lams):
    rng = random.Random(11)
    cases = []
    for store in (False, True):
        for transposed in (False, True):
            for sew in SEWS:
                for lmul in LMULS:
                    for lam in lams[sew]:
                        vlmax = lmul * cfg.vlen // sew
                        if lam * lmul > vlmax:
                            continue
                        lines = vlmax // (lam * lmul)
                        vt_lam = rng.choice(lams[sew])
                        lam_field = 0 if vt_lam == lam else lam_enc(lam)
                        vm = rng.choice((0, 1))
                        ld_extra = rng.choice((None, 0, 1, 3))
                        n = rng.choice(sorted({1, lines, rng.randint(1, lines)}))
                        cases.append(
                            tile_case(
                                rng,
                                cfg,
                                store,
                                transposed,
                                sew,
                                lmul,
                                vt_lam,
                                lam_field,
                                vm,
                                n,
                                ld_extra,
                            )
                        )
    sew = 32
    lam = lams[sew][0]
    cases.append(
        tile_illegal(
            cfg, "vmtl.v masked load into v0", sew, 1, lam, lam, tile_encode(False, False, 0, 0, 0)
        )
    )
    cases.append(
        tile_illegal(
            cfg, "vmttl.v masked load into v0", sew, 1, lam, lam, tile_encode(False, True, 0, 0, 0)
        )
    )
    unsupported = [x for x in (1, 2, 4, 8, 16, 32, 64) if x not in lams[sew]]
    if unsupported:
        for store in (False, True):
            cases.append(
                tile_illegal(
                    cfg,
                    f"unsupported lambda field {unsupported[0]} at SEW={sew}",
                    sew,
                    1,
                    lam,
                    lam,
                    tile_encode(store, False, lam_enc(unsupported[0]), 1, VD),
                )
            )
    big = max(lams[sew])
    if big > 1 and big < cfg.vlen // sew:
        cases.append(
            tile_illegal(
                cfg,
                "VL not a multiple of lambda * LMUL",
                sew,
                1,
                big,
                big + 1,
                tile_encode(False, False, 0, 1, VD),
            )
        )
    cases.append(
        tile_illegal(
            cfg,
            "register group of LMUL=2 at an odd register",
            sew,
            2,
            lam,
            2 * lam,
            tile_encode(True, True, 0, 1, VD + 1),
        )
    )
    return cases


def emit_tile(path, cases):
    out = []
    o = out.append
    o("# Generated by gen_int_tile.py.")
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
            o(f"  li t1, {vlenb}")
            o("  vsetvli t2, t1, e8, m1, ta, ma")
            o(f"  la a0, regs{k}")
            for reg in list(range(VD, VD + 8)) + [0]:
                o(f"  vle8.v v{reg}, (a0)")
                o(f"  addi a0, a0, {vlenb}")
        o(f"  li t0, 0x{c['vtype']:x}")
        o(f"  li t1, {c['vl']}")
        o("  vsetvl t2, t1, t0")
        o("  bne t1, t2, fail")
        o(f"  la a0, mem{k}" if not c.get("illegal") else "  la a0, outbuf")
        o(f"  li a1, {c['ld']}")
        o("  li s1, 0")
        o(f"  .word 0x{c['inst']:08x}")
        if c.get("illegal"):
            o("  li t1, 1")
            o("  bne s1, t1, fail")
            continue
        o("  bnez s1, fail")
        o(f"  li TESTNUM, {k * 10**7}")
        if c["store"]:
            o(f"  la a0, mem{k}")
            o(f"  la a1, expmem{k}")
            o(f"  li a2, {len(c['mem_after'])}")
        else:
            o(f"  li t1, {vlenb}")
            o("  vsetvli t2, t1, e8, m1, ta, ma")
            o("  la a0, outbuf")
            for reg in range(VD, VD + 8):
                o(f"  vse8.v v{reg}, (a0)")
                o(f"  addi a0, a0, {vlenb}")
            o("  la a0, outbuf")
            o(f"  la a1, expregs{k}")
            o(f"  li a2, {8 * vlenb}")
        o("  jal ra, check_bytes")
    o("  TEST_PASSFAIL")
    o("")
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

    def blob(label, data):
        o(".align 4")
        o(f"{label}:")
        for i in range(0, len(data), 16):
            o("  .byte " + ", ".join(str(x) for x in data[i : i + 16]))

    for k, c in enumerate(cases, start=1):
        if c.get("illegal"):
            continue
        vlenb = c["vlenb"]
        blob(f"regs{k}", c["before"][VD * vlenb : (VD + 8) * vlenb] + c["before"][0:vlenb])
        blob(f"mem{k}", c["mem_before"])
        if c["store"]:
            blob(f"expmem{k}", c["mem_after"])
        else:
            blob(f"expregs{k}", c["after"][VD * vlenb : (VD + 8) * vlenb])
    o("RVTEST_DATA_END")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for name in ("rv64-ime", "rv64-ime-v256", "rv64-ime-v256-grp", "rv64-ime-v1024-grp"):
        cfg, lams = G.load_cfg(name)
        int_cases = int_suite(cfg, lams)
        G.emit(os.path.join(out_dir, f"zvvm_int_{name}.S"), int_cases)
        tile_cases = tile_suite(cfg, lams)
        emit_tile(os.path.join(out_dir, f"zvvm_tile_{name}.S"), tile_cases)
        print(f"{name}: {len(int_cases)} integer cases, {len(tile_cases)} tile cases")


if __name__ == "__main__":
    main()

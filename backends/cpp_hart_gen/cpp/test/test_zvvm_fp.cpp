// Copyright (c) Openchip and Software Technologies, S.L.
// SPDX-License-Identifier: BSD-3-Clause-Clear

// Tests for the Zvvm floating-point primitives in spec/std/isa/isa/ime_fp.idl,
// as generated into the rv64-ime hart.
//
// The expected values come from a reference written here with exact rational
// arithmetic (GMP mpq_class). It shares no code with the IDL. The reference is
// itself checked against Berkeley SoftFloat for FP16, FP32 and FP64. It covers
// the formats SoftFloat lacks: BF16, E5M2, E4M3, E2M1, E8M0, and round-to-odd.

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <gmpxx.h>

#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <udb/cfgs/rv64-ime/hart.hxx>
#include <udb/hart_factory.hxx>
#include <udb/iss_soc_model.hpp>

extern "C" {
#include <softfloat.h>
}

using namespace udb;

namespace {

using ImeHart = Rv64Ime_Hart<IssSocModel>;

constexpr uint32_t NX = 1, UF = 2, OF = 4, NV = 16;

// ─── Reference ────────────────────────────────────────────────────────────

struct FmtInfo {
  const char* name;
  ZvvmFpFmt::ValueType id;
  int ebits;
  int mbits;
  bool ieee;  // has IEEE 754 infinity and NaN encodings
};

const FmtInfo E2M1{"E2M1", ZvvmFpFmt::E2M1, 2, 1, false};
const FmtInfo E4M3{"E4M3", ZvvmFpFmt::E4M3, 4, 3, false};
const FmtInfo E5M2{"E5M2", ZvvmFpFmt::E5M2, 5, 2, true};
const FmtInfo FP16{"FP16", ZvvmFpFmt::FP16, 5, 10, true};
const FmtInfo BF16{"BF16", ZvvmFpFmt::BF16, 8, 7, true};
const FmtInfo FP32{"FP32", ZvvmFpFmt::FP32, 8, 23, true};
const FmtInfo FP64{"FP64", ZvvmFpFmt::FP64, 11, 52, true};

int width(const FmtInfo& f) { return 1 + f.ebits + f.mbits; }

enum class Kind { Fin, Inf, NaN };

struct Val {
  int sign = 0;
  Kind kind = Kind::Fin;
  mpq_class v;  // magnitude of a finite value
  bool snan = false;
};

mpq_class pow2(long e) {
  mpz_class p = 1;
  mpz_mul_2exp(p.get_mpz_t(), p.get_mpz_t(), static_cast<mp_bitcnt_t>(e >= 0 ? e : -e));
  return e >= 0 ? mpq_class(p) : mpq_class(mpz_class(1), p);
}

uint64_t pack(const FmtInfo& f, int s, uint64_t e, uint64_t m) {
  return (static_cast<uint64_t>(s) << (f.ebits + f.mbits)) | (e << f.mbits) | m;
}

uint64_t default_nan(const FmtInfo& f) {
  if (f.id == ZvvmFpFmt::E4M3) return 0x7f;
  return pack(f, 0, (1ull << f.ebits) - 1, 1ull << (f.mbits - 1));
}

Val decode(const FmtInfo& f, uint64_t bits) {
  const uint64_t m = bits & ((1ull << f.mbits) - 1);
  const uint64_t e = (bits >> f.mbits) & ((1ull << f.ebits) - 1);
  const long bias = (1l << (f.ebits - 1)) - 1;
  Val x;
  x.sign = static_cast<int>((bits >> (f.ebits + f.mbits)) & 1);
  if (f.id == ZvvmFpFmt::E4M3 && e == 15 && m == 7) {
    x.kind = Kind::NaN;
  } else if (f.ieee && e == (1ull << f.ebits) - 1) {
    x.kind = m == 0 ? Kind::Inf : Kind::NaN;
    x.snan = m != 0 && (m >> (f.mbits - 1)) == 0;
  } else if (e == 0) {
    x.v = mpq_class(mpz_class(static_cast<unsigned long>(m))) * pow2(1 - bias - f.mbits);
  } else {
    x.v = mpq_class(mpz_class(static_cast<unsigned long>(m | (1ull << f.mbits)))) *
          pow2(static_cast<long>(e) - bias - f.mbits);
  }
  return x;
}

// floor(log2(v)) for v > 0
long exponent_of(const mpq_class& v) {
  long e = static_cast<long>(mpz_sizeinbase(v.get_num_mpz_t(), 2)) -
           static_cast<long>(mpz_sizeinbase(v.get_den_mpz_t(), 2));
  while (pow2(e) > v) e--;
  while (pow2(e + 1) <= v) e++;
  return e;
}

// Round x > 0 to an integer; returns the integer and whether it was inexact.
std::pair<mpz_class, bool> round_int(int s, const mpq_class& x, const std::string& rm, bool rto) {
  mpz_class n;
  mpz_fdiv_q(n.get_mpz_t(), x.get_num_mpz_t(), x.get_den_mpz_t());
  const mpq_class rem = x - mpq_class(n);
  if (rem == 0) return {n, false};
  if (rto) return {n | 1, true};
  const mpq_class half(1, 2);
  bool up = false;
  if (rm == "RNE") up = rem > half || (rem == half && mpz_odd_p(n.get_mpz_t()));
  if (rm == "RDN") up = s == 1;
  if (rm == "RUP") up = s == 0;
  if (rm == "RMM") up = rem >= half;
  return {up ? mpz_class(n + 1) : n, true};
}

struct Res {
  uint64_t bits;
  uint32_t flags;
};

Res round_to(const FmtInfo& f, const Val& x, const std::string& rm, bool rto = false) {
  const long bias = (1l << (f.ebits - 1)) - 1;
  const uint64_t emax_field = (1ull << f.ebits) - 1;
  const long emin = 1 - bias;
  if (x.kind == Kind::NaN) return {default_nan(f), 0};
  if (x.kind == Kind::Inf) return {pack(f, x.sign, emax_field, 0), 0};
  if (x.v == 0) return {pack(f, x.sign, 0, 0), 0};

  const long E = exponent_of(x.v);
  const mpz_class carry_out = mpz_class(1) << (f.mbits + 1);
  auto [n1, inexact1] = round_int(x.sign, x.v / pow2(E - f.mbits), rm, rto);
  (void)inexact1;
  const bool tiny = (E + (n1 == carry_out ? 1 : 0)) < emin;
  long q = std::max(E, emin) - f.mbits;
  auto [n, inexact] = round_int(x.sign, x.v / pow2(q), rm, rto);
  if (n == carry_out) {
    n >>= 1;
    q++;
  }
  const bool normal = n >= (mpz_class(1) << f.mbits);
  const long biased = normal ? q + f.mbits + bias : 0;
  if (biased >= static_cast<long>(emax_field)) {
    const bool to_inf = !rto && (rm == "RNE" || rm == "RMM" || (rm == "RUP" && x.sign == 0) ||
                                 (rm == "RDN" && x.sign == 1));
    if (to_inf) return {pack(f, x.sign, emax_field, 0), OF | NX};
    return {pack(f, x.sign, emax_field - 1, (1ull << f.mbits) - 1), OF | NX};
  }
  const mpz_class frac = normal ? mpz_class(n - (mpz_class(1) << f.mbits)) : n;
  const uint32_t flags = inexact ? (NX | (tiny ? UF : 0)) : 0;
  return {pack(f, x.sign, static_cast<uint64_t>(biased), frac.get_ui()), flags};
}

// Exact sum of a sequence of values, with the special-value and zero-sign
// rules of IEEE 754 applied in order, starting from +start+ (by default +0,
// as a group sum does). Returns the value and the NV flag of the additions.
std::pair<Val, uint32_t> exact_sum(const std::vector<Val>& xs, const std::string& rm, const Val& start = Val{}) {
  Val acc = start;
  uint32_t flags = 0;
  for (const Val& x : xs) {
    if (acc.kind == Kind::NaN) continue;
    if (x.kind == Kind::NaN) {
      acc.kind = Kind::NaN;
    } else if (acc.kind == Kind::Inf) {
      if (x.kind == Kind::Inf && x.sign != acc.sign) {
        acc.kind = Kind::NaN;
        flags |= NV;
      }
    } else if (x.kind == Kind::Inf) {
      acc = x;
    } else {
      const mpq_class a = acc.sign ? mpq_class(-acc.v) : acc.v;
      const mpq_class b = x.sign ? mpq_class(-x.v) : x.v;
      const mpq_class s = a + b;
      if (s == 0) {
        const bool both_zero = acc.v == 0 && x.v == 0;
        acc.sign = (both_zero && acc.sign == x.sign) ? acc.sign : (rm == "RDN" ? 1 : 0);
        acc.v = 0;
      } else {
        acc.sign = s < 0 ? 1 : 0;
        acc.v = abs(s);
      }
    }
  }
  return {acc, flags};
}

std::pair<Val, uint32_t> exact_mul(const Val& a, const Val& b) {
  Val r;
  uint32_t flags = (a.snan || b.snan) ? NV : 0;
  r.sign = a.sign ^ b.sign;
  if (a.kind == Kind::NaN || b.kind == Kind::NaN) {
    r.kind = Kind::NaN;
  } else if (a.kind == Kind::Inf || b.kind == Kind::Inf) {
    const bool zero = (a.kind == Kind::Fin && a.v == 0) || (b.kind == Kind::Fin && b.v == 0);
    r.kind = zero ? Kind::NaN : Kind::Inf;
    if (zero) flags |= NV;
  } else {
    r.v = a.v * b.v;
  }
  return {r, flags};
}

Res ref_add(const FmtInfo& f, uint64_t a, uint64_t b, const std::string& rm) {
  const Val va = decode(f, a), vb = decode(f, b);
  auto [s, nv] = exact_sum({vb}, rm, va);
  Res r = round_to(f, s, rm);
  r.flags |= nv | ((va.snan || vb.snan) ? NV : 0);
  return r;
}

Res ref_mul(const FmtInfo& f, uint64_t a, uint64_t b, const std::string& rm, bool rto = false) {
  auto [p, nv] = exact_mul(decode(f, a), decode(f, b));
  Res r = round_to(f, p, rm, rto);
  r.flags |= nv;
  return r;
}

// ─── Hart access ──────────────────────────────────────────────────────────

const std::vector<std::string> RMS = {"RNE", "RTZ", "RDN", "RUP", "RMM"};

RoundingMode to_rm(const std::string& rm) {
  if (rm == "RNE") return RoundingMode{RoundingMode::RNE};
  if (rm == "RTZ") return RoundingMode{RoundingMode::RTZ};
  if (rm == "RDN") return RoundingMode{RoundingMode::RDN};
  if (rm == "RUP") return RoundingMode{RoundingMode::RUP};
  return RoundingMode{RoundingMode::RMM};
}

uint_fast8_t to_softfloat_rm(const std::string& rm) {
  if (rm == "RNE") return softfloat_round_near_even;
  if (rm == "RTZ") return softfloat_round_minMag;
  if (rm == "RDN") return softfloat_round_min;
  if (rm == "RUP") return softfloat_round_max;
  return softfloat_round_near_maxMag;
}

struct Fixture {
  IssSocModel soc{0x1000, 0};
  HartBase<IssSocModel>* base;
  ImeHart* hart;

  Fixture() {
    base = HartFactory::create("rv64-ime", 0, std::filesystem::path{UDB_RV64_IME_CFG}, soc);
    hart = dynamic_cast<ImeHart*>(base);
    REQUIRE(hart != nullptr);
    hart->reset(0);
  }
  ~Fixture() { delete base; }

  uint32_t fflags() {
    CsrBase* csr = hart->csr("fflags");
    return static_cast<uint32_t>(csr->sw_read(Bits<8>{hart->mxlen()}).to_defined().get());
  }
  void clear_fflags() {
    CsrBase* csr = hart->csr("fflags");
    REQUIRE(csr->sw_write(Bits<64>{0}, Bits<8>{hart->mxlen()}));
  }

  template <typename F>
  Res run(F&& f) {
    clear_fflags();
    const uint64_t bits = f().to_defined().get();
    return {bits, fflags()};
  }

  auto fpx(int sign, unsigned __int128 mant, int32_t exp) {
    return hart->zvvm_fpx_finite(Bits<1>{static_cast<unsigned>(sign)}, Bits<128>{mant},
                                 Bits<32>{static_cast<uint32_t>(exp)});
  }
};

bool is_nan(const FmtInfo& f, uint64_t bits) { return decode(f, bits).kind == Kind::NaN; }

// Operands: zeros, extremes, specials, near-boundary and random encodings.
std::vector<uint64_t> operands(const FmtInfo& f, std::mt19937_64& rng, size_t count) {
  const uint64_t emax = (1ull << f.ebits) - 1;
  const uint64_t mmax = (1ull << f.mbits) - 1;
  std::vector<uint64_t> v = {0, 1, mmax, 1ull << f.mbits, pack(f, 0, emax - 1, mmax), pack(f, 0, 1, 0),
                             pack(f, 0, (1ull << (f.ebits - 1)) - 1, 0), pack(f, 0, (1ull << (f.ebits - 1)), 1)};
  if (f.ieee) {
    v.push_back(pack(f, 0, emax, 0));
    v.push_back(default_nan(f));
    v.push_back(pack(f, 0, emax, 1));  // signaling NaN
  }
  const size_t n = v.size();
  for (size_t i = 0; i < n; i++) v.push_back(v[i] | (1ull << (width(f) - 1)));
  const uint64_t mask = width(f) == 64 ? ~0ull : ((1ull << width(f)) - 1);
  while (v.size() < count) {
    uint64_t r = rng() & mask;
    // bias half of the random values towards nearby exponents, so that
    // additions cancel and round
    if ((rng() & 1) != 0) {
      const uint64_t e = (1ull << (f.ebits - 1)) - 3 + (rng() % 6);
      r = pack(f, static_cast<int>(rng() & 1), e, rng() & mmax);
    }
    v.push_back(r);
  }
  return v;
}

}  // namespace

// ─── Reference against SoftFloat ──────────────────────────────────────────

TEST_CASE("Reference rounding agrees with SoftFloat for FP16, FP32 and FP64", "[zvvm][fp]") {
  std::mt19937_64 rng(1);
  for (const FmtInfo* f : {&FP16, &FP32, &FP64}) {
    const auto ops = operands(*f, rng, 160);
    for (const auto& rm : RMS) {
      for (size_t i = 0; i < ops.size(); i++) {
        for (size_t j = 0; j < ops.size(); j += (i < 24 ? 1 : 7)) {
          const uint64_t a = ops[i], b = ops[j];
          softfloat_roundingMode = to_softfloat_rm(rm);
          uint64_t sf_add = 0, sf_mul = 0;
          uint32_t sf_add_flags = 0, sf_mul_flags = 0;
          softfloat_exceptionFlags = 0;
          if (f == &FP16) sf_add = f16_add({static_cast<uint16_t>(a)}, {static_cast<uint16_t>(b)}).v;
          if (f == &FP32) sf_add = f32_add({static_cast<uint32_t>(a)}, {static_cast<uint32_t>(b)}).v;
          if (f == &FP64) sf_add = f64_add({a}, {b}).v;
          sf_add_flags = softfloat_exceptionFlags;
          softfloat_exceptionFlags = 0;
          if (f == &FP16) sf_mul = f16_mul({static_cast<uint16_t>(a)}, {static_cast<uint16_t>(b)}).v;
          if (f == &FP32) sf_mul = f32_mul({static_cast<uint32_t>(a)}, {static_cast<uint32_t>(b)}).v;
          if (f == &FP64) sf_mul = f64_mul({a}, {b}).v;
          sf_mul_flags = softfloat_exceptionFlags;

          CAPTURE(f->name, rm, fmt::format("{:#x}", a), fmt::format("{:#x}", b));
          const Res add = ref_add(*f, a, b, rm), mul = ref_mul(*f, a, b, rm);
          // SoftFloat's 8086-SSE specialization propagates NaN payloads.
          if (!(is_nan(*f, add.bits) && is_nan(*f, sf_add))) CHECK(add.bits == sf_add);
          CHECK(add.flags == sf_add_flags);
          if (!(is_nan(*f, mul.bits) && is_nan(*f, sf_mul))) CHECK(mul.bits == sf_mul);
          CHECK(mul.flags == sf_mul_flags);
        }
      }
    }
  }
}

// ─── Primitives against the reference ────────────────────────────────────

TEST_CASE("zvvm_fp_add and zvvm_fp_mul match the reference", "[zvvm][fp]") {
  Fixture fx;
  std::mt19937_64 rng(2);
  for (const FmtInfo* f : {&E5M2, &FP16, &BF16, &FP32, &FP64}) {
    const auto ops = operands(*f, rng, 120);
    for (const auto& rm : RMS) {
      for (size_t i = 0; i < ops.size(); i++) {
        for (size_t j = 0; j < ops.size(); j += (i < 22 ? 1 : 5)) {
          const uint64_t a = ops[i], b = ops[j];
          CAPTURE(f->name, rm, fmt::format("{:#x}", a), fmt::format("{:#x}", b));
          const ZvvmFpFmt zf{f->id};
          Res got = fx.run([&] { return fx.hart->zvvm_fp_add(Bits<64>{a}, Bits<64>{b}, zf, to_rm(rm)); });
          Res exp = ref_add(*f, a, b, rm);
          CHECK(got.bits == exp.bits);
          CHECK(got.flags == exp.flags);
          got = fx.run([&] { return fx.hart->zvvm_fp_mul(Bits<64>{a}, Bits<64>{b}, zf, to_rm(rm)); });
          exp = ref_mul(*f, a, b, rm);
          CHECK(got.bits == exp.bits);
          CHECK(got.flags == exp.flags);
        }
      }
    }
  }
}

TEST_CASE("zvvm_fp_round_to_rto rounds exact products to odd", "[zvvm][fp]") {
  Fixture fx;
  std::mt19937_64 rng(3);
  for (const FmtInfo* f : {&E5M2, &FP16, &BF16, &FP32, &FP64}) {
    const auto ops = operands(*f, rng, 100);
    for (size_t i = 0; i < ops.size(); i++) {
      for (size_t j = 0; j < ops.size(); j++) {
        const uint64_t a = ops[i], b = ops[j];
        CAPTURE(f->name, fmt::format("{:#x}", a), fmt::format("{:#x}", b));
        const ZvvmFpFmt zf{f->id};
        const Res got = fx.run([&] {
          return fx.hart->zvvm_fp_round_to_rto(
              fx.hart->zvvm_fp_mul_exact(Bits<64>{a}, zf, Bits<64>{b}, zf), zf);
        });
        const Res exp = ref_mul(*f, a, b, "RNE", true);
        CHECK(got.bits == exp.bits);
        CHECK(got.flags == exp.flags);
      }
    }
  }
}

TEST_CASE("OFP8 and OFP4 products round to the C format as the reference does", "[zvvm][fp]") {
  Fixture fx;
  struct Row {
    const FmtInfo* a;
    const FmtInfo* b;
    const FmtInfo* c;
  };
  for (const Row& row : {Row{&E4M3, &E4M3, &FP16}, Row{&E5M2, &E5M2, &FP16}, Row{&E4M3, &E5M2, &BF16},
                         Row{&E2M1, &E2M1, &FP16}, Row{&E5M2, &E4M3, &FP32}, Row{&E5M2, &E5M2, &E5M2}}) {
    const uint64_t na = 1ull << width(*row.a), nb = 1ull << width(*row.b);
    for (uint64_t a = 0; a < na; a++) {
      for (uint64_t b = 0; b < nb; b += (nb > 16 ? 5 : 1)) {
        for (const auto& rm : RMS) {
          CAPTURE(row.a->name, row.b->name, row.c->name, rm, a, b);
          const Res got = fx.run([&] {
            return fx.hart->zvvm_fp_round_to_frm(
                fx.hart->zvvm_fp_mul_exact(Bits<64>{a}, ZvvmFpFmt{row.a->id}, Bits<64>{b}, ZvvmFpFmt{row.b->id}),
                ZvvmFpFmt{row.c->id}, to_rm(rm));
          });
          auto [p, nv] = exact_mul(decode(*row.a, a), decode(*row.b, b));
          Res exp = round_to(*row.c, p, rm);
          exp.flags |= nv;
          CHECK(got.bits == exp.bits);
          CHECK(got.flags == exp.flags);
        }
      }
    }
  }
}

TEST_CASE("OFP decode: E5M2 signaling NaN, single E4M3 NaN, E2M1 values", "[zvvm][fp]") {
  Fixture fx;
  const ZvvmFpFmt e5m2{ZvvmFpFmt::E5M2}, e4m3{ZvvmFpFmt::E4M3}, e2m1{ZvvmFpFmt::E2M1};
  CHECK(fx.hart->zvvm_fp_signaling_nan_Q_(e5m2, Bits<64>{0x7d}));    // S.11111.01
  CHECK_FALSE(fx.hart->zvvm_fp_signaling_nan_Q_(e5m2, Bits<64>{0x7e}));  // S.11111.10
  CHECK_FALSE(fx.hart->zvvm_fp_signaling_nan_Q_(e5m2, Bits<64>{0x7c}));  // infinity
  for (uint64_t v = 0; v < 256; v++) {
    CAPTURE(v);
    CHECK_FALSE(fx.hart->zvvm_fp_signaling_nan_Q_(e4m3, Bits<64>{v}));
    CHECK(fx.hart->zvvm_fp_is_nan_Q_(Bits<64>{v}, e4m3) == ((v & 0x7f) == 0x7f));
  }
  // 0x7e is 448, the largest E4M3 value; 0x0f is 6, the largest E2M1 magnitude
  CHECK(fx.run([&] {
          return fx.hart->zvvm_fp_round_to_frm(
              fx.hart->zvvm_fp_mul_exact(Bits<64>{0x7e}, e4m3, Bits<64>{0x0f}, e2m1), ZvvmFpFmt{ZvvmFpFmt::FP16},
              to_rm("RNE"));
        }).bits == 0xe940);  // -2688 = -1.3125 x 2^11
  // an E5M2 signaling NaN operand raises NV
  const Res r = fx.run([&] {
    return fx.hart->zvvm_fp_round_to_frm(fx.hart->zvvm_fp_mul_exact(Bits<64>{0x7d}, e5m2, Bits<64>{0x3c}, e5m2),
                                         ZvvmFpFmt{ZvvmFpFmt::FP16}, to_rm("RNE"));
  });
  CHECK(r.bits == 0x7e00);
  CHECK(r.flags == NV);
}

TEST_CASE("zvvm_decode_scale converts every E8M0 value", "[zvvm][fp]") {
  Fixture fx;
  for (const FmtInfo* f : {&E5M2, &FP16, &BF16, &FP32, &FP64}) {
    for (uint64_t e = 0; e < 256; e++) {
      for (const auto& rm : RMS) {
        CAPTURE(f->name, e, rm);
        const Res got = fx.run([&] {
          return fx.hart->zvvm_decode_scale(Bits<8>{static_cast<uint8_t>(e)}, ZvvmFpFmt{f->id}, to_rm(rm));
        });
        Res exp{default_nan(*f), 0};
        if (e != 0xff) {
          Val v;
          v.v = pow2(static_cast<long>(e) - 127);
          exp = round_to(*f, v, rm);
        }
        CHECK(got.bits == exp.bits);
        CHECK(got.flags == exp.flags);
      }
    }
  }
}

TEST_CASE("zvvm_int_to_fp converts block dot products", "[zvvm][fp]") {
  Fixture fx;
  for (const FmtInfo* f : {&FP16, &BF16, &FP32}) {
    for (int64_t x : {0l, 1l, -1l, 255l, 256l, 257l, 2049l, -2049l, 4095l, 65519l, 65520l, -65520l,
                      1l << 19, -(1l << 19) + 3}) {
      for (const auto& rm : RMS) {
        CAPTURE(f->name, x, rm);
        const Res got = fx.run([&] {
          return fx.hart->zvvm_int_to_fp(Bits<64>{static_cast<uint64_t>(x)}, ZvvmFpFmt{f->id}, to_rm(rm));
        });
        Val v;
        v.sign = x < 0 ? 1 : 0;
        v.v = mpq_class(mpz_class(static_cast<long>(x < 0 ? -x : x)));
        const Res exp = round_to(*f, v, rm);
        CHECK(got.bits == exp.bits);
        CHECK(got.flags == exp.flags);
      }
    }
  }
}

// ─── Group sums ───────────────────────────────────────────────────────────

TEST_CASE("Exact group sums round as the reference does", "[zvvm][fp]") {
  Fixture fx;
  std::mt19937_64 rng(4);
  struct Row {
    const FmtInfo* in;
    const FmtInfo* c;
    bool long_sum;
  };
  for (const Row& row : {Row{&FP16, &FP32, false}, Row{&E5M2, &FP16, false}, Row{&E4M3, &BF16, false},
                         Row{&BF16, &FP32, true}, Row{&FP32, &FP64, true}, Row{&FP64, &FP64, true}}) {
    const auto ops = operands(*row.in, rng, 200);
    const ZvvmFpFmt fin{row.in->id}, fc{row.c->id};
    for (int trial = 0; trial < 300; trial++) {
      const size_t n = 1 + rng() % 8;
      std::vector<uint64_t> a(n), b(n);
      for (size_t k = 0; k < n; k++) {
        a[k] = ops[rng() % ops.size()];
        b[k] = ops[rng() % ops.size()];
      }
      // make some groups cancel exactly: append the negated first product
      if (trial % 3 == 0) {
        a.push_back(a[0] ^ (1ull << (width(*row.in) - 1)));
        b.push_back(b[0]);
      }
      const uint64_t c = operands(*row.c, rng, 40)[rng() % 40];
      for (const auto& rm : RMS) {
        CAPTURE(row.in->name, row.c->name, rm, trial, fmt::format("{:#x}", c));
        std::vector<Val> prods;
        uint32_t prod_flags = 0;
        for (size_t k = 0; k < a.size(); k++) {
          auto [p, nv] = exact_mul(decode(*row.in, a[k]), decode(*row.in, b[k]));
          prods.push_back(p);
          prod_flags |= nv;
        }
        auto [s, sum_flags] = exact_sum(prods, rm);

        // rnd = frm: round S, then add it to C
        Res exp_s = round_to(*row.c, s, rm);
        Res exp = ref_add(*row.c, c, exp_s.bits, rm);
        exp.flags |= exp_s.flags | prod_flags | sum_flags;
        Res got = fx.run([&] {
          Bits<64> s_bits;
          if (row.long_sum) {
            auto l = fx.hart->zvvm_fpx_long_zero();
            for (size_t k = 0; k < a.size(); k++)
              l = fx.hart->zvvm_fpx_long_add(l, fx.hart->zvvm_fp_mul_exact(Bits<64>{a[k]}, fin, Bits<64>{b[k]}, fin), to_rm(rm));
            s_bits = fx.hart->zvvm_fp_round_to_frm(fx.hart->zvvm_fpx_long_extract(l, fc), fc, to_rm(rm));
          } else {
            auto x = fx.hart->zvvm_fp_internal_zero();
            for (size_t k = 0; k < a.size(); k++)
              x = fx.hart->zvvm_fp_internal_add(x, fx.hart->zvvm_fp_mul_exact(Bits<64>{a[k]}, fin, Bits<64>{b[k]}, fin), to_rm(rm));
            s_bits = fx.hart->zvvm_fp_round_to_frm(x, fc, to_rm(rm));
          }
          return fx.hart->zvvm_fp_add(Bits<64>{c}, s_bits, fc, to_rm(rm));
        });
        CHECK(got.bits == exp.bits);
        CHECK(got.flags == exp.flags);

        // rnd = xct: round C + S once
        const Val vc = decode(*row.c, c);
        auto [s_only, nv_s] = exact_sum(prods, rm);
        auto [cs, nv_cs] = exact_sum({vc}, rm, s_only);
        exp = round_to(*row.c, cs, rm);
        exp.flags |= prod_flags | nv_s | nv_cs | (vc.snan ? NV : 0);
        got = fx.run([&] {
          if (row.long_sum) {
            auto l = fx.hart->zvvm_fpx_long_zero();
            for (size_t k = 0; k < a.size(); k++)
              l = fx.hart->zvvm_fpx_long_add(l, fx.hart->zvvm_fp_mul_exact(Bits<64>{a[k]}, fin, Bits<64>{b[k]}, fin), to_rm(rm));
            return fx.hart->zvvm_fp_add_internal_long(Bits<64>{c}, l, fc, to_rm(rm));
          }
          auto x = fx.hart->zvvm_fp_internal_zero();
          for (size_t k = 0; k < a.size(); k++)
            x = fx.hart->zvvm_fp_internal_add(x, fx.hart->zvvm_fp_mul_exact(Bits<64>{a[k]}, fin, Bits<64>{b[k]}, fin), to_rm(rm));
          return fx.hart->zvvm_fp_add_internal(Bits<64>{c}, x, fc, to_rm(rm));
        });
        CHECK(got.bits == exp.bits);
        CHECK(got.flags == exp.flags);
      }
    }
  }
}

TEST_CASE("Long accumulator keeps 2^-2148 next to cancelling 2^2047 terms", "[zvvm][fp]") {
  // Products +2^2046, -2^2046 and 2^-2148 (= 2^-1074 x 2^-1074), as in a group
  // of three FP64 x FP64 products with G >= 2 or rnd=xct. Rounding the exact
  // sum depends on frm, so dropping the small product is detectable.
  Fixture fx;
  const ZvvmFpFmt f64{ZvvmFpFmt::FP64};
  const uint64_t two_1023 = 0x7fe0000000000000ull;
  const uint64_t min_sub = 1, one = 0x3ff0000000000000ull;
  struct Case {
    std::string rm;
    uint64_t c;
    uint64_t s_frm;   // round_frm(S), then C + that
    uint64_t xct;     // round_frm(C + S)
    uint32_t xct_flags;
  };
  for (const Case& t : std::vector<Case>{
           {"RNE", 0, 0, 0, NX | UF},
           {"RTZ", 0, 0, 0, NX | UF},
           {"RDN", 0, 0, 0, NX | UF},
           {"RUP", 0, 1, 1, NX | UF},
           {"RMM", 0, 0, 0, NX | UF},
           {"RNE", one, one, one, NX},
           {"RUP", one, one + 1, one + 1, NX},
           {"RDN", one, one, one, NX},
       }) {
    CAPTURE(t.rm, t.c);
    auto group = [&] {
      auto l = fx.hart->zvvm_fpx_long_zero();
      l = fx.hart->zvvm_fpx_long_add(l, fx.hart->zvvm_fp_mul_exact(Bits<64>{two_1023}, f64, Bits<64>{two_1023}, f64), to_rm(t.rm));
      l = fx.hart->zvvm_fpx_long_add(l, fx.hart->zvvm_fp_mul_exact(Bits<64>{two_1023 | (1ull << 63)}, f64, Bits<64>{two_1023}, f64), to_rm(t.rm));
      l = fx.hart->zvvm_fpx_long_add(l, fx.hart->zvvm_fp_mul_exact(Bits<64>{min_sub}, f64, Bits<64>{min_sub}, f64), to_rm(t.rm));
      return l;
    };
    // rnd=frm: S rounds to +0 or the minimum subnormal, then C + S
    const Res s = fx.run([&] { return fx.hart->zvvm_fp_round_to_frm(fx.hart->zvvm_fpx_long_extract(group(), f64), f64, to_rm(t.rm)); });
    CHECK(s.bits == (t.rm == "RUP" ? min_sub : 0));
    CHECK(s.flags == (NX | UF));
    const Res frm = fx.run([&] { return fx.hart->zvvm_fp_add(Bits<64>{t.c}, Bits<64>{s.bits}, f64, to_rm(t.rm)); });
    CHECK(frm.bits == t.s_frm);
    // rnd=xct: C + S rounded once
    const Res xct = fx.run([&] { return fx.hart->zvvm_fp_add_internal_long(Bits<64>{t.c}, group(), f64, to_rm(t.rm)); });
    CHECK(xct.bits == t.xct);
    CHECK(xct.flags == t.xct_flags);
  }
}

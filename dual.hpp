// First-order forward-mode dual numbers, a + b*eps with eps^2 = 0.
//
// Used for one thing in this project: evaluating the dynamics along q + eps*qdot gives the exact
// time derivative of anything that depends only on q. The momentum observer needs C(q,qd)^T qd,
// and with Mdot = C + C^T it equals Mdot*qd - C*qd. Mdot*qd is the dual part of
// RNEA(q + eps*qd, 0, qd) with gravity off, which is O(n) and exact, versus O(n^2) for building
// Mdot by finite differences of the mass matrix (and those are not exact).
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <ostream>

namespace g1rtc {

struct Dual {
  double a = 0.0;  // value
  double b = 0.0;  // derivative
  constexpr Dual() = default;
  constexpr Dual(double v) : a(v) {}  // NOLINT(google-explicit-constructor): mixes with doubles
  constexpr Dual(double v, double d) : a(v), b(d) {}

  constexpr Dual& operator+=(const Dual& o) { a += o.a; b += o.b; return *this; }
  constexpr Dual& operator-=(const Dual& o) { a -= o.a; b -= o.b; return *this; }
  constexpr Dual& operator*=(const Dual& o) { b = b * o.a + a * o.b; a *= o.a; return *this; }
  constexpr Dual& operator/=(const Dual& o) {
    b = (b * o.a - a * o.b) / (o.a * o.a);
    a /= o.a;
    return *this;
  }
};

constexpr Dual operator+(Dual x, const Dual& y) { return x += y; }
constexpr Dual operator-(Dual x, const Dual& y) { return x -= y; }
constexpr Dual operator*(Dual x, const Dual& y) { return x *= y; }
constexpr Dual operator/(Dual x, const Dual& y) { return x /= y; }
constexpr Dual operator-(const Dual& x) { return {-x.a, -x.b}; }
constexpr Dual operator+(const Dual& x) { return x; }
constexpr bool operator<(const Dual& x, const Dual& y) { return x.a < y.a; }
constexpr bool operator>(const Dual& x, const Dual& y) { return x.a > y.a; }
constexpr bool operator<=(const Dual& x, const Dual& y) { return x.a <= y.a; }
constexpr bool operator>=(const Dual& x, const Dual& y) { return x.a >= y.a; }
constexpr bool operator==(const Dual& x, const Dual& y) { return x.a == y.a; }
constexpr bool operator!=(const Dual& x, const Dual& y) { return x.a != y.a; }

inline Dual sin(const Dual& x) { return {std::sin(x.a), x.b * std::cos(x.a)}; }
inline Dual cos(const Dual& x) { return {std::cos(x.a), -x.b * std::sin(x.a)}; }
inline Dual sqrt(const Dual& x) {
  const double s = std::sqrt(x.a);
  return {s, s > 0.0 ? x.b / (2.0 * s) : 0.0};
}
inline Dual abs(const Dual& x) { return x.a < 0.0 ? -x : x; }
inline std::ostream& operator<<(std::ostream& os, const Dual& x) {
  return os << x.a << "+" << x.b << "e";
}

inline double value(double x) { return x; }
inline double value(const Dual& x) { return x.a; }
inline double deriv(const Dual& x) { return x.b; }

}  // namespace g1rtc

namespace Eigen {
template <>
struct NumTraits<g1rtc::Dual> : NumTraits<double> {
  using Real = g1rtc::Dual;
  using NonInteger = g1rtc::Dual;
  using Nested = g1rtc::Dual;
  using Literal = g1rtc::Dual;
  enum { IsComplex = 0, IsInteger = 0, IsSigned = 1, RequireInitialization = 1, ReadCost = 2,
         AddCost = 2, MulCost = 4 };
};
template <typename BinaryOp>
struct ScalarBinaryOpTraits<g1rtc::Dual, double, BinaryOp> { using ReturnType = g1rtc::Dual; };
template <typename BinaryOp>
struct ScalarBinaryOpTraits<double, g1rtc::Dual, BinaryOp> { using ReturnType = g1rtc::Dual; };
}  // namespace Eigen

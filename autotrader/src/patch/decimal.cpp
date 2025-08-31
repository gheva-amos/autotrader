// Use the IB header that declares `typedef unsigned long long Decimal;`
// and the class `DecimalFunctions` with TWSAPIDLLEXP.
#include "Decimal.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace {
    // We store values as integer micro-units: value = raw / 1'000'000
    // This is a pragmatic stand-in for IB's BID decimal; good enough for a connect test
    // and typical prices/qty. Adjust if you need more precision.
    constexpr unsigned long long SCALE = 1000000ULL;

    // Helpers — promote to wider type for intermediate math when possible
    template <typename T>
    inline T sat_cast_ull(long double x) {
        long double maxv = static_cast<long double>(std::numeric_limits<unsigned long long>::max());
        if (x < 0.0L) return 0ULL;
        if (x > maxv) return std::numeric_limits<unsigned long long>::max();
        // round half away from zero
        return static_cast<unsigned long long>(std::llround(x));
    }

    inline unsigned long long add_raw(unsigned long long a, unsigned long long b) {
#if defined(__SIZEOF_INT128__)
        __uint128_t s = static_cast<__uint128_t>(a) + static_cast<__uint128_t>(b);
        if (s > std::numeric_limits<unsigned long long>::max())
            return std::numeric_limits<unsigned long long>::max();
        return static_cast<unsigned long long>(s);
#else
        unsigned long long s = a + b;
        if (s < a) return std::numeric_limits<unsigned long long>::max(); // overflow
        return s;
#endif
    }

    inline unsigned long long sub_raw(unsigned long long a, unsigned long long b) {
        return (a >= b) ? (a - b) : 0ULL; // floor at 0 for unsigned
    }

    inline unsigned long long mul_raw(unsigned long long a, unsigned long long b) {
#if defined(__SIZEOF_INT128__)
        // (a * b) / SCALE with 128-bit intermediate for precision
        __uint128_t prod = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
        prod = (prod + (SCALE / 2)) / SCALE; // rounded
        if (prod > std::numeric_limits<unsigned long long>::max())
            return std::numeric_limits<unsigned long long>::max();
        return static_cast<unsigned long long>(prod);
#else
        // Fallback via long double
        long double v = (static_cast<long double>(a) * static_cast<long double>(b)) / static_cast<long double>(SCALE);
        return sat_cast_ull<unsigned long long>(v);
#endif
    }

    inline unsigned long long div_raw(unsigned long long a, unsigned long long b) {
        if (b == 0ULL) return std::numeric_limits<unsigned long long>::max();
#if defined(__SIZEOF_INT128__)
        __uint128_t num = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(SCALE);
        num += b / 2; // rounded
        __uint128_t q = num / b;
        if (q > std::numeric_limits<unsigned long long>::max())
            return std::numeric_limits<unsigned long long>::max();
        return static_cast<unsigned long long>(q);
#else
        long double v = (static_cast<long double>(a) * static_cast<long double>(SCALE)) / static_cast<long double>(b);
        return sat_cast_ull<unsigned long long>(v);
#endif
    }

    inline double raw_to_double(unsigned long long raw) {
        return static_cast<double>(raw) / static_cast<double>(SCALE);
    }

    inline unsigned long long double_to_raw(double d) {
        long double scaled = static_cast<long double>(d) * static_cast<long double>(SCALE);
        return sat_cast_ull<unsigned long long>(scaled);
    }

    inline unsigned long long parse_double_like(const std::string& s) {
        // strtold to tolerate "123.45", "1e-3", etc.
        const char* p = s.c_str();
        char* end = nullptr;
        long double v = std::strtold(p, &end);
        // You can add stricter validation if you want (end == p -> no parse)
        return sat_cast_ull<unsigned long long>(v * static_cast<long double>(SCALE));
    }

    inline std::string format_decimal(unsigned long long raw, int precision = 6) {
        // prints with up to 6 fractional digits (micro-units)
        // avoid iostream overhead in minimal environments
        long double v = static_cast<long double>(raw) / static_cast<long double>(SCALE);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%.*Lf", precision, v);
        // optional: trim trailing zeros
        std::string out(buf);
        auto pos = out.find('.');
        if (pos != std::string::npos) {
            // remove trailing zeros
            size_t end = out.size();
            while (end > pos + 1 && out[end - 1] == '0') --end;
            if (end > pos + 1 && out[end - 1] == '.') --end; // remove dot if now integer
            out.resize(end);
        }
        return out;
    }
} // anonymous

// ---- DecimalFunctions ----

Decimal DecimalFunctions::add(Decimal decimal1, Decimal decimal2) {
    return add_raw(decimal1, decimal2);
}

Decimal DecimalFunctions::sub(Decimal decimal1, Decimal decimal2) {
    return sub_raw(decimal1, decimal2);
}

Decimal DecimalFunctions::mul(Decimal decimal1, Decimal decimal2) {
    return mul_raw(decimal1, decimal2);
}

Decimal DecimalFunctions::div(Decimal decimal1, Decimal decimal2) {
    return div_raw(decimal1, decimal2);
}

double DecimalFunctions::decimalToDouble(Decimal decimal) {
    return raw_to_double(decimal);
}

Decimal DecimalFunctions::doubleToDecimal(double d) {
    return double_to_raw(d);
}

Decimal DecimalFunctions::stringToDecimal(std::string str) {
    return parse_double_like(str);
}

std::string DecimalFunctions::decimalToString(Decimal value) {
    return format_decimal(value, 6);
}

std::string DecimalFunctions::decimalStringToDisplay(Decimal value) {
    // Often identical to decimalToString; adapt if you want locale/pretty formatting
    return format_decimal(value, 6);
}

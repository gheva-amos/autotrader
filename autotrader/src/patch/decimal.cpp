#include "Decimal.h"
#include <string>
#include <cctype>
#include <limits>

// saturating helpers (optional)
static inline unsigned long long sat_add(unsigned long long a, unsigned long long b) {
    unsigned long long s = a + b;
    if (s < a) return std::numeric_limits<unsigned long long>::max();
    return s;
}
static inline unsigned long long sat_mul(unsigned long long a, unsigned long long b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t p = (__uint128_t)a * (__uint128_t)b;
    if (p > std::numeric_limits<unsigned long long>::max())
        return std::numeric_limits<unsigned long long>::max();
    return (unsigned long long)p;
#else
    if (b && a > std::numeric_limits<unsigned long long>::max() / b)
        return std::numeric_limits<unsigned long long>::max();
    return a * b;
#endif
}

Decimal DecimalFunctions::add(Decimal a, Decimal b) { return sat_add(a, b); }
Decimal DecimalFunctions::sub(Decimal a, Decimal b) { return (a >= b) ? (a - b) : 0ULL; }
Decimal DecimalFunctions::mul(Decimal a, Decimal b) { return sat_mul(a, b); }
Decimal DecimalFunctions::div(Decimal a, Decimal b) { return b ? (a / b) : std::numeric_limits<unsigned long long>::max(); }

double  DecimalFunctions::decimalToDouble(Decimal v) { return static_cast<double>(v); }

// Round doubles to nearest whole unit (sizes are integers for stocks)
Decimal DecimalFunctions::doubleToDecimal(double d) {
    if (d <= 0) return 0ULL;
    long double ld = d;
    return static_cast<unsigned long long>(ld + 0.5L);
}

// Parse only the integer part; ignore any fractional tail
static inline Decimal parse_uint(const std::string& s) {
    unsigned long long acc = 0;
    for (char ch : s) {
        if (!std::isdigit((unsigned char)ch)) break;
        unsigned digit = unsigned(ch - '0');
        if (acc > (std::numeric_limits<unsigned long long>::max() - digit) / 10ULL)
            return std::numeric_limits<unsigned long long>::max();
        acc = acc * 10ULL + digit;
    }
    return acc;
}
Decimal DecimalFunctions::stringToDecimal(std::string str) {
    auto dot = str.find('.');
    if (dot != std::string::npos) str.resize(dot);  // drop fractional part
    size_t i = 0; while (i < str.size() && (str[i] == ' ' || str[i] == '+')) ++i;
    return parse_uint(str.substr(i));
}

std::string DecimalFunctions::decimalToString(Decimal v)           { return std::to_string(v); }
std::string DecimalFunctions::decimalStringToDisplay(Decimal v)    { return std::to_string(v); }

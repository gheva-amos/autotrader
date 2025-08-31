#ifndef CMULATE_DEBUG_H__
#define CMULATE_DEBUG_H__

#include <ostream>

namespace autotrader
{

std::ostream& get_log_stream(bool enabled);
#ifdef DEBUG
#define DBG_MSG(message) get_log_stream(true) << __FILE__ << ':' << __LINE__ << ' ' << (message) << ' '
#else
#define DBG_MSG(message) get_log_stream(false)
#endif
} // namespace

#endif // CMULATE_DEBUG_H__

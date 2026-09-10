#include "ingress_timing_buffer.h"
#include <cassert>
#include <cmath>
#include <limits>

int main()
{
  using struct_mapping::IngressTimingBuffer;
  int a, b, c;
  IngressTimingBuffer times(2);
  times.record(&a, 10.0);
  times.record(&b, 10.1);
  // TF can release messages out of receipt order.
  assert(std::abs(times.takeMs(&b, 10.15) - 50.0) < 1e-6);
  assert(std::abs(times.takeMs(&a, 10.2) - 200.0) < 1e-6);
  assert(times.takeMs(&a, 10.3) == -1.0);
  times.record(&a, 20.0);
  times.record(&b, 20.1);
  times.record(&c, 20.2);
  assert(times.takeMs(&a, 20.3) == -1.0);  // Bounded even without failure callbacks.
  times.record(&b, 30.0);  // Reused address must not retain an old receipt.
  assert(std::abs(times.takeMs(&b, 30.1) - 100.0) < 1e-6);
  assert(std::abs(times.takeMs(&c, 30.1) - 9900.0) < 1e-6);
  times.record(&a, std::numeric_limits<double>::quiet_NaN());
  assert(times.takeMs(&a, 40.) == -1.);
  times.record(nullptr, 40.);
  assert(times.takeMs(nullptr, 40.) == -1.);
  times.record(&a, 40.);
  assert(times.takeMs(&a, 39.) == -1.);
  times.record(&a, 40.);
  assert(times.takeMs(&a, std::numeric_limits<double>::infinity()) == -1.);
  assert(times.takeMs(&a, 41.) == -1.);  // Invalid-clock sample was consumed.
  IngressTimingBuffer disabled(0);
  disabled.record(&a, 10.);
  assert(disabled.takeMs(&a, 10.) == -1.);
}

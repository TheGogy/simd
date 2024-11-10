# simd

This is an optimized, single header file wrapper for __m128 for modern C++.

Example usage:

```cpp
#include <iostream>
#include "simd.h"

int main()
{
    Simd4 x(1.0, 2.0, 3.0, 4.0);
    Simd4 y(4.0, 3.0, 2.0, 1.0);

    std::cout << (x + y) << std::endl;
}
```
```bash
g++ -msse4.1 -mfma -std=c++20 main.cpp -o main
./main
# [5, 5, 5, 5]
```

Required flags are `-msse4.1 -mfma -std=c++20`.

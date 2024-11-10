# simd

This is an optimized, single header file wrapper for __m128 for modern C++.
It includes an implementation of Matrix4x4, a class for a 4x4 matrix.

Example usage:

```cpp
#include <iostream>

#include "simd.h"
#include "matrix.h"

int main()
{
    Simd4 x(1.0, 2.0, 3.0, 4.0);
    Simd4 y(4.0, 3.0, 2.0, 1.0);

    Matrix4x4 q(
        1, 3, 5, 9,
        1, 3, 1, 7,
        4, 3, 9, 7,
        5, 2, 0, 9
    );

    std::cout << (x + y) << std::endl;
    std::cout << q * x << std::endl;
}
```
```bash
g++ -msse4.1 -mfma -std=c++20 main.cpp -o main
./main
# [5, 5, 5, 5]
# [58, 38, 65, 45]
```

Required flags are `-msse4.1 -mfma -std=c++20`.

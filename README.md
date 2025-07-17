## RAIRS: Optimizing Redundant Assignment and List Layout for IVF-Based ANN Search

---

### Requirements
1. AVX2 instruction set
2. GCC or Clang supporting C++17 
3. Faiss v1.8.0 or higher
4. CMake v3.0.0 or higher

### Build
``` shell
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
./rairs_test
```

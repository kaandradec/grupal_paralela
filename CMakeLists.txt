cmake_minimum_required(VERSION 3.16)
project(Grupal)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -fopenmp")

# Toolchain de vcpkg
set(CMAKE_TOOLCHAIN_FILE "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
set(VCPKG_TARGET_TRIPLET "x64-mingw-dynamic")

# Ejecutable
add_executable(grupal
    main.cpp
    blelloch.cpp
    histograma.cpp
)

# fmt
find_package(fmt CONFIG REQUIRED)
target_link_libraries(grupal PRIVATE fmt::fmt)



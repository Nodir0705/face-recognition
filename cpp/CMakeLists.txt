cmake_minimum_required(VERSION 3.16)
project(attendance_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type defaults to Release — this is performance code, never ship debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Strict warnings — catch bugs before they catch us at 03:00 in production
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# Native arch tuning. On a Pi 4 -mcpu=cortex-a72 buys ~5-10% on the inner loop.
# Off by default to keep cross-builds simple — opt in with -DNATIVE=ON.
option(NATIVE "Tune for the host CPU (-march=native or -mcpu=native)" OFF)
if(NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    else()
        add_compile_options(-mcpu=native)
    endif()
endif()

# ---------- Dependencies ----------

# OpenCV — needs the C++ dev headers (libopencv-dev on Debian/Ubuntu/Bookworm)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

# ONNXRuntime — usually NOT shipped via apt. Install steps in cpp/README.md.
# We support two layouts:
#   1. ONNXRUNTIME_ROOT pointing at an extracted release tarball
#      (e.g. /opt/onnxruntime-linux-x64-1.18.1)
#   2. System install with onnxruntime/ headers under /usr/local/include
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNXRuntime root (with include/ and lib/)")
if(ONNXRUNTIME_ROOT)
    set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    find_library(ORT_LIBRARY onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
else()
    find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime onnxruntime/core/session)
    find_library(ORT_LIBRARY onnxruntime REQUIRED)
endif()
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR
        "ONNXRuntime not found. Pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime "
        "or install it system-wide. See cpp/README.md.")
endif()
message(STATUS "ONNXRuntime headers: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNXRuntime library: ${ORT_LIBRARY}")

# SQLite3 — the daemon writes to the same attendance.db the Python app uses
find_package(SQLite3 REQUIRED)

# ---------- Targets ----------

add_executable(bench_cpp bench.cpp)
target_include_directories(bench_cpp PRIVATE ${ORT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bench_cpp PRIVATE ${ORT_LIBRARY} ${OpenCV_LIBS})

add_executable(recognize_cpp recognize_cpp.cpp)
target_include_directories(recognize_cpp PRIVATE
    ${ORT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${SQLite3_INCLUDE_DIRS})
target_link_libraries(recognize_cpp PRIVATE
    ${ORT_LIBRARY} ${OpenCV_LIBS} SQLite::SQLite3 pthread)

# Help the runtime loader find libonnxruntime.so when we don't install it.
set_target_properties(bench_cpp recognize_cpp PROPERTIES
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE)
if(ONNXRUNTIME_ROOT)
    set_target_properties(bench_cpp recognize_cpp PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
endif()

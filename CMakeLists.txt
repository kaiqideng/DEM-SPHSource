# ===== CMakeLists.txt (root) ================================================
cmake_minimum_required(VERSION 3.18)  # >=3.17 for FindCUDAToolkit; 3.18 is safer
project(DEM-SPH LANGUAGES CXX CUDA)

# ---------- Language standards & output ----------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------- Options ----------
option(USE_FAST_MATH "Enable --use_fast_math for CUDA" ON)
# Allow override from CLI: -DCUDA_ARCHS="70;75;80;86;89"
set(CUDA_ARCHS "" CACHE STRING "CUDA architectures list (e.g. 70;75;80;86;89)")

# ---------- CUDA Toolkit ----------
# Require CUDA >= 11.8 for good VS2022/Linux compatibility
find_package(CUDAToolkit 11.8 REQUIRED)

# Default SM list if none provided
if (NOT CUDA_ARCHS)
    # Reasonable defaults for CUDA 11.8 era GPUs: Volta/Turing/Ampere/Ada
    set(CUDA_ARCHS "70;75;80;86;89")
endif()

# ---------- Include paths ----------
set(PROJ_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/include   # keep even if the dir doesn't exist
)

# ---------- Sources ----------
file(GLOB_RECURSE CORE_CPP CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/core/*.cpp)
file(GLOB_RECURSE CORE_CU  CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/core/*.cu)
file(GLOB_RECURSE DRIVERS  CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/driver/*.cpp)

# ---------- Core object library (no mains) ----------
add_library(core_obj OBJECT ${CORE_CPP} ${CORE_CU})

set_target_properties(core_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CUDA_ARCHS}"
)

target_compile_options(core_obj
    PRIVATE
        # CUDA TU: relaxed constexpr + line info
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
        # Optional fast math
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${USE_FAST_MATH}>>:--use_fast_math>
        # MSVC C++ flags
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/MP /bigobj /permissive- /Zc:__cplusplus /EHsc>
        # GCC/Clang C++ flags
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wno-unknown-pragmas>
)

target_include_directories(core_obj
    PUBLIC ${PROJ_INCLUDE_DIRS}
)

target_link_libraries(core_obj
    PUBLIC CUDA::cudart
    # PUBLIC CUDA::cublas CUDA::cusparse CUDA::nvToolsExt
)

# ---------- One executable per driver/**/*.cpp that has main() ----------
if (DRIVERS)
    message(STATUS "DRIVERS found: ${DRIVERS}") 
    foreach(MAIN_CPP ${DRIVERS})
        file(READ "${MAIN_CPP}" _src_txt LIMIT 5000) 
        if(NOT _src_txt MATCHES "int[ \t\r\n]+main\\s*\\(")
            continue()
        endif()

        get_filename_component(APP ${MAIN_CPP} NAME_WE)

        add_executable(${APP}
            ${MAIN_CPP}
            $<TARGET_OBJECTS:core_obj>
        )

        set_target_properties(${APP} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CUDA_ARCHS}"
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

        target_include_directories(${APP} PRIVATE ${PROJ_INCLUDE_DIRS})
        target_link_libraries(${APP} PRIVATE CUDA::cudart)

        target_compile_options(${APP}
            PRIVATE
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/MP /bigobj /permissive- /Zc:__cplusplus /EHsc>
                $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wno-unknown-pragmas>
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
                $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${USE_FAST_MATH}>>:--use_fast_math>
        )
    endforeach()
else()
    message(WARNING "No drivers found under driver/**/*.cpp — no executables will be built.")
endif()

# ---------- Optional installs ----------
# install(TARGETS <your_apps...> RUNTIME DESTINATION bin)

cmake_minimum_required(VERSION 3.16)
project(TP_OpenCV LANGUAGES CXX)


# === Compiler flags (équivalent CFLAGS) ===
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
  -fPIE
  -Wall -Wextra -Wpedantic
  -Werror
  -Wno-unused-parameter
  -ggdb
)

# Include dir (équivalent -I./lib/)
include_directories(${CMAKE_SOURCE_DIR}/lib)


# === OpenCV ===
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# === Sources communes ===
set(COMMON_SRC ${CMAKE_SOURCE_DIR}/src/common.cpp)

# Helpers
function(add_com_exe exe_name com_cpp extra_src)
  add_executable(${exe_name}
    ${CMAKE_SOURCE_DIR}/src/com/${com_cpp}
    ${COMMON_SRC}
    ${extra_src}
  )

  set_target_properties(${exe_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
  )

  target_link_libraries(${exe_name} PRIVATE ${OpenCV_LIBS})
  target_include_directories(${exe_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
endfunction()

# === TP1 (tpHistogram) ===
set(TP1_SRC ${CMAKE_SOURCE_DIR}/src/tpHistogram.cpp)
add_com_exe(inverse         inverse.cpp         ${TP1_SRC})
add_com_exe(threshold       threshold.cpp       ${TP1_SRC})
add_com_exe(quantize        quantize.cpp        ${TP1_SRC})
add_com_exe(normalize       normalize.cpp       ${TP1_SRC})
add_com_exe(equalize        equalize.cpp        ${TP1_SRC})
add_com_exe(thresholdOtsu   thresholdOtsu.cpp   ${TP1_SRC})

# === TP2 (tpConnectedComponents) ===
set(TP2_SRC ${CMAKE_SOURCE_DIR}/src/tpConnectedComponents.cpp)
add_com_exe(ccLabel         ccLabel.cpp         ${TP2_SRC})
add_com_exe(ccAreaFilter    ccAreaFilter.cpp    ${TP2_SRC})
add_com_exe(ccLabel2pass    ccLabel2pass.cpp    ${TP2_SRC})

# === TP3 (tpGeometry) ===
set(TP3_SRC ${CMAKE_SOURCE_DIR}/src/tpGeometry.cpp)
add_com_exe(transpose       transpose.cpp       ${TP3_SRC})
add_com_exe(expand          expand.cpp          ${TP3_SRC})
add_com_exe(rotate          rotate.cpp          ${TP3_SRC})

# === TP4 (tpConvolution) ===
set(TP4_SRC ${CMAKE_SOURCE_DIR}/src/tpConvolution.cpp)
add_com_exe(meanFilter      meanFilter.cpp      ${TP4_SRC})
add_com_exe(convolution     convolution.cpp     ${TP4_SRC})
add_com_exe(edgeSobel       edgeSobel.cpp       ${TP4_SRC})
add_com_exe(bilateralFilter bilateralFilter.cpp ${TP4_SRC})

# === TP5 (tpMorphology) ===
set(TP5_SRC ${CMAKE_SOURCE_DIR}/src/tpMorphology.cpp)
add_com_exe(median                median.cpp                ${TP5_SRC})
add_com_exe(erode                 erode.cpp                 ${TP5_SRC})
add_com_exe(dilate                dilate.cpp                ${TP5_SRC})
add_com_exe(open                  open.cpp                  ${TP5_SRC})
add_com_exe(close                 close.cpp                 ${TP5_SRC})
add_com_exe(morphologicalGradient morphologicalGradient.cpp ${TP5_SRC})

# === test (tpCommon seulement) ===
add_com_exe(test test.cpp "")  # pas de tp*.cpp

# === clean-like target (optionnel) ===
add_custom_target(distclean
  COMMAND ${CMAKE_COMMAND} -E rm -rf
    ${CMAKE_BINARY_DIR}/CMakeFiles
    ${CMAKE_BINARY_DIR}/CMakeCache.txt
    ${CMAKE_BINARY_DIR}/cmake_install.cmake
    ${CMAKE_SOURCE_DIR}/bin/*
    ${CMAKE_SOURCE_DIR}/obj/*
)


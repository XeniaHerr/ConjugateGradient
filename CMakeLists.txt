cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_COMPILER "acpp")
project(CG_SOLVER)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(CMAKE_CXX_STANDARD 20)
find_program(ACPP NAMES "acpp")
find_package(AdaptiveCpp CONFIG REQUIRED)

#add_subdirectory(src/hiflow3/contrib)

if(ACPP_NOTFOUND)
  message(FATAL_ERROR "Couldn't find acpp, please make shure it is installed and available")
else()
  message("Found acpp at ${ACPP}")
endif()

#include_directories(src/hiflow3/build/src/include)
find_package(Doxygen OPTIONAL_COMPONENTS mscgen dia dot)


if(DOXYGEN_FOUND)
  doxygen_add_docs(doc src)
endif()



add_compile_options("-O3")
include_directories(src)
include_directories(src/hiflow3/build/src/include)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/hiflow3/cmake")
add_subdirectory(src/hiflow3)

add_subdirectory(test)
#add_subdirectory(poisson)
#include(poisson/CMakeLists.txt)





#enable_testing()

#The second include directories statement doesn't seem to have any effect. I have a file config.h in src/hiflow3/src, 


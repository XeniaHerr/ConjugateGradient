cmake_minimum_required(VERSION 3.14)
project(CG_SOLVER)

find_program(ACPP NAMES "acpp")

if(ACPP_NOTFOUND)
  message(FATAL_ERROR "Couldn't find acpp, please make shure it is installed and available")
else()
  message("Found acpp at ${ACPP}")
endif()


set(CMAKE_CXX_COMPILER "acpp")

add_compile_options("-O3")
include_directories(src)
#add_subdirectory(src)
add_subdirectory(test)




enable_testing()



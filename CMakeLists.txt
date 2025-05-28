cmake_minimum_required(VERSION 3.14)
project(CG_SOLVER)

find_program(ACPP NAMES "acpp")

if(ACPP_NOTFOUND)
  message(FATAL_ERROR "Couldn't find acpp, please make shure it is installed and available")
else()
  message("Found acpp at ${ACPP}")
endif()



set(CMAKE_CXX_COMPILER "acpp")

add_executable(tester src/Tester.cpp src/mm_reader.cpp)


enable_testing()



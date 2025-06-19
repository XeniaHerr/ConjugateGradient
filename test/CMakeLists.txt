
add_executable(tester Tester.cpp mm_reader.cpp)
add_sycl_to_target(TARGET tester)

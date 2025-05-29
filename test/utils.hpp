/**
* Utility funcions like Time measuring, Logging and so on...
*/



#include <ratio>
#if __cplusplus > 201703L
//    1 = 2;
#endif

#include <chrono>
#include <vector>
#include <tuple>

#include <cassert>
#include <string>
#include <tuple>
#include <vector>




struct Timer {

Timer() { }


    void start_measure() {
	_begin_of_measure = std::chrono::steady_clock::now();
    }


    void stop_measure() {
	_end_of_measure = std::chrono::steady_clock::now();
	duration = _end_of_measure - _begin_of_measure;

    }


    std::chrono::duration<double, std::milli> get_duration() {
	return this->duration;
    }


private:
    std::chrono::steady_clock::time_point _begin_of_measure;
    std::chrono::steady_clock::time_point _end_of_measure;
    std::chrono::duration<double, std::milli> duration;

};



// File operations



//Implemented in mm_reader.cpp
std::tuple<std::vector<double>, std::vector<int>, std::vector<int>> read_file(std::string filename);




// Use the method of Manifactured Solutions to create a testcase 
template<typename T>
std::vector<T> generate_random_vector(std::size_t N, T min, T max);






/*
* @file: Read a Matrix market file 
*/

#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>



struct Matrixinfo {


    enum {
	array,
	coordinate
    } format;

    enum {

	real,
	complex,
	integer,
	pattern
    } type;

    enum {
	general,
	symmetric,
	skewsymmetric,
	hermitian,
    } qualifier;
};

static std::vector<std::string> split_words(std::string s) {
    std::istringstream iss(s);
    std::vector<std::string> words;
    std::string word;
    while(iss >> word) {
	words.push_back(word);
    }
    return words;
}


//Overload this function 4 times with double, int, bool and complex = std::complex
static std::tuple<std::vector<double>, std::vector<int>, std::vector<int>> read_coordinate_matrix(std::ifstream& f) {


    std::string line;
    std::getline(f, line);
    auto words = split_words(line);
   const int NNZ = std::stoi(words[2]);
   const int M = std::stoi(words[0]);

    //std::cout << "M = " << M << " ,NNZ = " << NNZ << std::endl;


    std::vector<double> data;
    std::vector<int> cols;
    std::vector<int> rows;

    std::vector<std::tuple<int, int, double>> coordinates;
    
    //std::cout << "Declarations done " << std::endl;

    /* HOW TO CREATE CSR FROM COORDINATES
    1. make a triple out of each line
    2. insert them all in a vector (size == NNZ)
    3. for the first NNZ elements in this vector, add a mirrored element if n != m
    4. sort the vector with respect to row, then column
    5. this is basically the representation. Reduce the row one fold it. 
    6. Done
    */

    int n, m, h = 0;
    double value;
    //while(std::getline(f, line)) {
    while( f >> n >> m >> value) {
	//std::cout << "step" << std::endl;

	//std::getline(f, line);
	words = split_words(line);
	//int m = std::stoi(words[0]);
	//int n = std::stoi(words[1]);
	//double value = std::stod(words[2]);
	coordinates.emplace_back(std::make_tuple(n-1,m-1,value));

	h++;

    }

    //std::cout << "Coordinates are read in, h = " << h << " NNZ = " << NNZ << std::endl;



    for (int i = 0; i < h; i++) {
	if (std::get<0>(coordinates[i]) != std::get<1>(coordinates[i])){
	    coordinates.emplace_back(std::make_tuple(
		std::get<1>(coordinates[i]),
		std::get<0>(coordinates[i]),
		std::get<2>(coordinates[i])
	    ));
	}
    }

    //std::cout << "Coordinates are mirrored" << std::endl;

    auto sorter = [&](std::tuple<int,int,double> a, std::tuple<int, int,double> b) {
	if (std::get<0>(a) < std::get<0>(b))
	    return true;
	else if ((std::get<0>(a) == std::get<0>(b)) && (std::get<1>(a) <= std::get<1>(b)))
		return true;
	else
	    return false;
    };


    std::sort(coordinates.begin(), coordinates.end(), sorter);

    //std::cout << "Sorted " << std::endl;

    //std::cout << "Coordinates containes now " << coordinates.size() << " entries" << std::endl;

    //for (auto a : coordinates)
	 //std::cout << std::get<0>(a) << " " << std::get<1>(a) << " " << std::get<2>(a) << "\n";


    int helper = 0;
    data.resize(coordinates.size());
    cols.resize(coordinates.size());
    
	data[0] = std::get<2>(coordinates[0]);
	cols[0] = std::get<1>(coordinates[0]); //changing o get to 1 get
	rows.push_back(0);
    for (int i = 1; i < coordinates.size(); i++) {
	data[i] = std::get<2>(coordinates[i]);
	cols[i] = std::get<1>(coordinates[i]);
	if (std::get<0>(coordinates[i]) > std::get<0>(coordinates[i-1])) {
	    rows.push_back(i);

	}
	helper++;
    }

    rows.push_back(coordinates.size());



    //for (auto a : rows)
	//std::cout << a << " ";
    //std::cout << std::endl;


    //std::cout << "Rows: " << rows.size() << std::endl;

    //std::cout << "Vectors are made, exiting " << __FUNCTION__ << std::endl;

    //assert(M == rows.size());
    //assert(NNZ == data.size());
    //The above assert might be correct, i have to look at the fileformat agein

    return std::make_tuple(data, cols, rows);

}



Matrixinfo parse_header(std::ifstream& f) {

    Matrixinfo info;

    std::string line;
    std::getline(f,line);

    auto words = split_words(line);

    assert(words.size() == 5);

    if (words[2] == "coordinate")
	info.format = Matrixinfo::coordinate;
    else
	info.format = Matrixinfo::array;


    if( words[3] == "real")
	info.type = Matrixinfo::real;
    else if (words[3] == "complex")
	info.type = Matrixinfo::complex;
    else if (words[3] == "integer")
	info.type = Matrixinfo::integer;
    else
	info.type = Matrixinfo::pattern;


    if (words[4] == "symmetric")
	info.qualifier = Matrixinfo::symmetric;
    else if (words[4] == "skew-symmetric")
	info.qualifier = Matrixinfo::skewsymmetric;
    else if ( words[4] == "Hermitian" && words[3] == "complex")
	info.qualifier = Matrixinfo::hermitian;
    else
	info.qualifier = Matrixinfo::general;


    return info;
}




std::tuple<std::vector<double>, std::vector<int>, std::vector<int>> 
read_file(std::string filename) {

    std::ifstream f(filename);

    std::string header;
    //std::getline(f, header);
    //auto words = split_words(header);


    Matrixinfo info = parse_header(f);
    //std::cout << "Format " << info.format << std::endl;
    //std::cout << "Type " << info.type << std::endl;
    //std::cout << "Qualifier " << info.qualifier << std::endl;

    std::string line;

    std::getline(f, line);
    do {
    std::getline(f, line);

} while (line.substr(0,2) != "%-");

     auto result = read_coordinate_matrix(f);

    return result;

    //std::cout << "Data " << std::endl;
    for (auto& a : std::get<0>(result)) {
	//std::cout << a << ", ";

    }
    
    //std::cout << std::endl;

    //std::cout << "Cols " << std::endl;
    for (auto& a : std::get<1>(result)) {
	//std::cout << a << ", ";

    }
    
    //std::cout << std::endl;

    //std::cout << "Rows" << std::endl;
    for (auto& a : std::get<2>(result)) {
	//std::cout << a << ", ";

    }
    
    //std::cout << std::endl;
    //This line is the first line of the matrix
    //std::getline(f, line);
    //std::cout << line;
}





/*
int main() {
    read_file("Hello.txt");
}

*/


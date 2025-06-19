/*
 * @file: Read a Matrix market file
 */

#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

struct Matrixinfo {

  enum { array, coordinate } format;

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
  while (iss >> word) {
    words.push_back(word);
  }
  return words;
}

// Overload this function 4 times with double, int, bool and complex =
// std::complex
static std::tuple<std::vector<double>, std::vector<int>, std::vector<int>>
read_real_coordinate_matrix(std::ifstream &f) {

  std::string line;
  std::getline(f, line);
  auto words = split_words(line);
  const int NNZ = std::stoi(words[2]);
  const int M = std::stoi(words[0]);

  std::vector<double> data;
  std::vector<int> cols;
  std::vector<int> rows;

  std::vector<std::tuple<int, int, double>> coordinates;

  int n, m, h = 0;
  double value;
  while (f >> n >> m >> value) {
    words = split_words(line);
    coordinates.emplace_back(std::make_tuple(n - 1, m - 1, value));
    h++;
  }

  for (int i = 0; i < h; i++) {
    if (std::get<0>(coordinates[i]) != std::get<1>(coordinates[i])) {
      coordinates.emplace_back(std::make_tuple(std::get<1>(coordinates[i]),
                                               std::get<0>(coordinates[i]),
                                               std::get<2>(coordinates[i])));
    }
  }

  auto sorter = [&](std::tuple<int, int, double> a,
                    std::tuple<int, int, double> b) {
    if (std::get<0>(a) < std::get<0>(b))
      return true;
    else if ((std::get<0>(a) == std::get<0>(b)) &&
             (std::get<1>(a) <= std::get<1>(b)))
      return true;
    else
      return false;
  };

  std::sort(coordinates.begin(), coordinates.end(), sorter);

  int helper = 0;
  data.resize(coordinates.size());
  cols.resize(coordinates.size());

  data[0] = std::get<2>(coordinates[0]);
  cols[0] = std::get<1>(coordinates[0]); // changing o get to 1 get
  rows.push_back(0);
  for (int i = 1; i < coordinates.size(); i++) {
    data[i] = std::get<2>(coordinates[i]);
    cols[i] = std::get<1>(coordinates[i]);
    if (std::get<0>(coordinates[i]) > std::get<0>(coordinates[i - 1])) {
      rows.push_back(i);
    }
    helper++;
  }

  rows.push_back(coordinates.size());

  return std::make_tuple(data, cols, rows);
}

Matrixinfo parse_header(std::ifstream &f) {

  Matrixinfo info;

  std::string line;
  std::getline(f, line);

  auto words = split_words(line);

  assert(words.size() == 5);

  if (words[2] == "coordinate")
    info.format = Matrixinfo::coordinate;
  else
    info.format = Matrixinfo::array;

  if (words[3] == "real")
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
  else if (words[4] == "Hermitian" && words[3] == "complex")
    info.qualifier = Matrixinfo::hermitian;
  else
    info.qualifier = Matrixinfo::general;

  return info;
}

static void skip_comments(std::ifstream &f) {

  std::string line;

  while (f.peek() == '%')
    std::getline(f, line);
}

std::tuple<std::vector<double>, std::vector<int>, std::vector<int>>
read_file(std::string filename) {

  std::ifstream f(filename);

  std::string header;

  Matrixinfo info = parse_header(f);

  std::string line;
  std::getline(f, line);

  // Skip all comments, they are unneccecary
  skip_comments(f);
  auto result = read_real_coordinate_matrix(f);

  return result;
}

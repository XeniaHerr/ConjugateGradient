#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <AdaptiveCpp/sycl/handler.hpp>
#include <AdaptiveCpp/sycl/queue.hpp>
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <AdaptiveCpp/sycl/usm.hpp>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

namespace CGSolver {

namespace asycl = acpp::sycl;

template <class DT> struct Asycl_deleter {
  asycl::queue _q;

  Asycl_deleter(asycl::queue &q) : _q(q) {}

  void operator()(DT *_ptr) { asycl::free(_ptr, _q); }
};

template <class DT> class Matrix {

public:
  Matrix(asycl::queue q) : _queue(q), _size(0) {}
  Matrix(asycl::queue q, const std::size_t size) : _queue(q), _size(size) {}
  Matrix(asycl::queue q, std::vector<DT> &data, std::vector<int> &cols,
         std::vector<int> &rows)
      : _queue(q), _size(rows.size()), _N(rows.size() - 1), _NNZ(data.size()) {
    init(data, cols, rows);
    _queue.wait(); // TODO: Handle exceptions
  }

  // Matrix(Matrix &&other)
  //     : _queue(other._queue), _N(other._N), _NNZ(other._NNZ),
  //       _size(other._size), _data(other._data), _columns(other._columns),
  //       _rows(other._rows) {

  //   other._data = nullptr;
  //   other._columns = nullptr;
  //   other._rows = nullptr;
  //   other._N = 0;
  //   other._NNZ = 0;
  //   other._size= 0;
  // }

  // Matrix &operator=(Matrix &&other) {

  //   this->_data = other._data;
  //   this->_columns = other._columns;
  //   this->_rows = other._rows;
  //   this->_queue = other._queue;
  //   this->_N = other._N;
  //   this->_NNZ = other._NNZ;
  //   this->_size = other._size;

  //   other._data = nullptr;
  //   other._columns = nullptr;
  //   other._rows = nullptr;
  //   other._N = 0;
  //   other._NNZ = 0;
  //   other._size = 0;

  //   return *this;
  // }

  auto data() { return _data; }
  auto data_ptr() { return _data.get(); }
  auto columns() { return _columns; }
  auto columns_ptr() { return _columns.get(); }
  auto rows() { return _rows; }
  auto rows_ptr() { return _rows.get(); }

  auto N() const { return _N; }
  auto NNZ() const { return _NNZ; }

  auto init(std::vector<DT> &data, std::vector<int> &cols,
            std::vector<int> &rows) {

    _size = (_size == 0) ? rows.size() : _size;

    _N = _size - 1;
    _NNZ = data.size();

    _data = std::shared_ptr<DT[]>(asycl::malloc_device<DT>(_NNZ, _queue),
                                  Asycl_deleter<DT>(_queue));
    _columns = std::shared_ptr<int[]>(asycl::malloc_device<int>(_NNZ, _queue),
                                      Asycl_deleter<int>(_queue));
    _rows = std::shared_ptr<int[]>(asycl::malloc_device<int>(_size, _queue),

                                   Asycl_deleter<int>(_queue));

    _queue.copy(data.data(), _data.get(), _NNZ);
    _queue.copy(cols.data(), _columns.get(), _NNZ);
    _queue.copy(rows.data(), _rows.get(), _size);
  }

private:
  std::size_t _size;
  std::size_t _NNZ;
  std::size_t _N;
  asycl::queue _queue;

  std::shared_ptr<DT[]> _data;
  std::shared_ptr<int[]> _columns;
  std::shared_ptr<int[]> _rows;
};

template <class DT> class Vector {

public:
  explicit Vector(asycl::queue q) : _q(q), _N(0) {}
  explicit Vector(asycl::queue q, const std::size_t N) : _q(q), _N(N) {}

  explicit Vector(asycl::queue q, std::vector<DT> &data) : _q(q) {

    init(data);

    _q.wait();
  }

  // explicit Vector(Vector&& other) : _q(other._q), _N(other._N), _ptr((other._ptr)) {
  //   other._ptr = nullptr;
  //   other._N = 0;

    
  // }

  // Vector& operator=(Vector &&other) {

  //   this->_q = other._q;
  //   this->_N = other._N;
  //   this->_ptr = other._ptr;
  //   other._ptr = nullptr;
  //   other._N = 0;

  //   return *this;
  // }
 
  asycl::event init_empty(std::size_t size = 0) {

    if (size != 0 && _N == 0)
      _N = size;

    assert(_N != 0);

    _ptr = std::shared_ptr<DT[]>(asycl::malloc_device<DT>(_N, _q),
                                 Asycl_deleter<DT>(_q));

    return _q.fill(_ptr.get(), static_cast<DT>(0), _N);
  }

  asycl::event init(std::vector<DT> &data) {

    _ptr = std::shared_ptr<DT[]>(asycl::malloc_device<DT>(data.size(), _q),
                                 Asycl_deleter<DT>(_q));

    return _q.copy(data.data(), _ptr.get(), data.size());
  }

  auto data() { return _ptr; }

  auto ptr() { return _ptr.get(); }

  auto N() { return _N; }

private:
  asycl::queue _q;

  std::size_t _N;

  std::shared_ptr<DT[]> _ptr;
};

template <class DT> class Scalar {

public:
  Scalar(asycl::queue q, DT value = static_cast<DT>(0)) : _q(q) {

    init(value);

    _q.wait();
  }

  // Scalar(Scalar && other) : _q(other._q), value(other.value) {
  //   other.value = nullptr;
    
  // }

  // Scalar(const Scalar& other) = default;

  // Scalar& operator=(Scalar&& other) {
  //   this->_q = other._q;
  //   this->value = other.value;
  //   other.value = nullptr;
  // }

  auto init(DT value) {

    this->value = std::shared_ptr<DT>(asycl::malloc_device<DT>(1, _q),
                                      Asycl_deleter<DT>(_q));

    _q.copy(&value, this->value.get(), 1);
  }

  auto ptr() { return value.get(); }

  operator DT *() const { // Maybe make this constexpr

    return value.get();
  }

private:
  asycl::queue _q;

  std::shared_ptr<DT> value;
};

} // namespace CGSolver

#endif /*MATRIX_HPP*/

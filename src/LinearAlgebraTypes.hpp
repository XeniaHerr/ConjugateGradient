/**
 * @file LinearAlgebraTypes.hpp
 *Storage Classes for Linear Algebra Objects.
 *
 * This class provides Classes to manage the data for Matrizies, Vectors and
 *Scalars. They don't provide Actual mathematical operations, but are only
 *focused with allcating and freeing memory on the device. */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <AdaptiveCpp/sycl/handler.hpp>
#include <AdaptiveCpp/sycl/queue.hpp>
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <AdaptiveCpp/sycl/usm.hpp>
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace CGSolver {

namespace asycl = acpp::sycl;

/**
 * @class Asycl_deleter
 *
 * @brief Helper deallocator class
 *
 * Provide a deallocator Functor that can deallocate provided memory when
 * called. An instance of this class is used as an argument in each shared_ptr
 * construction to enable RAII Principles.*/
template <class DT> struct Asycl_deleter {
  asycl::queue _q;

  Asycl_deleter(asycl::queue &q) : _q(q) {}

  void operator()(DT *_ptr) { asycl::free(_ptr, _q); }
};

/**
 * @class Matrix
 *
 * @brief Store Matrix on device
 *
 * Store a Matrix in CSR-Fromat on the device.*/
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

  /**
   * @brief get data smartpointer*/
  auto data() { return _data; }
  /**
   * @brief get data raw pointer*/
  auto data_ptr() { return _data.get(); }
  /**
   * @brief get columns smartpointer*/
  auto columns() { return _columns; }
  /**
   * @brief get columns raw pointer*/
  auto columns_ptr() { return _columns.get(); }
  /**
   * @brief get row smartpointer*/
  auto rows() { return _rows; }
  /**
   * @brief get row raw pointer*/
  auto rows_ptr() { return _rows.get(); }

  /**
   * @brief get Matrix dimension*/
  auto N() const { return _N; }
  /**
   * @brief get Number of Non zero Entries*/
  auto NNZ() const { return _NNZ; }

  /**
   * @brief Init a Matrix from given Vectors
   *
   * @param data std::vector of Matrix entries
   * @param cols std::vector of column idizes of Entries
   * @param rows std::vector of row offsets for cols vector*/
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

/**
 * @class Vector
 *
 * @brief Store Vector on device
 *
 * This Vector can be initialized without a size, but doesn't provide
 *capabillities to be resized.
 *
 **/
template <class DT> class Vector {

public:
  explicit Vector(asycl::queue q) : _q(q), _N(0) {}
  explicit Vector(asycl::queue q, const std::size_t N) : _q(q), _N(N) {}

  explicit Vector(asycl::queue q, std::vector<DT> &data) : _q(q) {

    init(data);

    _q.wait();
  }

  /**
   * @brief Init an Vector with zeroes
   *
   * @param size size of the vector*/
  asycl::event init_empty(std::size_t size = 0) {

    if (size != 0 && _N == 0)
      _N = size;

    assert(_N != 0);

    _ptr = std::shared_ptr<DT[]>(asycl::malloc_device<DT>(_N, _q),
                                 Asycl_deleter<DT>(_q));

    return _q.fill(_ptr.get(), static_cast<DT>(0), _N);
  }

  /**
   * @brief Init an Vector from a std::vector
   *
   * @param data reference to std::vector to be copied from*/
  asycl::event init(std::vector<DT> &data) {

    _ptr = std::shared_ptr<DT[]>(asycl::malloc_device<DT>(data.size(), _q),
                                 Asycl_deleter<DT>(_q));

    return _q.copy(data.data(), _ptr.get(), data.size());
  }

  /**
   * @brief get accecc to smartpointer*/
  auto data() { return _ptr; }

  /**
   * @brief get accesss to raw pointer*/
  auto ptr() { return _ptr.get(); }

  /**
   * @brief get size*/
  auto N() { return _N; }

private:
  asycl::queue _q;

  std::size_t _N;

  std::shared_ptr<DT[]> _ptr;
};
/**
 * @class Scalar
 *
 * @brief Store Scalar value on Device
 *
 * */
template <class DT> class Scalar {

public:
  Scalar(asycl::queue q, DT value = static_cast<DT>(0)) : _q(q) { init(value); }
  /**
   *@brief set initial value
   **/
  auto init(DT value) {

    this->value = std::shared_ptr<DT>(asycl::malloc_device<DT>(1, _q),
                                      Asycl_deleter<DT>(_q));

    _q.copy(&value, this->value.get(), 1);
    _q.wait();
  }

  /**
   * @brief access the poiner
   *
   */
  auto ptr() { return value.get(); }

  /**
   * @brief cast to a raw Pointer for ergonomics
   *
   */
  operator DT *() const { // Maybe make this constexpr

    return value.get();
  }

private:
  asycl::queue _q;

  std::shared_ptr<DT> value;
};

} // namespace CGSolver

#endif /*MATRIX_HPP*/

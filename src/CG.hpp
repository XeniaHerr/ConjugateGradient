#ifndef CG_HPP
#define CG_HPP

// IMPORTANT: Only include sycl.hpp header. All other headers might lead to
// compilation errors
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "VectorOperations.hpp"

namespace CGSolver {

namespace asycl = acpp::sycl;

constexpr static size_t TILE = 128;

template <typename DT> struct Matrix {

  DT *data;
  int *columns;
  int *rows;

  int NNZ;
  int N;
};

template <typename DT> class CG {

public:
  // Do not allocate Memory at the beginning
  CG(asycl::queue &queue) : _queue(queue) {
    b = nullptr;
    x = nullptr;
    A.columns = nullptr;
    A.rows = nullptr;
    A.data = nullptr;
  }

  void setMatrix(std::vector<DT> &_data, std::vector<int> &columns,
                 std::vector<int> &_rows) {

    const int NNZ = _data.size();
    const int N = _rows.size() - 1;

    auto m = std::max_element(columns.begin(), columns.end());

    this->A.NNZ = NNZ;
    this->A.N = N;
    this->N = A.N;

    this->A.data = asycl::malloc_device<DT>(NNZ, this->_queue);
    this->A.columns = asycl::malloc_device<int>(NNZ, this->_queue);

    this->A.rows = asycl::malloc_device<int>(_rows.size(), this->_queue);

    this->_queue.copy(_data.data(), this->A.data, NNZ);
    this->_queue.copy(columns.data(), this->A.columns, NNZ);
    this->_queue.copy(_rows.data(), this->A.rows, N + 1);
    _queue.wait();
  }

  int getDimension() const { return this->A.N; }

  void setTarget(std::vector<DT> &_data) {

    this->b = asycl::malloc_device<DT>(_data.size(), this->_queue);

    if (this->A.N != 0)
      assert(this->A.N == _data.size());

    this->_queue.copy(_data.data(), this->b, _data.size());

    _queue.wait();
  }

  void setInital(std::vector<DT> &_data) {
    this->x = asycl::malloc_device<DT>(_data.size(), this->_queue);

    this->_queue.copy(_data.data(), this->x, _data.size());
    _queue.wait();
  }

  void solve(DT error = static_cast<DT>(0)) {

    VectorOperations<DT> vecops(this->_queue);
    vecops.setVectorSize(A.N);

    bool done = false;

    if (this->b == nullptr) {
      throw std::runtime_error("No right hand side to solve for");
    }

    if (this->A.columns == nullptr) {
      throw std::runtime_error("No Matrix given");
    }

    // Helper buffers
    DT *helper = asycl::malloc_device<DT>(A.N, _queue);
    DT *r = asycl::malloc_device<DT>(A.N, _queue);
    DT *rnext = asycl::malloc_device<DT>(A.N, _queue);
    DT *p = asycl::malloc_device<DT>(A.N, _queue);
    DT *rxr = asycl::malloc_device<DT>(1, _queue);
    DT *value2 = asycl::malloc_device<DT>(1, _queue);
    DT *value3 = asycl::malloc_device<DT>(1, _queue);
    DT *alpha = asycl::malloc_device<DT>(1, _queue);
    DT *beta = asycl::malloc_device<DT>(1, _queue);

    // Must be available to both host and device
    bool *is_done = asycl::malloc_shared<bool>(1, _queue);

    if (this->x == nullptr) {
      this->x = asycl::malloc_device<DT>(A.N, this->_queue);
    }
    _queue.fill(x, static_cast<DT>(0), A.N);

    _queue.fill(r, static_cast<DT>(0), A.N);
    _queue.fill(rnext, static_cast<DT>(0), A.N);
    _queue.fill(p, static_cast<DT>(0), A.N);
    *is_done = false;

    _queue.wait();

    if (!r || !rnext || !p || !helper || !x) {
      throw std::runtime_error("Failed to allocate memory");
    }

    int counter = 0;
    DT res;

    // Compute r0 = b - Ax0
    auto initializer = _queue.submit([&](asycl::handler &chg) {
      auto rows = A.rows;
      auto columns = A.columns;
      auto data = A.data;
      auto x = this->x;
      auto b = this->b;

      chg.parallel_for<class InitializeVectors>(
          this->A.N, [=](asycl::item<1> id) {
            DT single_result = 0;
            for (int j = rows[id]; j < rows[id + 1]; j++) {
              single_result += data[j] * x[columns[j]];
            }
            r[id] = b[id] - single_result;
            p[id] = r[id];
            rnext[id] = p[id];
          });
    });

    auto erxr = vecops.dot_product(r, r, rxr, 0, {initializer});
    _queue.wait();

    do {

      // Make Clean enviromnet
      auto eclean_helper = _queue.fill(helper, static_cast<DT>(0), A.N);

      auto eresetter = _queue.submit([&](asycl::handler &chg) {
        chg.single_task<class CleanUp>([=]() {
          *value2 = 0;
          *value3 = 0;
          *alpha = 0;
          *beta = 0;
        });
      });

      //  A * p
      auto evector_matrix_product =
          vecops.spmv(A.data, A.columns, A.rows, p, helper, A.NNZ, 0,
                      {eclean_helper, eresetter});

// scalarproduct of AP with p
      auto eapxp = vecops.dot_product(helper, p, value2, 0, {evector_matrix_product});

      // Alpha = rxr/ value2
      auto ealpha = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(erxr);
        chg.depends_on(eapxp);
        chg.single_task<class AlphaCalculation>(
            [=]() { *alpha = *rxr / *value2; });
      });

      // Calculate x_k+1 and r_k+1
      auto exnext = vecops.sapbx(x, p, alpha, x, 0, {ealpha});

      auto ernext = vecops.sambx(rnext, helper, alpha, rnext, 0,
                                 {ealpha, evector_matrix_product});

      // Check size of rk
      auto eaccuracy = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(ernext);

        chg.single_task([=]() {
          if (asycl::sqrt(*rxr) <= error)
            *is_done = true;
        });
      });

      auto ernextxrnext = vecops.dot_product(rnext, rnext, value3, 0, {ernext});


      // Calculate beta
      auto ebeta = _queue.single_task<class BetaCalculation>(
          {ernextxrnext, erxr}, [=]() {
            *beta = *value3 / *rxr;
            *rxr = *value3;
          });
      auto epnext = vecops.sapbx(rnext, p, beta, p, 0, {ebeta, ernext});

      auto enext_step = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(ebeta);
        chg.copy(rnext, r, A.N);
      });

      _queue.wait();

      // One Algorithm iteration done

      std::cout << "\r\033[2K";
      std::cout << ((static_cast<double>(counter) / A.N) * 100) << "%";
      std::flush(std::cout);

    } while (counter++ < A.N && !(*is_done));

    std::cout << std::endl;
    // Clean up helpers
    asycl::free(helper, _queue);
    asycl::free(rxr, _queue);
    asycl::free(value2, _queue);
    asycl::free(beta, _queue);
    asycl::free(alpha, _queue);
    asycl::free(value3, _queue);
    asycl::free(r, _queue);
    asycl::free(rnext, _queue);
    asycl::free(p, _queue);

    this->is_solved = true;
  }


  // Calculate the relative error
  DT accuracy() {

    std::cout << "accuracy" << std::endl;
    DT *normres = asycl::malloc_device<DT>(1, _queue);
    DT *normx = asycl::malloc_device<DT>(1, _queue);

    _queue
        .single_task([=]() {
          *normres = 0;
          *normx = 0;
        })
        .wait();

    _queue.submit([&](asycl::handler &chg) {
      auto SumRed = asycl::reduction(normres, asycl::plus<DT>());
      auto SumRed2 = asycl::reduction(normx, asycl::plus<DT>());

      auto rows = this->A.rows;
      auto columns = this->A.columns;
      auto data = this->A.data;
      auto x = this->x;
      auto b = this->b;

      chg.parallel_for<class AccuracyOperation>(
          A.N, SumRed, SumRed2, [=](asycl::item<1> id, auto &sum, auto &xsum) {
            DT single_result = 0;
            for (int j = rows[id]; j < rows[id + 1]; j++) {
              single_result += data[j] * x[columns[j]];
            }

            auto a = b[id] - single_result;
            sum += a * a;
            xsum += x[id] * x[id];
          });
    });

    _queue.wait();

    // return abs_error;
    DT *host_norm_res = new DT;
    DT *host_norm_x = new DT;
    _queue.copy(normres, host_norm_res, 1);
    _queue.copy(normx, host_norm_x, 1);
    _queue.wait();

    DT abs_error = std::abs((*host_norm_res) / *host_norm_x);
    asycl::free(normres, _queue);
    asycl::free(normx, _queue);
    delete host_norm_res;
    delete host_norm_x;

    return abs_error;
  }


  std::vector<DT> extract() {

    std::vector<DT> ret(A.N);

    this->_queue.copy(this->x, ret.data(), A.N).wait();
    return ret;
  }

  void extractTo(std::vector<DT> &result) {}

  std::size_t memoryFootprint() const {
    return (2 * this->A.NNZ + (6 * this->A.N));
  }

  ~CG() {
    if (A.data != nullptr)
      asycl::free(A.data, _queue);
    if (A.columns != nullptr)
      asycl::free(A.columns, _queue);
    if (A.data != nullptr)
      asycl::free(A.rows, _queue);
    if (x != nullptr)
      asycl::free(x, _queue);
    if (b != nullptr)
      asycl::free(b, _queue);
  }

private:
  void printVector(DT *data, std::string name) {

    std::vector<DT> vec(A.N);
    _queue.copy(data, vec.data(), A.N).wait();

    std::cout << name << " = [";
    for (auto &a : vec)
      std::cout << a << " ";
    std::cout << "]\n";
  }

  asycl::queue _queue;

  bool is_solved = false;

  size_t max_workgroup_count;

  Matrix<DT> A;

  DT *x;

  DT *b;

  size_t N;
};

}; // namespace CGSolver

#endif /*CG_HPP */

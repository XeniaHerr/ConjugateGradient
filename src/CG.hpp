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

#include "Matrix.hpp"
#include "VectorOperations.hpp"

namespace CGSolver {

namespace asycl = acpp::sycl;

constexpr static size_t TILE = 128;

enum Debuglevel {
  None,
  Verbose,

};

template <typename DT, Debuglevel debug = Debuglevel::None> class CG {

public:
  // Do not allocate Memory at the beginning
  CG(asycl::queue &queue) : _queue(queue), A(queue), x(_queue), b(_queue) {

    if constexpr (debug == Debuglevel::Verbose)
      std::cout << "Constructing CG Object\n";
    // A.columns = nullptr;
    // A.rows = nullptr;
    // A.data = nullptr;
  }

  void setMatrix(std::vector<DT> &data, std::vector<int> &columns,
                 std::vector<int> &rows) {
    if constexpr (debug == Debuglevel::Verbose)
      std::cout << "Setting Matrix\n";

    A.init(data, columns, rows);
  }

  int getDimension() const { return this->A.N(); }

  void setTarget(std::vector<DT> &_data) {

    b.init(_data);

    _queue.wait();
  }

  void setInital(std::vector<DT> &_data) {
    x.init(_data);
    _queue.wait();
  }

  void solve(DT improvement = static_cast<DT>(0)) {

    if constexpr (debug == Debuglevel::Verbose)
      std::cout << "Solving System\n";

    VectorOperations<DT> vecops(this->_queue);
    vecops.setVectorSize(A.N());

    bool done = false;

    if (this->b.data() == nullptr) {
      throw std::runtime_error("No right hand side to solve for");
    }

    if (this->A.columns().get() == nullptr) {
      throw std::runtime_error("No Matrix given");
    }

    const auto N = A.N();

    Vector<DT> helper(_queue, N);
    Vector<DT> r(_queue, N);
    Vector<DT> rnext(_queue, N);
    Vector<DT> p(_queue, N);
    Scalar<DT> rxr(_queue);
    Scalar<DT> value2(_queue);
    Scalar<DT> value3(_queue);
    Scalar<DT> alpha(_queue);
    Scalar<DT> beta(_queue);

    // Must be available to both host and device
    bool *is_done = asycl::malloc_shared<bool>(1, _queue);

    x.init_empty(N);
    r.init_empty(N);
    rnext.init_empty(N);
    p.init_empty(N);
    helper.init_empty();
    *is_done = false;

    _queue.wait();

    if constexpr (debug == Debuglevel::Verbose) {
      std::cout << "Prepared Memory" << std::endl;
    }

    int counter = 0;
    DT res;

    // Compute r0 = b - Ax0
    auto initializer = _queue.submit([&](asycl::handler &chg) {
      int *rows = A.rows().get();
      int *columns = A.columns().get();
      DT *data = A.data().get();
      auto x = this->x.ptr();
      auto b = this->b.ptr();
      auto _r = r.ptr();
      auto _rnext = rnext.ptr();
      auto _p = p.ptr();

      chg.parallel_for<class InitializeVectors>(N, [=](asycl::item<1> id) {
        DT single_result = 0;
        for (int j = rows[id]; j < rows[id + 1]; j++) {
          single_result += data[j] * x[columns[j]];
        }
        _r[id] = b[id] - single_result;
        _p[id] = _r[id];
        _rnext[id] = _p[id];
      });
    });

    if constexpr (debug == Debuglevel::Verbose) {
      std::cout << "Init done" << std::endl;
    }
    auto erxr = vecops.dot_product_optimised(r, r, rxr, 0, {initializer});
    _queue.wait();

    if constexpr (debug == Debuglevel::Verbose) {
      std::cout << "Entering Loop" << std::endl;
    }
    do {

      // Make Clean enviromnet
      auto eclean_helper = _queue.fill(helper.ptr(), static_cast<DT>(0), N);

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
          vecops.spmv(A, p, helper, A.NNZ(), 0, {eclean_helper, eresetter});

      // scalarproduct of AP with p
      auto eapxp = vecops.dot_product_optimised(helper, p, value2, 0,
                                                {evector_matrix_product});

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
          if (asycl::sqrt(*rxr) <= improvement)
            *is_done = true;
        });
      });

      auto ernextxrnext =
          vecops.dot_product_optimised(rnext, rnext, value3, 0, {ernext});

      // Calculate beta
      auto ebeta = _queue.single_task<class BetaCalculation>(
          {ernextxrnext, erxr}, [=]() {
            *beta = *value3 / *rxr;
            *rxr = *value3;
          });
      auto epnext = vecops.sapbx(rnext, p, beta, p, 0, {ebeta, ernext});

      auto enext_step = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(ebeta);
        chg.copy(rnext.ptr(), r.ptr(), N);
      });

      _queue.wait();

      // One Algorithm iteration done

      if constexpr (debug == Debuglevel::Verbose) {
        if (counter % 100 == 0) {
          std::cout << "\r\033[2K";
          std::cout << ((static_cast<double>(counter) / A.N()) * 100) << "%";
          std::flush(std::cout);
        }
      }

    } while (counter++ < N && !(*is_done));
    DT val;
    this->_queue.copy(rxr.ptr(), &val, 1);

    this->_queue.wait();

    if constexpr (debug == Debuglevel::Verbose)
      std::cout << "Last rxr = " << val << std::endl;

    this->is_solved = true;
  }

  // Calculate the relative error
  DT accuracy() {

    if constexpr (debug == Debuglevel::Verbose)
      std::cout << "accuracy" << std::endl;
    Scalar<DT> normres(_queue);
    Scalar<DT> normx(_queue);

    _queue
        .single_task([=]() {
          *normres = 0;
          *normx = 0;
        })
        .wait();

    _queue.submit([&](asycl::handler &chg) {
      auto SumRed = asycl::reduction(normres.ptr(), asycl::plus<DT>());
      auto SumRed2 = asycl::reduction(normx.ptr(), asycl::plus<DT>());

      auto rows = this->A.rows().get();
      auto columns = this->A.columns().get();
      auto data = this->A.data().get();
      auto x = this->x.ptr();
      auto b = this->b.ptr();

      chg.parallel_for<class AccuracyOperation>(
          A.N(), SumRed, SumRed2,
          [=](asycl::item<1> id, auto &sum, auto &xsum) {
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
    _queue.copy(normres.ptr(), host_norm_res, 1);
    _queue.copy(normx.ptr(), host_norm_x, 1);
    _queue.wait();

    DT abs_error = std::abs((*host_norm_res) / *host_norm_x);
    delete host_norm_res;
    delete host_norm_x;

    return abs_error;
  }

  std::vector<DT> extract() {

    std::vector<DT> ret(A.N());

    this->_queue.copy(this->x.ptr(), ret.data(), A.N()).wait();
    return ret;
  }

  void extractTo(std::vector<DT> &result) {}

  std::size_t memoryFootprint() const {
    return (2 * this->A.NNZ() + (6 * this->A.N()));
  }

private:
  void printVector(DT *data, std::string name) {

    std::vector<DT> vec(A.N());
    _queue.copy(data, vec.data(), A.N()).wait();

    std::cout << name << " = [";
    for (auto &a : vec)
      std::cout << a << " ";
    std::cout << "]\n";
  }

  asycl::queue _queue;

  bool is_solved = false;

  size_t max_workgroup_count;

  Matrix<DT> A;

  Vector<DT> x;

  Vector<DT> b;

  size_t N;
};

}; // namespace CGSolver

#endif /*CG_HPP */

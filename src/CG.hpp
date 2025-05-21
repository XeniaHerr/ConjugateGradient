#ifndef CG_HPP
#define CG_HPP

#include "AdaptiveCpp/hipSYCL/sycl/access.hpp"
#include "AdaptiveCpp/hipSYCL/sycl/handler.hpp"
#include "AdaptiveCpp/hipSYCL/sycl/interop_handle.hpp"
#include "AdaptiveCpp/hipSYCL/sycl/libkernel/accessor.hpp"
#include "AdaptiveCpp/hipSYCL/sycl/libkernel/nd_item.hpp"
#include "AdaptiveCpp/hipSYCL/sycl/libkernel/nd_range.hpp"
#include "AdaptiveCpp/hipSYCL/sycl/libkernel/reduction.hpp"
#include "AdaptiveCpp/hipSYCL/sycl/queue.hpp"
#include "AdaptiveCpp/hipSYCL/sycl/usm.hpp"
#include "Matrix.hpp"
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace CGSolver {

namespace asycl = acpp::sycl;

constexpr static size_t TILE = 128;

template <class DT>
asycl::event dot_product(asycl::queue q, DT *a, DT *b, DT *result, size_t N) {

  const auto workgroupsize = (N + TILE - 1) / TILE;

  DT *group_results = asycl::malloc_device<DT>(workgroupsize, q);

  auto init = q.fill(group_results, static_cast<DT>(0), workgroupsize);

  auto first = q.submit([&](asycl::handler &chg) {
    chg.depends_on(init);

    asycl::local_accessor<DT, 1> local_results(TILE, chg);

    chg.parallel_for(asycl::nd_range<1>(asycl::range<1>(workgroupsize * TILE),
                                        asycl::range<1>(TILE)),
                     [=](asycl::nd_item<1> item) {
                       size_t lid = item.get_local_id(0);
                       size_t gid = item.get_global_id(0);
                       size_t group = item.get_group(0);

                       DT val =
                           (gid < N) ? a[gid] * b[gid] : static_cast<DT>(0);
                       local_results[lid] = val;

                       item.barrier(asycl::access::fence_space::local_space);

                       for (size_t offset = TILE / 2; offset > 0; offset /= 2) {
                         if (lid < offset)
                           local_results[lid] += local_results[offset + lid];
                         item.barrier(asycl::access::fence_space::local_space);
                       }

                       if (lid == 0) {
                         group_results[group] = local_results[0];
                       }
                     });
  });

  auto second = q.submit([&](asycl::handler &chg) {
    chg.depends_on(first);

    auto SumRed = asycl::reduction(result, asycl::plus<DT>());
    chg.parallel_for(workgroupsize, SumRed, [=](asycl::item<1> id, auto &sum) {
      sum += group_results[id];
    });
  });

  auto third = q.submit([&](asycl::handler &chg) {
    chg.depends_on(second);
    chg.AdaptiveCpp_enqueue_custom_operation(
        [=](asycl::interop_handle &h) { asycl::free(group_results, q); });
  });
  return second; // maybe return second directly? as third is not needed and
                 // will called at the end in wait
}

template <typename DT> struct Matrix {

  DT *data;
  int *columns;
  int *rows;

  int NNZ;
  int N;
};

template <typename DT, enum Debug debuglevel = Debug::None> class CG {

public:
  // Initalize everything at the beginning
  CG(asycl::queue &queue) : _queue(queue) {
    b = nullptr;
    x = nullptr;
    p = nullptr;
    r = nullptr;
    rnext = nullptr;
  }

  void setMatrix(std::vector<DT> &_data, std::vector<int> &columns,
                 std::vector<int> &_rows) {

    const int NNZ = _data.size();
    const int N = _rows.size() - 1;

    auto m = std::max_element(columns.begin(), columns.end());

    this->A.NNZ = NNZ; // Count of
    this->A.N = N;
    this->N = A.N;

    this->A.data = asycl::malloc_device<DT>(NNZ, this->_queue);
    this->A.columns = asycl::malloc_device<int>(NNZ, this->_queue);

    this->A.rows = asycl::malloc_device<int>(_rows.size(), this->_queue); // N+1

    this->_queue.copy(_data.data(), this->A.data, NNZ);
    this->_queue.copy(columns.data(), this->A.columns, NNZ);
    this->_queue.copy(_rows.data(), this->A.rows, _rows.size());
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

    bool done = false;

    if (this->b == nullptr) {
      throw std::runtime_error("No right hand side to solve for");
    }

    r = asycl::malloc_device<DT>(A.N, _queue);
    rnext = asycl::malloc_device<DT>(A.N, _queue);
    p = asycl::malloc_device<DT>(A.N, _queue);

    // Helper buffers
    DT *helper = asycl::malloc_device<DT>(A.N, _queue);
    DT *rxr = asycl::malloc_device<DT>(1, _queue);
    DT *value2 = asycl::malloc_device<DT>(1, _queue);
    DT *value3 = asycl::malloc_device<DT>(1, _queue);
    DT *alpha = asycl::malloc_device<DT>(1, _queue);
    DT *beta = asycl::malloc_device<DT>(1, _queue);

    if (this->x == nullptr)
      this->x = asycl::malloc_device<DT>(A.N, this->_queue);
    _queue.fill(r, static_cast<DT>(0), A.N);
    _queue.fill(rnext, static_cast<DT>(0), A.N);
    _queue.fill(p, static_cast<DT>(0), A.N);
    _queue.fill(x, static_cast<DT>(0), A.N);

    _queue.wait(); // unvommented again

    if (!r || !rnext || !p || !helper || !x) {
      throw std::runtime_error("Failed to allocate memory");
    }

    int counter = 0;
    DT res;

    // COmpute r0 = b - Ax0
    auto initializer = _queue.submit([&](asycl::handler &chg) {
      auto rows = A.rows;
      auto columns = A.columns;
      auto data = A.data;
      auto x = this->x;
      auto r = this->r;
      auto rnext = this->rnext;
      auto b = this->b;
      auto p = this->p;

      chg.parallel_for<class MyFirstKernel>(this->A.N, [=](asycl::item<1> id) {
        DT single_result = 0;
        for (int j = rows[id]; j < rows[id + 1]; j++) { 
          single_result += data[j] * x[columns[j]];     
        }
        r[id] = b[id] - single_result;
        p[id] = r[id];
        rnext[id] = p[id];
      });
    });

    // Precompute <r,r>
    auto erxr = _queue.submit([&](asycl::handler &chg) {
      auto r = this->r;

      chg.depends_on(initializer);
      auto SumRed = asycl::reduction(rxr, asycl::plus<DT>());
      chg.parallel_for(A.N, SumRed, [=](asycl::item<1> id, auto &sum) {
        sum += r[id] * r[id];
      });
    });
    _queue.wait();


    do {

      // Make Clean enviromnet
      auto clean_helper = _queue.fill(helper, static_cast<DT>(0), A.N);

      auto reseter = _queue.submit([&](asycl::handler &chg) {
        chg.single_task<class CleanUp>([=]() {
          *value2 = 0;
          *value3 = 0;
          *alpha = 0;
          *beta = 0;
        });
      });

      //  p * A * p
      auto e1 = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(clean_helper);
        chg.depends_on(reseter);

        auto rows = this->A.rows;
        auto columns = this->A.columns;
        auto data = this->A.data;
        auto p = this->p;

        chg.parallel_for(this->A.N, [=](asycl::item<1> id) {
          DT single_result = 0;
          for (int j = rows[id]; j < rows[id + 1]; j++) {
            single_result += (data[j] * p[columns[j]]);
          }

          helper[id] = single_result;
        });
      });

      // std::cout << "e1\n";

      // scalarproduct of AP with p
      auto e3 = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(e1);

        asycl::local_accessor<DT, 1> local_reduction(TILE, chg);

        auto p = this->p;

        const auto N = this->A.N;
        auto SumRed = asycl::reduction(value2, asycl::plus<DT>());
        chg.parallel_for<class APwithp>(
            A.N, SumRed,
            [=](asycl::item<1> id, auto &sum) { sum += helper[id] * p[id]; });
      });

      // std::cout << "e3\n";
      //  r * r

      // Alpha = value / value2
      auto aAlpha = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(erxr);
        chg.depends_on(e3);
        chg.single_task<class AlphaCalculation>(
            [=]() { *alpha = *rxr / *value2; });
      });

      // Calculate x_k+1 and r_k+1

      auto exnext = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(aAlpha);

        auto p = this->p;
        auto x = this->x;
        chg.parallel_for<class XNext>(
            A.N, [=](asycl::item<1> id) { x[id] += *alpha * p[id]; });
      });

      // Calculate rnext

      auto ernext = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(aAlpha);
        chg.depends_on(e1);

        auto rnext = this->rnext;
        chg.parallel_for<class RNext>(
            A.N, [=](asycl::item<1> id) { rnext[id] -= *alpha * helper[id]; });
      });

      // Check size of rk
      /*
      auto eaccuracy = _queue.submit([&](asycl::handler& chg){
          chg.depends_on(ernext);

          // Circumvent value capture restriction
          bool* is_done = &done;

          chg.single_task([=](){
              if (asycl::sqrt(*rxr) <= error)
                  *is_done = true;
          });
      });

*/
      // Comment above out for testing
      // Calculate beta
      // Calculate r_k+1 x r_k+1
      auto e4 = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(ernext);

        auto rnext = this->rnext;
        auto SumRed = asycl::reduction(value3, asycl::plus<DT>());
        chg.parallel_for<class rk1withrk1>(A.N, SumRed,
                                           [=](asycl::item<1> id, auto &sum) {
                                             sum += rnext[id] * rnext[id];
                                           });
      });
      // auto e4 = dot_product(this->_queue, rnext, rnext, value3, A.N);

      auto aBeta = _queue.single_task<class BetaCalculation>({e4, erxr}, [=]() {
        *beta = *value3 / *rxr;
        *rxr = *value3;
      });

      auto epnext = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(aBeta);
        chg.depends_on(ernext);
        auto p = this->p;
        auto rnext = this->rnext;
        chg.parallel_for<class NextP>(
            A.N, [=](asycl::item<1> id) { p[id] = rnext[id] + *beta * p[id]; });
      });

      auto next_step = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(aBeta);
        chg.copy(rnext, r, A.N);
      });

      _queue.wait();

      // One Algorithm iteration

    } while (counter++ < A.N && !done);

    asycl::free(helper, _queue);
    asycl::free(rxr, _queue);
    asycl::free(value2, _queue);
    asycl::free(beta, _queue);
    asycl::free(alpha, _queue);
    asycl::free(value3, _queue);

    this->is_solved = true;
  }

  bool verify(DT tolerance = static_cast<DT>(0)) {

    DT *value = asycl::malloc_device<DT>(1, _queue);
    DT *helper = asycl::malloc_device<DT>(A.N, _queue);

    _queue.submit([&](asycl::handler &chg) {
      auto SumRed = asycl::reduction(value, asycl::plus<DT>());

      auto rows = this->A.rows;
      auto columns = this->A.columns;
      auto data = this->A.data;
      auto x = this->x;
      auto b = this->b;

      chg.parallel_for(A.N, SumRed, [=](asycl::item<1> id, auto &sum) {
        DT single_result = 0;
        for (int j = rows[id]; j < rows[id + 1]; j++) {
          single_result += data[j] * x[columns[j]];
        }

        auto a = b[id] - single_result;
        sum += a * a;
      });
    });

    _queue.wait();

    asycl::free(helper, _queue);

    DT abs_error = *value;
    asycl::free(value, _queue);

    return abs_error <= tolerance;
  }

  void verify2(std::vector<DT> &result) {}

  std::vector<DT> extract() {

    std::vector<DT> ret(A.N);
    _queue.copy(this->x, ret.data(), A.N).wait();

    return ret;
  }

  DT accuracy(DT tolerance = static_cast<DT>(0)) {
    DT *normres = asycl::malloc_device<DT>(1, _queue);
    DT *normx = asycl::malloc_device<DT>(1, _queue);
    DT *helper = asycl::malloc_device<DT>(A.N, _queue);

    _queue.submit([&](asycl::handler &chg) {
      auto SumRed = asycl::reduction(normres, asycl::plus<DT>());
      auto SumRed2 = asycl::reduction(normx, asycl::plus<DT>());

      auto rows = this->A.rows;
      auto columns = this->A.columns;
      auto data = this->A.data;
      auto x = this->x;
      auto b = this->b;

      chg.parallel_for(A.N, SumRed, SumRed2,
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

    asycl::free(helper, _queue);

    DT abs_error = std::abs(*normres / *normx);
    asycl::free(normres, _queue);
    asycl::free(normx, _queue);

    return abs_error;
  }
  void extractTo(std::vector<DT> &result) {}

  std::size_t memoryFootprint() const {
    return (2 * this->A.NNZ + (6 * this->A.N))
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
    if (r != nullptr)
      asycl::free(r, _queue);
    if (rnext != nullptr)
      asycl::free(rnext, _queue);
    if (p != nullptr)
      asycl::free(p, _queue);
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

  asycl::event MatrixVectorMultiplikation(std::shared_ptr<DT> vector,
                                          std::shared_ptr<DT> result);

  asycl::event Vectorscaling(std::shared_ptr<DT> vector, DT scalar,
                             std::shared_ptr<DT> result);

  asycl::queue _queue;

  bool is_solved = false;

  size_t max_workgroup_count;

  Matrix<DT> A;

  DT *x;
  DT *p;
  DT *r;
  DT *rnext;

  DT *b;

  size_t N;
};

}; // namespace CGSolver

#endif /*CG_HPP */

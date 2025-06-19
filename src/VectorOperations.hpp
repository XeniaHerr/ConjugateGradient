#ifndef VECTOROPERATIONS_HPP
#define VECTOROPERATIONS_HPP

#include <AdaptiveCpp/hipSYCL/sycl/usm.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>

#include <AdaptiveCpp/sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include "Matrix.hpp"
#include <memory>

namespace CGSolver {

namespace asycl = acpp::sycl;

/**
 * when count == 0, we assume that the vector_size has been set beforehand*/
template <class DT> class VectorOperations {

private:
  // vector_size == to_reduce_size, workgroupcount == to_reduce_to
  asycl::event reduction_step(DT *to_reduce, DT *to_reduce_to,
                              size_t to_reduce_size, size_t to_reduce_to_size,
                              std::vector<asycl::event> events = {}) {

    auto event = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(events);

      asycl::local_accessor<DT, 1> local_results(workgroupsize, chg);
      chg.parallel_for<class FirstRoundReduction>(
          asycl::nd_range<1>(asycl::range<1>(to_reduce_to_size * workgroupsize),
                             asycl::range<1>(workgroupsize)),
          [=](asycl::nd_item<1> item) {
            auto local_id = item.get_local_id(0);
            auto global_id = item.get_global_id(0);
            auto group = item.get_group(0);

            DT val = (global_id < to_reduce_size) ? to_reduce[global_id] : 0;

            if (global_id < to_reduce_size) {
              val = to_reduce[global_id];
            } else {
              val = static_cast<DT>(0);
            }
            local_results[local_id] = val;

            item.barrier(asycl::access::fence_space::local_space);

            for (size_t offset = workgroupsize / 2; offset > 0; offset /= 2) {
              if (local_id < offset)
                local_results[local_id] += local_results[local_id + offset];
              item.barrier(asycl::access::fence_space::local_space);
            }

            if (local_id == 0) {
              to_reduce_to[group] = local_results[0];
            }
          });
    });

    return event;
  }

public:
  VectorOperations(asycl::queue q) : _queue(q) {
    workgroupsize = calculateWorkgroupSize();
    group_results = nullptr;
  }
  void setVectorSize(size_t size) { this->vector_size = size; }

  asycl::event
  dot_product_optimised(Vector<DT> &Left, Vector<DT> &Right, DT *result, size_t count = 0,
                        std::vector<asycl::event> dependencies = {}) {

    vector_size = count == 0 ? vector_size : count;

    assert(vector_size != 0);

    auto workgroupcount = (vector_size + workgroupsize - 1) / workgroupsize;

    // if (group_results == nullptr)
    DT *group_results = asycl::malloc_device<DT>(workgroupcount, _queue);

    auto filler =
        _queue.fill(group_results, static_cast<DT>(0), workgroupcount);

    auto event = _queue.submit([&](asycl::handler &chg) {
      const auto vector_size = this->vector_size;

      chg.depends_on(filler);
      chg.depends_on(dependencies);

      auto left = Left.ptr();
      auto right = Right.ptr();

      asycl::local_accessor<DT, 1> local_results(workgroupsize, chg);
      chg.parallel_for<class FirstRoundReduction>(
          asycl::nd_range<1>(asycl::range<1>(workgroupcount * workgroupsize),
                             asycl::range<1>(workgroupsize)),
          [=](asycl::nd_item<1> item) {
            auto local_id = item.get_local_id(0);
            auto global_id = item.get_global_id(0);
            auto group = item.get_group(0);

            DT val;

            if (global_id < vector_size) {
              val = left[global_id] * right[global_id];
            } else {
              val = static_cast<DT>(0);
            }
            local_results[local_id] = val;

            item.barrier(asycl::access::fence_space::local_space);

            for (size_t offset = workgroupsize / 2; offset > 0; offset /= 2) {
              if (local_id < offset)
                local_results[local_id] += local_results[local_id + offset];
              item.barrier(asycl::access::fence_space::local_space);
            }

            if (local_id == 0) {
              group_results[group] = local_results[0];
            }
          });
    });
    // Initial reduction step done

    asycl::event step_event = filler;

    while (workgroupcount >= workgroupsize * workgroupsize) {

      const auto next_workgroupcount =
          (workgroupcount + workgroupsize - 1) / workgroupsize;

      DT *step_buffer =
          asycl::malloc_device<DT>(next_workgroupcount, this->_queue);

      step_event =
          this->reduction_step(group_results, step_buffer, workgroupcount,
                               next_workgroupcount, {step_event});
      asycl::free(group_results, this->_queue);
      group_results = step_buffer;
      asycl::free(step_buffer, this->_queue);

      workgroupcount = next_workgroupcount;
    }
    auto final_reduction = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(event);

      auto Reducer = asycl::reduction(result, asycl::plus<DT>());

      chg.parallel_for<class FinalReduction>(
          workgroupcount, Reducer,
          [=](asycl::item<1> item, auto &sum) { sum += group_results[item]; });
    });

    auto cleanup = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(final_reduction);

      chg.AdaptiveCpp_enqueue_custom_operation(
          [=](auto f) { asycl::free(group_results, _queue); });
    });

    return final_reduction;
  }

  //  template <class DT>
  asycl::event dot_product(DT *left, DT *right, DT *result, size_t count = 0,
                           std::vector<asycl::event> dependencies = {}) {

    vector_size = count == 0 ? vector_size : count;

    assert(vector_size != 0);

    const auto workgroupcount =
        (vector_size + workgroupsize - 1) / workgroupsize;

    // if (group_results == nullptr)
    DT *group_results = asycl::malloc_device<DT>(workgroupcount, _queue);

    auto filler =
        _queue.fill(group_results, static_cast<DT>(0), workgroupcount);

    auto event = _queue.submit([&](asycl::handler &chg) {
      const auto vector_size = this->vector_size;

      chg.depends_on(filler);
      chg.depends_on(dependencies);

      asycl::local_accessor<DT, 1> local_results(workgroupsize, chg);
      chg.parallel_for<class FirstRoundReduction>(
          asycl::nd_range<1>(asycl::range<1>(workgroupcount * workgroupsize),
                             asycl::range<1>(workgroupsize)),
          [=](asycl::nd_item<1> item) {
            auto local_id = item.get_local_id(0);
            auto global_id = item.get_global_id(0);
            auto group = item.get_group(0);

            DT val;

            if (global_id < vector_size) {
              val = left[global_id] * right[global_id];
            } else {
              val = static_cast<DT>(0);
            }
            local_results[local_id] = val;

            item.barrier(asycl::access::fence_space::local_space);

            for (size_t offset = workgroupsize / 2; offset > 0; offset /= 2) {
              if (local_id < offset)
                local_results[local_id] += local_results[local_id + offset];
              item.barrier(asycl::access::fence_space::local_space);
            }

            if (local_id == 0) {
              group_results[group] = local_results[0];
            }
          });
    });

    auto final_reduction = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(event);

      auto Reducer = asycl::reduction(result, asycl::plus<DT>());

      chg.parallel_for<class FinalReduction>(
          workgroupcount, Reducer,
          [=](asycl::item<1> item, auto &sum) { sum += group_results[item]; });
    });

    auto cleanup = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(final_reduction);

      chg.AdaptiveCpp_enqueue_custom_operation(
          [=](auto f) { asycl::free(group_results, _queue); });
    });

    return final_reduction;
  }
  inline asycl::event saxpby(Vector<DT> &X, Vector<DT> &Y, DT *a, DT* b, Vector<DT> &Result,
                             size_t count = 0,
                             std::vector<asycl::event> events = {}) {

    auto event = _queue.submit([&](asycl::handler &chg) {
      auto x = X.ptr();
      auto y = Y.ptr();
      auto result = Result.ptr();
        chg.depends_on(events);

      chg.parallel_for<class saxpbyKernel>(vector_size, [=](asycl::item<1> id) {
        result[id] = *a * x[id] + *b * y[id];
      });
    });

    return event;
  }

  /// result = x - by
  inline asycl::event sambx(Vector<DT> &X, Vector<DT> &Y, DT *b, Vector<DT>& Result, size_t count = 0,
                            std::vector<asycl::event> events = {}) {

    auto event = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(events);

      auto x = X.ptr();
      auto y = Y.ptr();
      auto result = Result.ptr();
      chg.parallel_for<class sambxKernel>(vector_size, [=](asycl::item<1> id) {
        result[id] = x[id] - *b * y[id];
      });
    });

    return event;
  }

  /// result = x + by
  inline asycl::event sapbx(Vector<DT> &X, Vector<DT> &Y, DT *b, Vector<DT> &Result, size_t count = 0,
                            std::vector<asycl::event> events = {}) {

    auto event = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(events);

      auto x = X.ptr();
      auto y = Y.ptr();
      auto result = Result.ptr();

      chg.parallel_for<class saxpbyKernel>(vector_size, [=](asycl::item<1> id) {
        result[id] = x[id] + *b * y[id];
      });
    });

    return event;
  }

  inline asycl::event spmv(Matrix<DT>& A, Vector<DT>& vec, Vector<DT> &Result,
                    size_t NNZ, size_t count = 0,
                    std::vector<asycl::event> events = {}) {

    vector_size = count == 0 ? vector_size : count;

    assert(vector_size != 0);

    auto evector_matrix_product = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(events);

      auto rows = A.rows().get();
      auto columns = A.columns().get();
      auto data = A.data().get();
      auto vector = vec.ptr();
      auto result = Result.ptr();

      chg.parallel_for(vector_size, [=](asycl::item<1> id) {
        DT single_result = 0;
        for (int j = rows[id]; j < rows[id + 1]; j++) {
          single_result += (data[j] * vector[columns[j]]);
        }

        result[id] = single_result;
      });
    });

    return evector_matrix_product;
  }

  ~VectorOperations() {}

private:
  size_t workgroupsize;
  asycl::queue _queue;
  size_t vector_size;

  DT *group_results;

  // TODO: Implement
  size_t calculateWorkgroupSize() {
    auto device = _queue.get_device();

    const auto max_wg_size =
        device.get_info<asycl::info::device::max_work_group_size>();
    std::cout << "work group size is " << max_wg_size << std::endl;
    return std::min(static_cast<size_t>(128), max_wg_size);
  }
};
} // namespace CGSolver

#endif /*VECTOROPERATIONS_HPP*/

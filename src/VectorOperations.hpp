#ifndef VECTOROPERATIONS_HPP
#define VECTOROPERATIONS_HPP

/**
 * @file VectorOperations.hpp
 *
 * This file contains the VectorOperations class, which Provides the Kernels for
 * the needed Linear Algebra Operations
 */

#include "LinearAlgebraTypes.hpp"
#include <AdaptiveCpp/hipSYCL/sycl/usm.hpp>
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

namespace CGSolver {

namespace asycl = acpp::sycl;

/**
 * @class VectorOperations
 *
 * @brief Collection of Kernels for Linear Algebra Opearations.
 *
 * All methods that implement Kernels return a sycl event. This event can be
 * used in Future Kernels to create a Dependencie Hierarchy. This class doesn't
 * call queue.wait() on its own.*/
template <class DT> class VectorOperations {

private:
  /**
   * @brief Internal method for a single Reduction step*/
  asycl::event reduction_step(DT *to_reduce, std::size_t begin_buffer,
                              size_t to_reduce_size, size_t to_reduce_to_size,
                              std::vector<asycl::event> events = {}) {

    auto event = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(events);

      asycl::local_accessor<DT, 1> local_results(workgroupsize, chg);
      chg.parallel_for<class ReductionStep>(
          asycl::nd_range<1>(asycl::range<1>(to_reduce_to_size * workgroupsize),
                             asycl::range<1>(workgroupsize)),
          [=](asycl::nd_item<1> item) {
            auto local_id = item.get_local_id(0);
            auto global_id = item.get_global_id(0);
            auto group = item.get_group(0);

            DT val = (global_id < to_reduce_size)
                         ? to_reduce[begin_buffer + global_id]
                         : 0;

            if ((global_id + begin_buffer) < to_reduce_size) {
              val = to_reduce[global_id + begin_buffer];
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
              to_reduce[group + begin_buffer + to_reduce_size] =
                  local_results[0];
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

  /**
   * @brief Set the Lenght of the used Vectors once*/
  void setVectorSize(size_t size) { this->vector_size = size; }

  /**
   * @brief Calculate the Standart Scalarproduct of 2 Vectors using Parallel
   * Tree Reductions.
   *
   * @param Left Left Vector
   * @param Right Right Vector
   * @param result Pointer to store the Resulting Scalar.
   * @param dependencies std::vector of asycl::events that should be performed
   * before this kernel
   * @param Vectorsize optional size of the vector*/
  asycl::event
  dot_product_optimised(Vector<DT> &Left, Vector<DT> &Right, DT *result,
                        std::vector<asycl::event> dependencies = {},
                        size_t count = 0) {

    vector_size = count == 0 ? vector_size : count;

    assert(vector_size != 0);

    auto workgroupcount = (vector_size + workgroupsize - 1) / workgroupsize;

    // if (group_results == nullptr)
    DT *group_results = asycl::malloc_device<DT>(workgroupcount * 2, _queue);

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
    auto buffer_offset = workgroupcount;

    while (workgroupcount >= workgroupsize * workgroupsize) {

      const auto next_workgroupcount =
          (workgroupcount + workgroupsize - 1) / workgroupsize;

      // DT *step_buffer =
      //     asycl::malloc_device<DT>(next_workgroupcount, this->_queue);

      step_event =
          this->reduction_step(group_results, buffer_offset, workgroupcount,
                               next_workgroupcount, {step_event});
      // asycl::free(group_results, this->_queue);
      // group_results = step_buffer;
      // asycl::free(step_buffer, this->_queue);

      workgroupcount = next_workgroupcount;
      buffer_offset += workgroupcount;
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

  /**
   @deprecated Use the optimised version*/
  asycl::event dot_product(DT *left, DT *right, DT *result,
                           std::vector<asycl::event> dependencies = {},
                           size_t vec_size = 0) {

    vector_size = vec_size == 0 ? vector_size : vec_size;

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



  /**
   * @brief Add two scaled Vectors
   *
   * @param X Vector
   * @param Y Vector
   * @param a scalar for X
   * @param b scalar for Y
   * @param Result resulting vector
   * @param events dependencies
   * @param count Vectorsize
   */
  inline asycl::event saxpby(Vector<DT> &X, Vector<DT> &Y, DT *a, DT *b,
                             Vector<DT> &Result,
                             std::vector<asycl::event> events = {},
                             size_t vec_size = 0) {

    vector_size = vec_size == 0 ? vector_size : vec_size;
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
  /**
   * @brief Subtract a scaled vector from another vector
   *
   * @param X Vector to subtract from
   * @param Y Vector to be subtracted
   * @param b scalar for Y
   * @param Result resulting vector
   * @param events dependencies
   * @param count Vectorsize
   */
  inline asycl::event sambx(Vector<DT> &X, Vector<DT> &Y, DT *b,
                            Vector<DT> &Result,
                            std::vector<asycl::event> events = {},
                            size_t count = 0) {

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
  /**
   * @brief Add a scaled vector to another vector
   *
   * @param X Vector 
   * @param Y Vector 
   * @param b scalar for Y
   * @param Result resulting vector
   * @param events dependencies
   * @param count Vectorsize
   */
  inline asycl::event sapbx(Vector<DT> &X, Vector<DT> &Y, DT *b,
                            Vector<DT> &Result,
                            std::vector<asycl::event> events = {},
                            size_t count = 0) {

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

  /**
   * @brief Sparse Matrix Vector Multiplikation
   *
   * @param A Matrix
   * @param vec Vector
   * @param Result Result Vector
   * @param events dependencies
   * @param vec_size Size of Vectors*/
  inline asycl::event spmv(Matrix<DT> &A, Vector<DT> &vec, Vector<DT> &Result,
                           size_t NNZ, std::vector<asycl::event> events = {},
                           size_t count = 0) {

    vector_size = count == 0 ? vector_size : count;

    assert(vector_size != 0 && A.N() == vector_size);

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

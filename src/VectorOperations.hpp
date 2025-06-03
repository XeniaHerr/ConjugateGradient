#ifndef VECTOROPERATIONS_HPP
#define VECTOROPERATIONS_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>

#include <AdaptiveCpp/sycl/sycl.hpp>
#include <iostream>

namespace asycl = acpp::sycl;

/**
 * when count == 0, we assume that the vector_size has been set beforehand*/
template <class DT> class VectorOperations {

public:
  VectorOperations(asycl::queue q) : _queue(q) {
    workgroupsize = calculateWorkgroupSize();
    group_results = nullptr;
  }
  void setVectorSize(size_t size) { this->vector_size = size; }

  //  template <class DT>
  asycl::event dot_product(DT *left, DT *right, DT *result, size_t count = 0,
                           std::vector<asycl::event> dependencies = {}) {

    vector_size = count == 0 ? vector_size : count;

    assert(vector_size != 0);

    const auto workgroupcount =
        (vector_size + workgroupsize - 1) / workgroupsize;

    //if (group_results == nullptr)
     DT * group_results = asycl::malloc_device<DT>(workgroupcount, _queue);

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


    auto cleanup = _queue.submit([&](asycl::handler& chg){

      chg.depends_on(final_reduction);

      chg.AdaptiveCpp_enqueue_custom_operation([=](auto f) {
	asycl::free(group_results, _queue);

	
      });
    });

    return final_reduction;
  }
  inline asycl::event saxpby(DT *x, DT *y, DT *a, DT *b, DT *result,
                             size_t count = 0,
                             std::vector<asycl::event> events = {}) {

    auto event = _queue.submit([&](asycl::handler &chg) {
      for (auto &event : events)
        chg.depends_on(event);

      chg.parallel_for<class saxpbyKernel>(vector_size, [=](asycl::item<1> id) {
        result[id] = *a * x[id] + *b * y[id];
      });
    });

    return event;
  }

  /// result = x - by
  inline asycl::event sambx(DT *x, DT *y, DT *b, DT *result, size_t count = 0,
                            std::vector<asycl::event> events = {}) {

    auto event = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(events);

      chg.parallel_for<class saxpbyKernel>(vector_size, [=](asycl::item<1> id) {
        result[id] = x[id] - *b * y[id];
      });
    });

    return event;
  }

  /// result = x + by
  inline asycl::event sapbx(DT *x, DT *y, DT *b, DT *result, size_t count = 0,
                            std::vector<asycl::event> events = {}) {

    auto event = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(events);

      chg.parallel_for<class saxpbyKernel>(vector_size, [=](asycl::item<1> id) {
        result[id] = x[id] + *b * y[id];
      });
    });

    return event;
  }

  asycl::event spmv(DT *data, int *columns, int *rows, DT *vector, DT *result,
                    size_t NNZ, size_t count = 0,
                    std::vector<asycl::event> events = {}) {

    vector_size = count == 0 ? vector_size : count;

    assert(vector_size != 0);

    auto evector_matrix_product = _queue.submit([&](asycl::handler &chg) {
      chg.depends_on(events);

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

  ~VectorOperations() {
    //if (group_results != nullptr)
    //asycl::free(group_results, _queue);
  }

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

#endif /*VECTOROPERATIONS_HPP*/

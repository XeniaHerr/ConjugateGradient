#ifndef CG_HPP
#define CG_HPP

/**
 * @file CG.hpp
 *
 * Main Class CG
 *
 * This file conatins the Class CG, which contains the main algorithm. To use
 * this class the user has to provide a sycl::queue object and both parts of the
 * LGS as std::vectors*/

// IMPORTANT: Only include sycl.hpp header. All other headers might lead to
// compilation errors
#include <AdaptiveCpp/sycl/handler.hpp>
#include <AdaptiveCpp/sycl/libkernel/builtins.hpp>
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <hipSYCL/sycl/exception.hpp>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "LinearAlgebraTypes.hpp"
#include "VectorOperations.hpp"

namespace CGSolver {

namespace asycl = acpp::sycl;

/**
 * @enum Debuglevel
 * @brief Specify how verbose the Output is*/
enum Debuglevel {
  None,    ///< No output
  Verbose, ///< All possible outputs

};

  /**
   * @class CG
   *
   * @brief Set the parameters of the Algorithm and execute it.
   *
   * Set the Matrix, the Right hand side and the inital Value of the LGS and solve it with a given precision. Extract the Result afterwards to Host memory.*/
template <typename DT, Debuglevel debug = Debuglevel::None> class CG {

public:
  using Scalar = Scalar<DT>;
  using Matrix = Matrix<DT>;
  using Vector = Vector<DT>;
  // Do not allocate Memory at the beginning
  CG(asycl::queue &queue) : _queue(queue), A(queue), x(_queue), b(_queue) {

    if constexpr (debug == Debuglevel::Verbose)
      std::cout << "Constructing CG Object\n";
  }

  /**
   * @brief create Dynamically Allocated CG object without having to expose the queue to the user*/
  static  std::unique_ptr<CG> createCG() {

    asycl::queue q;

    std::unique_ptr<CG> cg(new CG(q));

    return cg;
  }
  /**
   * @brief Set Matrix of the LGS in CSR Format
   *
   * Must be called before @see solve or an exception will be thrown
   *
   * @param data std::vector of Matrix values
   * @param columns std::vector of Column coordinates of data
   * @param rows std::vector of beginning of new rows*/
  void setMatrix(std::vector<DT> &data, std::vector<int> &columns,
                 std::vector<int> &rows) {
    if constexpr (debug == Debuglevel::Verbose)
      std::cout << "Setting Matrix\n";

    A.init(data, columns, rows);
  }

  /**
   * @brief Set Matrix of the LGS in CSR Format
   *
   * Must be called before solve or an exception will be thrown
   *
   * @param M Matrix stored on device. Will be moved
   */
  void setMatrix(Matrix &&M) { A = std::forward<Matrix>(M); }

  /**
   * @return returns the Dimension of the Matrix
   */
  int getDimension() const { return this->A.N(); }

  /**
   * @brief set the Right hand side of the LGS
   *
   * Must be called before solve or an exception will be thrown
   *
   * @param _data std::vector of the righ hand side of the LGS*/
  void setTarget(std::vector<DT> &_data) {

    b.init(_data);

    _queue.wait();
  }

  /**
   * @brief Set the Right hand size of the LGS
   *
   * Must be called before solve or an exception will be thrown
   *
   * @param V Vector stored on device. Will be moved
   */
  void setTarget(Vector &&V) { b = std::forward<Vector>(V); }

  /**
   * @brief set initial guess for x
   *
   * If not set, x will be the 0 vector
   *
   * @param _data std::vector containing the initial guess
   */
  void setInital(std::vector<DT> &_data) {
    x.init(_data);
    _queue.wait();
  }



  void calculateExpectedStepCount(DT accuracy) {


    
  }

  /**
   * @brief Set the Initial Guess
   *
   * Must be called before solve or an exception will be thrown
   *
   * @param V Vector stored on device. Will be moved
   */
  void setInitial(Vector &&V) { x = std::forward<Vector>(V); }

  /**
   * @brief Solve the given LGS
   *
   * Perform Conjugate Gradient on with the provided Matrix and right hand side.
   *
   * @param improvement If provided terminate the algorithm after the
   * improvement rate is less that the argument
   *
   * @throw runtime_error Matrix or Right hand side of LGS is missing*/
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

    Vector helper(_queue, N);
    Vector r(_queue, N);
    Vector rnext(_queue, N);
    Vector p(_queue, N);
    Scalar rxr(_queue);
    Scalar value2(_queue);
    Scalar value3(_queue);
    Scalar alpha(_queue);
    Scalar beta(_queue);
    Scalar r0(_queue);
    Scalar acc(_queue, improvement);

    // Must be available to both host and device
    bool *is_done = asycl::malloc_shared<bool>(1, _queue);

    if (x.ptr() == nullptr)
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
    auto erxr = vecops.dot_product_optimised(r, r, rxr, {initializer}, 0);
    _queue.wait();

    auto er0 = _queue.submit([&] (asycl::handler& chg) {

      chg.depends_on({erxr});
      auto r0p = r0.ptr();
      auto rxrp = rxr.ptr();
      auto accp = acc.ptr();
      chg.single_task([=](){
	*r0p = asycl::sqrt( *rxrp) * *accp;
      });
    });

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
          vecops.spmv(A, p, helper, A.NNZ(), {eclean_helper, eresetter});

      // scalarproduct of AP with p
      auto eapxp = vecops.dot_product_optimised(helper, p, value2,
                                                {evector_matrix_product});

      // Alpha = rxr/ value2
      auto ealpha = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(erxr);
        chg.depends_on(eapxp);
        chg.single_task<class AlphaCalculation>(
            [=]() { *alpha = *rxr / *value2; });
      });

      // Calculate x_k+1 and r_k+1
      auto exnext = vecops.sapbx(x, p, alpha, x, {ealpha});

      auto ernext = vecops.sambx(rnext, helper, alpha, rnext,
                                 {ealpha, evector_matrix_product});

      // Check size of rk
      auto eaccuracy = _queue.submit([&](asycl::handler &chg) {
	auto accp = r0.ptr();
        chg.depends_on({ernext, er0});

        chg.single_task([=]() {
          if (asycl::sqrt(*rxr) <= *accp)
            *is_done = true;
        });
      });

      auto ernextxrnext =
          vecops.dot_product_optimised(rnext, rnext, value3, {ernext});

      // Calculate beta
      auto ebeta = _queue.single_task<class BetaCalculation>(
          {ernextxrnext, erxr}, [=]() {
            *beta = *value3 / *rxr;
            *rxr = *value3;
          });
      auto epnext = vecops.sapbx(rnext, p, beta, p, {ebeta, ernext});

      auto enext_step = _queue.submit([&](asycl::handler &chg) {
        chg.depends_on(ebeta);
        chg.copy(rnext.ptr(), r.ptr(), N);
      });

      this->executeQueue();
      //      _queue.wait();

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

    this->is_solved = true;

    if constexpr( debug == Debuglevel::Verbose) {
      std::cout << std::endl;
    }
  }

  /**
   * @brief calculate the error
   *
   * Performs a Matrix Vector Multiplikation with the calculated x and
   * calculates the distance to b
   *
   * @return Norm of the Distance*/
  DT accuracy() {

    if constexpr (debug == Debuglevel::Verbose)
      std::cout << "Calculating accuracy" << std::endl;
    Scalar normres(_queue);
    Scalar normx(_queue);

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

  /**
   * @brief Copy the result into the provided Vector
   * @param result Vector the result will be copied into. Will be resized
   */
  void extractTo(std::vector<DT> &result) {
    result.resize(A.N());
    this->_queue.copy(this->x.ptr(), result.data(), A.N()).wait();
  }

  /**
   * @brief estimate the required Memory on the device
   * @returns Number of bytes as std::size_t
   */
  std::size_t memoryFootprint() const {
    return (2 * this->A.NNZ() + (4 * this->A.N())) * sizeof(DT) +
           (2 * this->A.N() * sizeof(int));
  }

private:
  void executeQueue() {

    try {
      this->_queue.wait_and_throw();
    } catch (asycl::exception &e) {

      throw e;

    }

    catch (std::exception &e) {

      std::cerr << "Caught Exception " << e.what() << std::endl;

      throw e;
    }
  }
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

  Matrix A;

  Vector x;

  Vector b;

  size_t N;
};

}; // namespace CGSolver

#endif /*CG_HPP */

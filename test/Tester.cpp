#include "CG.hpp"
#include "utils.hpp"
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <ostream>
#include <string>

using namespace acpp::sycl;
using namespace CGSolver;

using DataType = double;

int main(int argc, char *argv[]) {

  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " [filename]" << std::endl;
    std::cout << "Returns Matrixdimensions\tNumberNonZero\tTime in ms\tRelative error"
              << std::endl;
    return 1;
  }

  std::string testfile = argv[1];

  auto [data, cols, rows] = read_file(testfile);

  auto NNZ = data.size();

  std::vector<DataType> target(rows.size() - 1);

  for (int i = 0; i < target.size(); i++)
    target[i] = i + 1;

  queue q;

  CG<DataType, CGSolver::Debuglevel::Verbose> cg(q);

  cg.setMatrix(data, cols, rows);

  cg.setTarget(target);

  Timer t;
  t.start_measure();
  cg.solve(1e-23);
  t.stop_measure();
  auto elapsed = t.get_duration();

  auto result = cg.extract();

  auto dim = cg.getDimension();

  auto correct = cg.accuracy();
  std::cout << dim << " " << NNZ << " " << elapsed.count() << " " << correct << std::endl;

  return 0;
};

#include "CG.hpp"
#include "utils.hpp"
#include <AdaptiveCpp/sycl/sycl.hpp>
#include <iomanip>
#include <string>

using namespace acpp::sycl;
using namespace CGSolver;

using DataType = double;

int main(int argc, char *argv[]) {

  std::string testfile = argv[1];


  auto [data, cols, rows] = read_file(testfile);

  std::vector<DataType> target(rows.size() - 1);


  for (int i = 0; i < target.size(); i++)
    target[i] = i + 1;

  queue q;

  CG<DataType> cg(q);

  cg.setMatrix(data, cols, rows);


  cg.setTarget(target);

  Timer t;
  t.start_measure();
  cg.solve(1e-23);
  t.stop_measure();
  auto elapsed = t.get_duration();


  auto result = cg.extract();

  auto dim = cg.getDimension();

  auto correct = cg.accuracy(10);
  std::cout << dim << " " << elapsed.count() << " " << correct << std::endl;

  return 0;
};

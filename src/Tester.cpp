#include "utils.hpp"
#include "CG.hpp"
#include <iomanip>
#include <string>
#include <AdaptiveCpp/sycl/sycl.hpp>




using namespace acpp::sycl;
using namespace CGSolver;


using DataType = double;

int main(int argc, char* argv[]) {

    std::string testfile = argv[1];


//    std::cout << "Testing with Matrix in " << std::quoted(testfile);


    auto [data, cols, rows] = read_file(testfile);



    std::vector<DataType> target(rows.size()-1);

    //std::vector<DataType> initial(data.size(), 0);


    for( int i = 0; i < target.size(); i++)
	target[i] = i+1;


 //   std::cout << "All Test data is ready" << std::endl;
    //queue q(property::queue::in_order{});
    queue q;

    CG<DataType> cg(q);

    cg.setMatrix(data, cols, rows);

    //Vector<DataType> init(q, initial);

    //cg.setInitial(std::move(init));

    cg.setTarget(target);


  //  std::cout << "CG Object has been initialized" << std::endl;
    Timer t;
    t.start_measure();
    cg.solve();
    t.stop_measure();
    auto elapsed = t.get_duration();

    //std::cout << elapsed.count() << std::endl;

    auto result = cg.extract();

//    for (auto& a: result)
	//std::cout << a << " ";
    //std::cout << std::endl;
    auto dim = cg.getDimension();

    auto correct = cg.accuracy(10);
    std::cout << dim << " " << elapsed.count() << " " << correct << std::endl;

    








    return 0;

};



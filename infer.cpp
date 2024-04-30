#include "infer.h"
TRTEngineSingleton* TRTEngineSingleton::instance = nullptr;
int main() {
    // Example usage
    TRTEngineSingleton* trt=TRTEngineSingleton::getInstance();
    std::vector<float> out =  trt->infer("/home/dms/DMS/tensorrt/models downloaded/test.jpg");
    for (float x : out){
	printf("%f \n",x);
	}
    std::vector<float> out2 =  trt->infer("/home/dms/DMS/tensorrt/models downloaded/test.jpg");
    for (float x : out2){
	printf("%f \n",x);
	}
    return 0;
}

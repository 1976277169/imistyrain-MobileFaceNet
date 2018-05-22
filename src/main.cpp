#include <iostream>
#include <time.h>
#include "net.h"
#include "mobilefacenet.h"
#include "mropencv.h"
#pragma comment(lib,"ncnn.lib")

std::vector<float> getFaceFeature(MobileNetFeatureExtractor *pfe,const std::string filepath)
{
    std::vector<float> feature;
    cv::Mat img = cv::imread(filepath);
    pfe->getFeature(img, feature);
    return feature;
}

int test2images(MobileNetFeatureExtractor *pfe,const std::string filepath1,const std::string filepath2)
{
    cv::TickMeter tm;
    tm.start();
    std::vector<float> feature1;
    std::vector<float> feature2;
#pragma omp parallel sections
    {
#pragma omp section
        {
            cv::Mat img1 = cv::imread(filepath1);
            pfe->getFeature(img1, feature1);
        }
#pragma omp section
        {
            cv::Mat img2 = cv::imread(filepath2);
            pfe->getFeature(img2, feature2);
        }
    }
    tm.stop();
    std::cout << tm.getTimeMilli()<<"ms"<< std::endl;
    double similarity = calculSimilar(feature1, feature2);
    std::cout << similarity << std::endl;
    return 0;
}

int main()
{
    char *model_path = "../ncnn";
    MobileNetFeatureExtractor *pfe = new MobileNetFeatureExtractor(model_path);
    const std::string filepath1="../images/Aaron_Tippin_0001.jpg";
    //const std::string filepath1 = "../images/Aaron_Peirsol_0003.jpg";
    const std::string filepath2 = "../images/Aaron_Peirsol_0004.jpg";
    test2images(pfe, filepath1, filepath2);
	return 0;
}
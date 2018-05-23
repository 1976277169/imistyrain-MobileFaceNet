#include "mtcnn.h"
#include "mobilefacenet.h"
#include "mropencv.h"
#pragma comment(lib,"ncnn.lib")

#define MAXFACEOPEN 0 //设置是否开关最大人脸调试，1为开，其它为关

cv::Mat drawDetection(const cv::Mat &img, std::vector<Bbox> &box)
{
    cv::Mat show = img.clone();
    const int num_box = box.size();
    std::vector<cv::Rect> bbox;
    bbox.resize(num_box);
    for (int i = 0; i < num_box; i++) {
        bbox[i] = cv::Rect(box[i].x1, box[i].y1, box[i].x2 - box[i].x1 + 1,box[i].y2 - box[i].y1 + 1);

        for (int j = 0; j < 5; j = j + 1)
        {
            cv::circle(show, cvPoint(box[i].ppoint[j], box[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
        }
    }
    for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
        rectangle(show, (*it), Scalar(0, 0, 255), 2, 8, 0);
    }
    return show;
}

void test_detection(const int cameraindex=0)
{
	char *model_path = "../ncnn";
	MTCNN mtcnn;
    mtcnn.init(model_path);
	mtcnn.SetMinFace(40);
	cv::VideoCapture mVideoCapture(cameraindex);
	if (!mVideoCapture.isOpened()) {
		return;
	}
	cv::Mat frame;
	mVideoCapture >> frame;
	while (!frame.empty()) {
		mVideoCapture >> frame;
		if (frame.empty()) {
			break;
		}
		clock_t start_time = clock();	
		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
		std::vector<Bbox> finalBbox;
#if(MAXFACEOPEN==1)
		mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
		mtcnn.detect(ncnn_img, finalBbox);
#endif
        cv::Mat show = drawDetection(frame, finalBbox);
		
		clock_t finish_time = clock();
		double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
		std::cout << "cost " << total_time * 1000 << "ms" << std::endl;
        cv::imshow("img", show);
        cv::waitKey(1);
    }
	return ;
}

std::vector<float> getFaceFeature(MobileNetFeatureExtractor *pfe, const std::string filepath)
{
    std::vector<float> feature;
    cv::Mat img = cv::imread(filepath);
    pfe->getFeature(img, feature);
    return feature;
}

int test2images(MobileNetFeatureExtractor *pfe, const std::string filepath1, const std::string filepath2)
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
    std::cout << tm.getTimeMilli() << "ms" << std::endl;
    double similarity = calculSimilar(feature1, feature2);
    std::cout << similarity << std::endl;
    return 0;
}

int testvalidation()
{
    char *model_path = "../models";
    MobileNetFeatureExtractor *pfe = new MobileNetFeatureExtractor(model_path);
    const std::string filepath1 = "../images/Aaron_Tippin_0001.jpg";
    //const std::string filepath1 = "../images/Aaron_Peirsol_0003.jpg";
    const std::string filepath2 = "../images/Aaron_Peirsol_0004.jpg";
    test2images(pfe, filepath1, filepath2);
    return 0;
}

int main(int argc, char** argv)
{	
	//test_detection();
    testvalidation();
	return 0;
}
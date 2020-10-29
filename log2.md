## Log:

* Created a cpp file for inference.

* **CPP file**:

```cpp
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace std;

/*
* ===  Class  ======================================================================
*         Name:  Detector
*  Description:  SSD CXX Detector
* =====================================================================================
*/

class Detector {
public:
	Detector(const string& model_file, const string& weights_file);
	void Detect(string im_name);
	void vis_detections(cv::Mat image, vector<vector<float>> pred_boxes, vector<float> confidence, vector<int> labels , int h, int w);

private:
	boost::shared_ptr<Net<float> > net_;
	Detector() {}
};

using namespace caffe;
using namespace std;

/*
* ===  FUNCTION  ======================================================================
*         Name:  Detector
*  Description:  Load the model file and weights file
* =====================================================================================
*/
//load modelfile and weights
Detector::Detector(const string& model_file, const string& weights_file)
{
	net_ = boost::shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(weights_file);
}

//perform detection operation
//input image max size 1000*600
void Detector::Detect(string im_name)
{

    cv::Mat result;
	cv::Mat cv_image = cv::imread(im_name);
	cv::cvtColor(cv_image, result, cv::COLOR_BGR2RGB);
	result.convertTo(result,CV_32F);
	if (result.empty())
	{
		std::cout << "Can not get the image file !" << endl;
		return;
	}

    result /= 255.0;

	int height = int(result.rows);
	int width = int(result.cols);
	float *data_buf= new float[height * width * 3];

	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			data_buf[(0 * height + h)*width + w] = float(result.at<cv::Vec3f>(cv::Point(w, h))[0]);
			data_buf[(1 * height + h)*width + w] = float(result.at<cv::Vec3f>(cv::Point(w, h))[1]);
			data_buf[(2 * height + h)*width + w] = float(result.at<cv::Vec3f>(cv::Point(w, h))[2]);
		}
	}

	const float* detections = NULL;
    
	net_->blob_by_name("data")->Reshape(1, 3, height, width);
	net_->blob_by_name("data")->set_cpu_data(data_buf);
    clock_t t1 = clock();
	net_->ForwardFrom(0);
	std::cout << " Time Using CPU: " << (clock() - t1)*1.0 / CLOCKS_PER_SEC << std::endl;

	detections = net_->blob_by_name("detection_out")->cpu_data();
	
	vector<vector<float>> boxes;
    vector<float> conf;
	vector<int> labels;

	while(1) {
    
	if (*detections == 0)   
	   detections++;

	else if (*detections !=0 && *detections <= 0.001)
		break;

	else
	{
	vector<float> v1; 
    labels.push_back(*detections);
	detections++;
	conf.push_back(*detections);
	detections++;
	v1.push_back(*detections);
	detections++;
	v1.push_back(*detections);
	detections++;
	v1.push_back(*detections);
	detections++;
	v1.push_back(*detections);
    boxes.push_back(v1);
	detections++;
	}

	
}
vis_detections(cv_image, boxes, conf, labels ,height, width);
cv::imwrite("vis.jpg",  cv_image);
}


void Detector::vis_detections(cv::Mat image, vector<vector<float> > pred_boxes, vector<float> confidence,vector<int> labels, int h , int w)
{
	int lab;
	string cl;
	string labs[2] = { "Mask", "Unmask"};
	for(int i=0; i<confidence.size();i++){
		lab = labels[i] - 1;
		//std::string conf = std::to_string(float(round(confidence[i])));
		//cl = labs[lab] + ':' + conf;
		cv::rectangle(image, cv::Point(pred_boxes[i][0] * w, pred_boxes[i][1] * h), cv::Point(pred_boxes[i][2] * w, pred_boxes[i][3] * h), cv::Scalar(255, 0, 255));
		cv::putText(image, labs[lab] ,cv::Point(pred_boxes[i][0] * w, pred_boxes[i][1] * h - 4),cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 255, 0));
	}
}

int main(int argc, char **argv)
{
	string model_file = "/home/rahul/caffe-mask/mobilenet-only.prototxt";
	string weights_file = "/home/rahul/caffe-mask/ssd_final.caffemodel";
	Caffe::set_mode(Caffe::CPU);
    //Caffe::set_mode(Caffe::CPU);
	Detector det = Detector(model_file, weights_file);
	string im_names=argv[1];
    det.Detect(im_names);
	//det.Detect_video("/data/zou/code/video01.avi");
    return 0;
}
```

*  Input is read in integer format by cv::imread
*  Converted that to float format.
*  bboxes are pushed into other vectors individually.
*  confs and classes are also put inside vectors.


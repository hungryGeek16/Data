### The File: 

#### Created file for cpp inference:

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

	cv::Mat cv_img = cv::imread(im_name);  //Read image file
	cv::cvtColor(cv_img, cv_img, cv::COLOR_BGR2RGB); //Change image format
	if (cv_img.empty())
	{
		std::cout << "Can not get the image file !" << endl;
		return;
	}

        
        cv_img /= 255.0;
	cv_img -= 0.5 ;

	int height = int(cv_img.rows); //capture image height
	int width = int(cv_img.cols); //capture image width
	float *data_buf= new float[height * width * 3]; //initialize image width data buffer

	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			data_buf[(0 * height + h)*width + w] = float(cv_img.at<cv::Vec3f>(cv::Point(w, h))[0]);  //Copy 1st channel of image into the buffer
			data_buf[(1 * height + h)*width + w] = float(cv_img.at<cv::Vec3f>(cv::Point(w, h))[1]); //Copy 2nd channel of image into the buffer
			data_buf[(2 * height + h)*width + w] = float(cv_img.at<cv::Vec3f>(cv::Point(w, h))[2]); //Copy 3rd channel of image into the buffer
		}
	}

	const float* detections = NULL; # intialize pointer array for collecting outputs in array format

	net_->blob_by_name("data")->Reshape(1, 3, height, width); //Reshape data layer according to the image
	net_->blob_by_name("data")->set_cpu_data(data_buf); //Pass image through the network.
    clock_t t1 = clock(); //Note down clock time
	net_->ForwardFrom(0); //Inference model
	std::cout << " Time Using CPU: " << (clock() - t1)*1.0 / CLOCKS_PER_SEC << std::endl; //Time taken for inference
	detections = net_->blob_by_name("detection_out")->cpu_data();  //Pointer array assigned to the output
}



int main()
{
	string model_file = "/home/rahul/caffe-mask/mobilenet-only.prototxt";
	string weights_file = "/home/rahul/caffe-mask/ssd_final.caffemodel";
	Caffe::set_mode(Caffe::CPU);
	Detector det = Detector(model_file, weights_file);
	string im_names="/home/rahul/caffe-mask/assets/image.jpg";
  det.Detect(im_names);
  return 0;
}
```
* This is the file used for inference which I pasted in caffe/examples/cpp_classification/ with name ssd.cpp .

* Ran ```make all``` in caffe main directory. Hencce it created ssd.bin in build/example/cpp_classification/

* Ran this command ```./ssd.bin```  and the output detections are not found in the image and the inference took 1.16363 ms which is double the time taken in python.

* **Log**:
```bash

I1028 16:39:00.106056 13994 net.cpp:228] data_input_0_split does not need backward computation.
I1028 16:39:00.106070 13994 net.cpp:228] input does not need backward computation.
I1028 16:39:00.106077 13994 net.cpp:270] This network produces output detection_out
I1028 16:39:00.106252 13994 net.cpp:283] Network initialization done.
I1028 16:39:00.113222 13994 upgrade_proto.cpp:77] Attempting to upgrade batch norm layers using deprecated params: /home/rahul/caffe-mask/ssd_final.caffemodel
I1028 16:39:00.113268 13994 upgrade_proto.cpp:80] Successfully upgraded batch norm layers using deprecated params.
 Time Using CPU: 1.16363
```

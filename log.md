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
	cv_img.convertTo(cv_img,CV_32F); // imread reads the image in int format but for further calculations we need image to be float.
	if (cv_img.empty())
	{
		std::cout << "Can not get the image file !" << endl;
		return;
	}

        
        cv_img /= 255.0; //divde by 255
	cv_img -= 0.5 ; // subtract offset 0.5

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

* Ran this command ```./ssd.bin```  and the output detections are found.

* The inference took 1.21901 ms which greater than time taken in python which is 0.76.

* **Log from cpp**:
```bash
I1029 10:37:43.044219 17563 upgrade_proto.cpp:77] Attempting to upgrade batch norm layers using deprecated params: /home/rahul/caffe-mask/ssd_final.caffemodel
I1029 10:37:43.044250 17563 upgrade_proto.cpp:80] Successfully upgraded batch norm layers using deprecated params.
 Time Using CPU: 1.21901

0 
2 # predicted label for first detection
0.99968  # predicted confidence for first detection
0.426656 # xmin
0.27332 #ymin
0.526396  #xmax
0.430169 #ymax

0 
2 # predicted label for first detection
0.995123  # predicted confidence for first detection
0.601631 # xmin
0.199528 #ymin
0.710229 #xmax
0.38495 #ymax

### 3rd detection
0
2 
0.996713 
0.765845 
0.107682 
0.888089 
0.300617 


### 4th detection
0 
2 
0.966352 
0.108485 
0.0765258 
0.214637 
0.282962 
```
* **Log from python**:

```bash

I1029 10:44:16.291287 17831 upgrade_proto.cpp:80] Successfully upgraded batch norm layers using deprecated params.
[*] Predict assets/image.jpg image.. 
{'detection_out': array([[[[0.        , 2.        , 0.9998617 , 0.4246654 , 0.27943492,0.5274798 , 0.4283561 ],
         [0.        , 2.        , 0.99984086, 0.60743713, 0.2034382 ,0.7110909 , 0.3792821 ],
         [0.        , 2.        , 0.9974874 , 0.769586  , 0.10440747,0.88734215, 0.30089438],
         [0.        , 2.        , 0.9838861 , 0.10643686, 0.07560761,0.20823473, 0.28588688]]]], dtype=float32)}
Time taken for detections: 0.75697922706604
```

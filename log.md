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

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))

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

	for (int h = 0; h < cv_img.rows; ++h)
	{
		for (int w = 0; w < cv_img.cols; ++w)
		{
			cv_img.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(254.5); // Subtracts 255-0.5 from channel 1, 0.5 is offset
			cv_img.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(254.5); // Subtracts 255-0.5 from channel 2, 0.5 is offset
			cv_img.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(254.5); // Subtracts 255-0.5 from channel 3, 0.5 is offset

		}
	}

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

* Ran this command ```./ssd.bin```  and the output said there's no detection found in the image and the inference took 1.16363 ms which is dobule the time taken in python.

* At the end it prints some gibberish lines which it calls **MEMORY_MAPS*..

* **Log**:
```bash

I1028 16:39:00.106056 13994 net.cpp:228] data_input_0_split does not need backward computation.
I1028 16:39:00.106070 13994 net.cpp:228] input does not need backward computation.
I1028 16:39:00.106077 13994 net.cpp:270] This network produces output detection_out
I1028 16:39:00.106252 13994 net.cpp:283] Network initialization done.
I1028 16:39:00.113222 13994 upgrade_proto.cpp:77] Attempting to upgrade batch norm layers using deprecated params: /home/rahul/caffe-mask/ssd_final.caffemodel
I1028 16:39:00.113268 13994 upgrade_proto.cpp:80] Successfully upgraded batch norm layers using deprecated params.
I1028 16:39:00.841055 13994 detection_output_layer.cpp:282] Couldn't find any detections
 Time Using CPU: 1.16363
*** Error in `./ssd.bin': double free or corruption (out): 0x00007ff59e1be010 ***
======= Backtrace: =========
/lib/x86_64-linux-gnu/libc.so.6(+0x777f5)[0x7ff5ccdd57f5]
/lib/x86_64-linux-gnu/libc.so.6(+0x8038a)[0x7ff5ccdde38a]
/lib/x86_64-linux-gnu/libc.so.6(cfree+0x4c)[0x7ff5ccde258c]
/home/rahul/ssd/caffe/.build_release/examples/cpp_classification/../../lib/libcaffe.so.1.0.0-rc3(_ZN5boost6detail17sp_counted_impl_pIN5caffe12SyncedMemoryEE7disposeEv+0x12)[0x7ff5ce7d5cb2]
/home/rahul/ssd/caffe/.build_release/examples/cpp_classification/../../lib/libcaffe.so.1.0.0-rc3(_ZN5boost6detail17sp_counted_impl_pIN5caffe4BlobIfEEE7disposeEv+0xa2)[0x7ff5ce705872]
/home/rahul/ssd/caffe/.build_release/examples/cpp_classification/../../lib/libcaffe.so.1.0.0-rc3(_ZN5caffe16ConvolutionLayerIfED0Ev+0x428)[0x7ff5ce7b03f8]
./ssd.bin(_ZN5boost6detail17sp_counted_impl_pIN5caffe3NetIfEEE7disposeEv+0x532)[0x404352]
./ssd.bin[0x4025fa]
./ssd.bin[0x402415]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf0)[0x7ff5ccd7e840]
./ssd.bin[0x402519]
======= Memory map: ========
00400000-00405000 r-xp 00000000 08:06 6560130                            /home/rahul/ssd/caffe/.build_release/examples/cpp_classification/ssd.bin
00605000-00606000 r--p 00005000 08:06 6560130                            /home/rahul/ssd/caffe/.build_release/examples/cpp_classification/ssd.bin
00606000-00607000 rw-p 00006000 08:06 6560130                            /home/rahul/ssd/caffe/.build_release/examples/cpp_classification/ssd.bin
01b00000-06997000 rw-p 00000000 00:00 0                                  [heap]
7ff547cb5000-7ff54aeed000 rw-p 00000000 00:00 0 
7ff55abe8000-7ff55bcf4000 rw-p 00000000 00:00 0 
7ff55e983000-7ff55f84d000 rw-p 00000000 00:00 0 
7ff574000000-7ff574021000 rw-p 00000000 00:00 0 
7ff574021000-7ff578000000 ---p 00000000 00:00 0 
7ff57c000000-7ff57c021000 rw-p 00000000 00:00 0 
7ff57c021000-7ff580000000 ---p 00000000 00:00 0 
7ff582b4c000-7ff582b4d000 ---p 00000000 00:00 0 
7ff582b4d000-7ff58334d000 rw-p 00000000 00:00 0 
7ff58334d000-7ff583eda000 r-xp 00000000 08:06 533632                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_vml_avx2.so
7ff583eda000-7ff5840da000 ---p 00b8d000 08:06 533632                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_vml_avx2.so
7ff5840da000-7ff5840dd000 r--p 00b8d000 08:06 533632                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_vml_avx2.so
7ff5840dd000-7ff5840f4000 rw-p 00b90000 08:06 533632                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_vml_avx2.so
7ff5840f4000-7ff5840f5000 rw-p 00000000 00:00 0 
7ff5840f5000-7ff586d95000 r-xp 00000000 08:06 533619                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_avx2.so
7ff586d95000-7ff586f95000 ---p 02ca0000 08:06 533619                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_avx2.so
7ff586f95000-7ff586f9c000 r--p 02ca0000 08:06 533619                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_avx2.so
7ff586f9c000-7ff586faa000 rw-p 02ca7000 08:06 533619                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_avx2.so
7ff586faa000-7ff586fac000 rw-p 00000000 00:00 0 
7ff586fac000-7ff587903000 r-xp 00000000 08:06 533625                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_intel_lp64.so
7ff587903000-7ff587b03000 ---p 00957000 08:06 533625                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_intel_lp64.so
7ff587b03000-7ff587b04000 r--p 00957000 08:06 533625                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_intel_lp64.so
7ff587b04000-7ff587b16000 rw-p 00958000 08:06 533625                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_intel_lp64.so
7ff587b16000-7ff587b1c000 rw-p 00000000 00:00 0 
7ff587b1c000-7ff5891e2000 r-xp 00000000 08:06 533791                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_gnu_thread.so
7ff5891e2000-7ff5893e1000 ---p 016c6000 08:06 533791                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_gnu_thread.so
7ff5893e1000-7ff5893e5000 r--p 016c5000 08:06 533791                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_gnu_thread.so
7ff5893e5000-7ff5893fd000 rw-p 016c9000 08:06 533791                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_gnu_thread.so
7ff5893fd000-7ff5893fe000 rw-p 00000000 00:00 0 
7ff5893fe000-7ff58d520000 r-xp 00000000 08:06 533622                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_core.so
7ff58d520000-7ff58d720000 ---p 04122000 08:06 533622                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_core.so
7ff58d720000-7ff58d727000 r--p 04122000 08:06 533622                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_core.so
7ff58d727000-7ff58d757000 rw-p 04129000 08:06 533622                     /opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin/libmkl_core.so
7ff58d757000-7ff58d77e000 rw-p 00000000 00:00 0 
7ff58f729000-7ff590000000 rw-p 00000000 00:00 0 
7ff590000000-7ff590021000 rw-p 00000000 00:00 0 
7ff590021000-7ff594000000 ---p 00000000 00:00 0 
7ff594000000-7ff594021000 rw-p 00000000 00:00 0 
7ff594021000-7ff598000000 ---p 00000000 00:00 0 
7ff598000000-7ff598021000 rw-p 00000000 00:00 0 
7ff598021000-7ff59c000000 ---p 00000000 00:00 0 
7ff59c4a0000-7ff59c667000 r-xp 00000000 08:06 533377                     /opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin/libiomp5.so
7ff59c667000-7ff59c867000 ---p 001c7000 08:06 533377                     /opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin/libiomp5.so
7ff59c867000-7ff59c86a000 r--p 001c7000 08:06 533377                     /opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin/libiomp5.so
7ff59c86a000-7ff59c874000 rw-p 001ca000 08:06 533377                     /opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin/libiomp5.so
7ff59c874000-7ff59c8a0000 rw-p 00000000 00:00 0 
7ff59c8a0000-7ff59c8a1000 ---p 00000000 00:00 0 
7ff59c8a1000-7ff59cca1000 rw-p 00000000 00:00 0 
7ff59cca1000-7ff59cca2000 ---p 00000000 00:00 0 
7ff59cca2000-7ff59d0a2000 rw-p 00000000 00:00 0 
7ff59d0a2000-7ff59d0a3000 ---p 00000000 00:00 0 
7ff59d0a3000-7ff59daf7000 rw-p 00000000 00:00 0 
7ff59daf7000-7ff59db2d000 r-xp 00000000 08:06 533346                     /opt/intel/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8/libtbbmalloc.so.2
7ff59db2d000-7ff59dd2d000 ---p 00036000 08:06 533346                     /opt/intel/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8/libtbbmalloc.so.2
7ff59dd2d000-7ff59dd2e000 r--p 00036000 08:06 533346                     /opt/intel/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8/libtbbmalloc.so.2
7ff59dd2e000-7ff59dd31000 rw-p 00037000 08:06 533346                     /opt/intel/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8/libtbbmalloc.so.2
7ff59dd31000-7ff59dd52000 rw-p 00000000 00:00 0 
7ff59e1be000-7ff59e207000 rw-p 00000000 00:00 0 
7ff59e463000-7ff59e464000 ---p 00000000 00:00 0 
7ff59e464000-7ff59ec64000 rw-p 00000000 00:00 0 
7ff59ec64000-7ff5a4c64000 rw-p 00000000 00:00 0 
7ff5a4c64000-7ff5a4c65000 ---p 00000000 00:00 0 
7ff5a4c65000-7ff5a5465000 rw-p 00000000 00:00 0 
7ff5a5465000-7ff5a5466000 ---p 00000000 00:00 0 
7ff5a5466000-7ff5a5c66000 rw-p 00000000 00:00 0 
7ff5a5c66000-7ff5a5c67000 ---p 00000000 00:00 0 
7ff5a5c67000-7ff5a6467000 rw-p 00000000 00:00 0 
7ff5a6467000-7ff5a6470000 r-xp 00000000 08:06 16720564                   /lib/x86_64-linux-gnu/libcrypt-2.23.so
7ff5a6470000-7ff5a666f000 ---p 00009000 08:06 16720564                   /lib/x86_64-linux-gnu/libcrypt-2.23.so
7ff5a666f000-7ff5a6670000 r--p 00008000 08:06 16720564                   /lib/x86_64-linux-gnu/libcrypt-2.23.so
7ff5a6670000-7ff5a6671000 rw-p 00009000 08:06 16720564                   /lib/x86_64-linux-gnu/libcrypt-2.23.so
7ff5a6671000-7ff5a669f000 rw-p 00000000 00:00 0 
7ff5a669f000-7ff5a66e6000 r-xp 00000000 08:06 401788                     /usr/lib/x86_64-linux-gnu/libhx509.so.5.0.0
7ff5a66e6000-7ff5a68e5000 ---p 00047000 08:06 401788                     /usr/lib/x86_64-linux-gnu/libhx509.so.5.0.0
7ff5a68e5000-7ff5a68e7000 r--p 00046000 08:06 401788                     /usr/lib/x86_64-linux-gnu/libhx509.so.5.0.0
7ff5a68e7000-7ff5a68e9000 rw-p 00048000 08:06 401788                     /usr/lib/x86_64-linux-gnu/libhx509.so.5.0.0
7ff5a68e9000-7ff5a68ea000 rw-p 00000000 00:00 0 
7ff5a68ea000-7ff5a68f8000 r-xp 00000000 08:06 401770                     /usr/lib/x86_64-linux-gnu/libheimbase.so.1.0.0
7ff5a68f8000-7ff5a6af7000 ---p 0000e000 08:06 401770                     /usr/lib/x86_64-linux-gnu/libheimbase.so.1.0.0
7ff5a6af7000-7ff5a6af8000 r--p 0000d000 08:06 401770                     /usr/lib/x86_64-linux-gnu/libheimbase.so.1.0.0
7ff5a6af8000-7ff5a6af9000 rw-p 0000e000 08:06 401770                     /usr/lib/x86_64-linux-gnu/libheimbase.so.1.0.0
7ff5a6af9000-7ff5a6b20000 r-xp 00000000 08:06 402451                     /usr/lib/x86_64-linux-gnu/libwind.so.0.0.0
7ff5a6b20000-7ff5a6d20000 ---p 00027000 08:06 402451                     /usr/lib/x86_64-linux-gnu/libwind.so.0.0.0
7ff5a6d20000-7ff5a6d21000 r--p 00027000 08:06 402451                     /usr/lib/x86_64-linux-gnu/libwind.so.0.0.0
7ff5a6d21000-7ff5a6d22000 rw-p 00028000 08:06 402451                     /usr/lib/x86_64-linux-gnu/libwind.so.0.0.0
7ff5a6d22000-7ff5a6d37000 r-xp 00000000 08:06 402190                     /usr/lib/x86_64-linux-gnu/libroken.so.18.1.0
7ff5a6d37000-7ff5a6f36000 ---p 00015000 08:06 402190                     /usr/lib/x86_64-linux-gnu/libroken.so.18.1.0
7ff5a6f36000-7ff5a6f37000 r--p 00014000 08:06 402190                     /usr/lib/x86_64-linux-gnu/libroken.so.18.1.0
7ff5a6f37000-7ff5a6f38000 rw-p 00015000 08:06 402190                     /usr/lib/x86_64-linux-gnu/libroken.so.18.1.0
7ff5a6f38000-7ff5a6f68000 r-xp 00000000 08:06 401768                     /usr/lib/x86_64-linux-gnu/libhcrypto.so.4.1.0
7ff5a6f68000-7ff5a7168000 ---p 00030000 08:06 401768                     /usr/lib/x86_64-linux-gnu/libhcrypto.so.4.1.0
7ff5a7168000-7ff5a7169000 r--p 00030000 08:06 401768                     /usr/lib/x86_64-linux-gnu/libhcrypto.so.4.1.0
7ff5a7169000-7ff5a716a000 rw-p 00031000 08:06 401768                     /usr/lib/x86_64-linux-gnu/libhcrypto.so.4.1.0
7ff5a716a000-7ff5a716b000 rw-p 00000000 00:00 0 
7ff5a716b000-7ff5a720a000 r-xp 00000000 08:06 401205                     /usr/lib/x86_64-linux-gnu/libasn1.so.8.0.0
7ff5a720a000-7ff5a7409000 ---p 0009f000 08:06 401205                     /usr/lib/x86_64-linux-gnu/libasn1.so.8.0.0
7ff5a7409000-7ff5a740a000 r--p 0009e000 08:06 401205                     /usr/lib/x86_64-linux-gnu/libasn1.so.8.0.0
7ff5a740a000-7ff5a740d000 rw-p 0009f000 08:06 401205                     /usr/lib/x86_64-linux-gnu/libasn1.so.8.0.0
7ff5a740d000-7ff5a7491000 r-xp 00000000 08:06 401875                     /usr/lib/x86_64-linux-gnu/libkrb5.so.26.0.0
7ff5a7491000-7ff5a7690000 ---p 00084000 08:06 401875                     /usr/lib/x86_64-linux-gnu/libkrb5.so.26.0.0
7ff5a7690000-7ff5a7693000 r--p 00083000 08:06 401875                     /usr/lib/x86_64-linux-gnu/libkrb5.so.26.0.0
7ff5a7693000-7ff5a7696000 rw-p 00086000 08:06 401875                     /usr/lib/x86_64-linux-gnu/libkrb5.so.26.0.0
7ff5a7696000-7ff5a7697000 rw-p 00000000 00:00 0 
7ff5a7697000-7ff5a769f000 r-xp 00000000 08:06 401772                     /usr/lib/x86_64-linux-gnu/libheimntlm.so.0.1.0
7ff5a769f000-7ff5a789e000 ---p 00008000 08:06 401772                     /usr/lib/x86_64-linux-gnu/libheimntlm.so.0.1.0
7ff5a789e000-7ff5a789f000 r--p 00007000 08:06 401772                     /usr/lib/x86_64-linux-gnu/libheimntlm.so.0.1.0
7ff5a789f000-7ff5a78a0000 rw-p 00008000 08:06 401772                     /usr/lib/x86_64-linux-gnu/libheimntlm.so.0.1.0
7ff5a78a0000-7ff5a78a3000 r-xp 00000000 08:06 16650882                   /lib/x86_64-linux-gnu/libkeyutils.so.1.5
7ff5a78a3000-7ff5a7aa2000 ---p 00003000 08:06 16650882                   /lib/x86_64-linux-gnu/libkeyutils.so.1.5
7ff5a7aa2000-7ff5a7aa3000 r--p 00002000 08:06 16650882                   /lib/x86_64-linux-gnu/libkeyutils.so.1.5
7ff5a7aa3000-7ff5a7aa4000 rw-p 00003000 08:06 16650882                   /lib/x86_64-linux-gnu/libkeyutils.so.1.5
7ff5a7aa4000-7ff5a7aea000 r-xp 00000000 08:06 407579                     /usr/lib/x86_64-linux-gnu/libquadmath.so.0.0.0
7ff5a7aea000-7ff5a7ce9000 ---p 00046000 08:06 407579                     /usr/lib/x86_64-linux-gnu/libquadmath.so.0.0.0
7ff5a7ce9000-7ff5a7cea000 r--p 00045000 08:06 407579                     /usr/lib/x86_64-linux-gnu/libquadmath.so.0.0.0
7ff5a7cea000-7ff5a7ceb000 rw-p 00046000 08:06 407579                     /usr/lib/x86_64-linux-gnu/libquadmath.so.0.0.0
7ff5a7ceb000-7ff5a7cfd000 r-xp 00000000 08:06 16650865                   /lib/x86_64-linux-gnu/libgpg-error.so.0.17.0
7ff5a7cfd000-7ff5a7efd000 ---p 00012000 08:06 16650865                   /lib/x86_64-linux-gnu/libgpg-error.so.0.17.0
7ff5a7efd000-7ff5a7efe000 r--p 00012000 08:06 16650865                   /lib/x86_64-linux-gnu/libgpg-error.so.0.17.0
7ff5a7efe000-7ff5a7eff000 rw-p 00013000 08:06 16650865                   /lib/x86_64-linux-gnu/libgpg-error.so.0.17.0
7ff5a7eff000-7ff5a7f35000 r-xp 00000000 08:06 393404                     /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
7ff5a7f35000-7ff5a8134000 ---p 00036000 08:06 393404                     /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
7ff5a8134000-7ff5a8135000 r--p 00035000 08:06 393404                     /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
7ff5a8135000-7ff5a8136000 rw-p 00036000 08:06 393404                     /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
7ff5a8136000-7ff5a8173000 r-xp 00000000 08:06 401691                     /usr/lib/x86_64-linux-gnu/libgssapi.so.3.0.0
7ff5a8173000-7ff5a8373000 ---p 0003d000 08:06 401691                     /usr/lib/x86_64-linux-gnu/libgssapi.so.3.0.0
7ff5a8373000-7ff5a8374000 r--p 0003d000 08:06 401691                     /usr/lib/x86_64-linux-gnu/libgssapi.so.3.0.0
7ff5a8374000-7ff5a8376000 rw-p 0003e000 08:06 401691                     /usr/lib/x86_64-linux-gnu/libgssapi.so.3.0.0
7ff5a8376000-7ff5a8377000 rw-p 00000000 00:00 0 
7ff5a8377000-7ff5a8390000 r-xp 00000000 08:06 398999                     /usr/lib/x86_64-linux-gnu/libsasl2.so.2.0.25
7ff5a8390000-7ff5a8590000 ---p 00019000 08:06 398999                     /usr/lib/x86_64-linux-gnu/libsasl2.so.2.0.25
7ff5a8590000-7ff5a8591000 r--p 00019000 08:06 398999                     /usr/lib/x86_64-linux-gnu/libsasl2.so.2.0.25
7ff5a8591000-7ff5a8592000 rw-p 0001a000 08:06 398999                     /usr/lib/x86_64-linux-gnu/libsasl2.so.2.0.25
7ff5a8592000-7ff5a85a9000 r-xp 00000000 08:06 16720613                   /lib/x86_64-linux-gnu/libresolv-2.23.so
7ff5a85a9000-7ff5a87a9000 ---p 00017000 08:06 16720613                   /lib/x86_64-linux-gnu/libresolv-2.23.so
7ff5a87a9000-7ff5a87aa000 r--p 00017000 08:06 16720613                   /lib/x86_64-linux-gnu/libresolv-2.23.so
7ff5a87aa000-7ff5a87ab000 rw-p 00018000 08:06 16720613                   /lib/x86_64-linux-gnu/libresolv-2.23.so
7ff5a87ab000-7ff5a87ad000 rw-p 00000000 00:00 0 
7ff5a87ad000-7ff5a87b7000 r-xp 00000000 08:06 401879                     /usr/lib/x86_64-linux-gnu/libkrb5support.so.0.1
7ff5a87b7000-7ff5a89b6000 ---p 0000a000 08:06 401879                     /usr/lib/x86_64-linux-gnu/libkrb5support.so.0.1
7ff5a89b6000-7ff5a89b7000 r--p 00009000 08:06 401879                     /usr/lib/x86_64-linux-gnu/libkrb5support.so.0.1
7ff5a89b7000-7ff5a89b8000 rw-p 0000a000 08:06 401879                     /usr/lib/x86_64-linux-gnu/libkrb5support.so.0.1
7ff5a89b8000-7ff5a89bb000 r-xp 00000000 08:06 16652342                   /lib/x86_64-linux-gnu/libcom_err.so.2.1
7ff5a89bb000-7ff5a8bba000 ---p 00003000 08:06 16652342                   /lib/x86_64-linux-gnu/libcom_err.so.2.1
7ff5a8bba000-7ff5a8bbb000 r--p 00002000 08:06 16652342                   /lib/x86_64-linux-gnu/libcom_err.so.2.1
7ff5a8bbb000-7ff5a8bbc000 rw-p 00003000 08:06 16652342                   /lib/x86_64-linux-gnu/libcom_err.so.2.1
7ff5a8bbc000-7ff5a8be8000 r-xp 00000000 08:06 401869                     /usr/lib/x86_64-linux-gnu/libk5crypto.so.3.1
7ff5a8be8000-7ff5a8de7000 ---p 0002c000 08:06 401869                     /usr/lib/x86_64-linux-gnu/libk5crypto.so.3.1
7ff5a8de7000-7ff5a8de9000 r--p 0002b000 08:06 401869                     /usr/lib/x86_64-linux-gnu/libk5crypto.so.3.1
7ff5a8de9000-7ff5a8dea000 rw-p 0002d000 08:06 401869                     /usr/lib/x86_64-linux-gnu/libk5crypto.so.3.1
7ff5a8dea000-7ff5a8deb000 rw-p 00000000 00:00 0 
7ff5a8deb000-7ff5a8eae000 r-xp 00000000 08:06 401877                     /usr/lib/x86_64-linux-gnu/libkrb5.so.3.3
7ff5a8eae000-7ff5a90ae000 ---p 000c3000 08:06 401877                     /usr/lib/x86_64-linux-gnu/libkrb5.so.3.3
7ff5a90ae000-7ff5a90bb000 r--p 000c3000 08:06 401877                     /usr/lib/x86_64-linux-gnu/libkrb5.so.3.3
7ff5a90bb000-7ff5a90bd000 rw-p 000d0000 08:06 401877                     /usr/lib/x86_64-linux-gnu/libkrb5.so.3.3
7ff5a90bd000-7ff5a91eb000 r-xp 00000000 08:06 399572                     /usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0
7ff5a91eb000-7ff5a93eb000 ---p 0012e000 08:06 399572                     /usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0
7ff5a93eb000-7ff5a93ec000 r--p 0012e000 08:06 399572                     /usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0
7ff5a93ec000-7ff5a93ee000 rw-p 0012f000 08:06 399572                     /usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0
7ff5a93ee000-7ff5ab252000 r-xp 00000000 08:06 396625                     /usr/lib/libopenblasp-r0.2.18.so
7ff5ab252000-7ff5ab451000 ---p 01e64000 08:06 396625                     /usr/lib/libopenblasp-r0.2.18.so
7ff5ab451000-7ff5ab457000 r--p 01e63000 08:06 396625                     /usr/lib/libopenblasp-r0.2.18.so
7ff5ab457000-7ff5ab469000 rw-p 01e69000 08:06 396625                     /usr/lib/libopenblasp-r0.2.18.so
7ff5ab469000-7ff5ab482000 rw-p 00000000 00:00 0 
7ff5ab482000-7ff5ab493000 r-xp 00000000 08:06 402295                     /usr/lib/x86_64-linux-gnu/libtasn1.so.6.5.1
7ff5ab493000-7ff5ab693000 ---p 00011000 08:06 402295                     /usr/lib/x86_64-linux-gnu/libtasn1.so.6.5.1
7ff5ab693000-7ff5ab694000 r--p 00011000 08:06 402295                     /usr/lib/x86_64-linux-gnu/libtasn1.so.6.5.1
7ff5ab694000-7ff5ab695000 rw-p 00012000 08:06 402295                     /usr/lib/x86_64-linux-gnu/libtasn1.so.6.5.1
7ff5ab695000-7ff5ab6ee000 r-xp 00000000 08:06 402062                     /usr/lib/x86_64-linux-gnu/libp11-kit.so.0.1.0
7ff5ab6ee000-7ff5ab8ed000 ---p 00059000 08:06 402062                     /usr/lib/x86_64-linux-gnu/libp11-kit.so.0.1.0
7ff5ab8ed000-7ff5ab8f7000 r--p 00058000 08:06 402062                     /usr/lib/x86_64-linux-gnu/libp11-kit.so.0.1.0
7ff5ab8f7000-7ff5ab8f9000 rw-p 00062000 08:06 402062                     /usr/lib/x86_64-linux-gnu/libp11-kit.so.0.1.0
7ff5ab8f9000-7ff5ab978000 r-xp 00000000 08:06 401636                     /usr/lib/x86_64-linux-gnu/libgmp.so.10.3.0
7ff5ab978000-7ff5abb77000 ---p 0007f000 08:06 401636                     /usr/lib/x86_64-linux-gnu/libgmp.so.10.3.0
7ff5abb77000-7ff5abb78000 r--p 0007e000 08:06 401636                     /usr/lib/x86_64-linux-gnu/libgmp.so.10.3.0
7ff5abb78000-7ff5abb79000 rw-p 0007f000 08:06 401636                     /usr/lib/x86_64-linux-gnu/libgmp.so.10.3.0
7ff5abb79000-7ff5abbab000 r-xp 00000000 08:06 401774                     /usr/lib/x86_64-linux-gnu/libhogweed.so.4.2
7ff5abbab000-7ff5abdaa000 ---p 00032000 08:06 401774                     /usr/lib/x86_64-linux-gnu/libhogweed.so.4.2
7ff5abdaa000-7ff5abdab000 r--p 00031000 08:06 401774                     /usr/lib/x86_64-linux-gnu/libhogweed.so.4.2
7ff5abdab000-7ff5abdac000 rw-p 00032000 08:06 401774                     /usr/lib/x86_64-linux-gnu/libhogweed.so.4.2
7ff5abdac000-7ff5abe83000 r-xp 00000000 08:06 16650829                   /lib/x86_64-linux-gnu/libgcrypt.so.20.0.5
7ff5abe83000-7ff5ac083000 ---p 000d7000 08:06 16650829                   /lib/x86_64-linux-gnu/libgcrypt.so.20.0.5
7ff5ac083000-7ff5ac084000 r--p 000d7000 08:06 16650829                   /lib/x86_64-linux-gnu/libgcrypt.so.20.0.5
7ff5ac084000-7ff5ac08c000 rw-p 000d8000 08:06 16650829                   /lib/x86_64-linux-gnu/libgcrypt.so.20.0.5
7ff5ac08c000-7ff5ac08d000 rw-p 00000000 00:00 0 
7ff5ac08d000-7ff5ac108000 r-xp 00000000 08:06 402052                     /usr/lib/x86_64-linux-gnu/liborc-0.4.so.0.25.0
7ff5ac108000-7ff5ac307000 ---p 0007b000 08:06 402052                     /usr/lib/x86_64-linux-gnu/liborc-0.4.so.0.25.0
7ff5ac307000-7ff5ac309000 r--p 0007a000 08:06 402052                     /usr/lib/x86_64-linux-gnu/liborc-0.4.so.0.25.0
7ff5ac309000-7ff5ac30d000 rw-p 0007c000 08:06 402052                     /usr/lib/x86_64-linux-gnu/liborc-0.4.so.0.25.0
7ff5ac30d000-7ff5ac314000 r-xp 00000000 08:06 402046                     /usr/lib/x86_64-linux-gnu/libogg.so.0.8.2
7ff5ac314000-7ff5ac514000 ---p 00007000 08:06 402046                     /usr/lib/x86_64-linux-gnu/libogg.so.0.8.2
7ff5ac514000-7ff5ac515000 r--p 00007000 08:06 402046                     /usr/lib/x86_64-linux-gnu/libogg.so.0.8.2
7ff5ac515000-7ff5ac516000 rw-p 00008000 08:06 402046                     /usr/lib/x86_64-linux-gnu/libogg.so.0.8.2
7ff5ac516000-7ff5ac520000 r-xp 00000000 08:06 402034                     /usr/lib/x86_64-linux-gnu/libnuma.so.1.0.0
7ff5ac520000-7ff5ac71f000 ---p 0000a000 08:06 402034                     /usr/lib/x86_64-linux-gnu/libnuma.so.1.0.0
7ff5ac71f000-7ff5ac720000 r--p 00009000 08:06 402034                     /usr/lib/x86_64-linux-gnu/libnuma.so.1.0.0
7ff5ac720000-7ff5ac721000 rw-p 0000a000 08:06 402034                     /usr/lib/x86_64-linux-gnu/libnuma.so.1.0.0
7ff5ac721000-7ff5ac74f000 r-xp 00000000 08:06 393722                     /usr/lib/x86_64-linux-gnu/libsoxr.so.0.1.1
7ff5ac74f000-7ff5ac94e000 ---p 0002e000 08:06 393722                     /usr/lib/x86_64-linux-gnu/libsoxr.so.0.1.1
7ff5ac94e000-7ff5ac950000 r--p 0002d000 08:06 393722                     /usr/lib/x86_64-linux-gnu/libsoxr.so.0.1.1
7ff5ac950000-7ff5ac951000 rw-p 0002f000 08:06 393722                     /usr/lib/x86_64-linux-gnu/libsoxr.so.0.1.1
7ff5ac951000-7ff5ac986000 rw-p 00000000 00:00 0 
7ff5ac986000-7ff5ac9aa000 r-xp 00000000 08:06 401678                     /usr/lib/x86_64-linux-gnu/libgraphite2.so.3.0.1
7ff5ac9aa000-7ff5acba9000 ---p 00024000 08:06 401678                     /usr/lib/x86_64-linux-gnu/libgraphite2.so.3.0.1
7ff5acba9000-7ff5acbab000 r--p 00023000 08:06 401678                     /usr/lib/x86_64-linux-gnu/libgraphite2.so.3.0.1
7ff5acbab000-7ff5acbac000 rw-p 00025000 08:06 401678                     /usr/lib/x86_64-linux-gnu/libgraphite2.so.3.0.1
7ff5acbac000-7ff5acbb3000 r-xp 00000000 08:06 401503                     /usr/lib/x86_64-linux-gnu/libffi.so.6.0.4
7ff5acbb3000-7ff5acdb2000 ---p 00007000 08:06 401503                     /usr/lib/x86_64-linux-gnu/libffi.so.6.0.4
7ff5acdb2000-7ff5acdb3000 r--p 00006000 08:06 401503                     /usr/lib/x86_64-linux-gnu/libffi.so.6.0.4
7ff5acdb3000-7ff5acdb4000 rw-p 00007000 08:06 401503                     /usr/lib/x86_64-linux-gnu/libffi.so.6.0.4
7ff5acdb4000-7ff5acdc1000 r-xp 00000000 08:06 394801                     /usr/lib/x86_64-linux-gnu/liblber-2.4.so.2.10.5
7ff5acdc1000-7ff5acfc1000 ---p 0000d000 08:06 394801                     /usr/lib/x86_64-linux-gnu/liblber-2.4.so.2.10.5
7ff5acfc1000-7ff5acfc2000 r--p 0000d000 08:06 394801                     /usr/lib/x86_64-linux-gnu/liblber-2.4.so.2.10.5
7ff5acfc2000-7ff5acfc3000 rw-p 0000e000 08:06 394801                     /usr/lib/x86_64-linux-gnu/liblber-2.4.so.2.10.5
7ff5acfc3000-7ff5acff7000 r-xp 00000000 08:06 401999                     /usr/lib/x86_64-linux-gnu/libnettle.so.6.2
7ff5acff7000-7ff5ad1f6000 ---p 00034000 08:06 401999                     /usr/lib/x86_64-linux-gnu/libnettle.so.6.2
7ff5ad1f6000-7ff5ad1f8000 r--p 00033000 08:06 401999                     /usr/lib/x86_64-linux-gnu/libnettle.so.6.2
7ff5ad1f8000-7ff5ad1f9000 rw-p 00035000 08:06 401999                     /usr/lib/x86_64-linux-gnu/libnettle.so.6.2
7ff5ad1f9000-7ff5ad22a000 r-xp 00000000 08:06 401823                     /usr/lib/x86_64-linux-gnu/libidn.so.11.6.15
7ff5ad22a000-7ff5ad42a000 ---p 00031000 08:06 401823                     /usr/lib/x86_64-linux-gnu/libidn.so.11.6.15
7ff5ad42a000-7ff5ad42b000 r--p 00031000 08:06 401823                     /usr/lib/x86_64-linux-gnu/libidn.so.11.6.15
7ff5ad42b000-7ff5ad42c000 rw-p 00032000 08:06 401823                     /usr/lib/x86_64-linux-gnu/libidn.so.11.6.15
7ff5ad42c000-7ff5ad4fc000 r-xp 00000000 08:06 414657                     /usr/lib/x86_64-linux-gnu/libsqlite3.so.0.8.6
7ff5ad4fc000-7ff5ad6fb000 ---p 000d0000 08:06 414657                     /usr/lib/x86_64-linux-gnu/libsqlite3.so.0.8.6
7ff5ad6fb000-7ff5ad6fe000 r--p 000cf000 08:06 414657                     /usr/lib/x86_64-linux-gnu/libsqlite3.so.0.8.6
7ff5ad6fe000-7ff5ad700000 rw-p 000d2000 08:06 414657                     /usr/lib/x86_64-linux-gnu/libsqlite3.so.0.8.6
7ff5ad700000-7ff5ad701000 rw-p 00000000 00:00 0 
7ff5ad701000-7ff5ad74e000 r-xp 00000000 08:06 394771                     /usr/lib/x86_64-linux-gnu/libldap_r-2.4.so.2.10.5
7ff5ad74e000-7ff5ad94d000 ---p 0004d000 08:06 394771                     /usr/lib/x86_64-linux-gnu/libldap_r-2.4.so.2.10.5
7ff5ad94d000-7ff5ad94f000 r--p 0004c000 08:06 394771                     /usr/lib/x86_64-linux-gnu/libldap_r-2.4.so.2.10.5
7ff5ad94f000-7ff5ad950000 rw-p 0004e000 08:06 394771                     /usr/lib/x86_64-linux-gnu/libldap_r-2.4.so.2.10.5
7ff5ad950000-7ff5ad952000 rw-p 00000000 00:00 0 
7ff5ad952000-7ff5ad999000 r-xp 00000000 08:06 401693                     /usr/lib/x86_64-linux-gnu/libgssapi_krb5.so.2.2
7ff5ad999000-7ff5adb98000 ---p 00047000 08:06 401693                     /usr/lib/x86_64-linux-gnu/libgssapi_krb5.so.2.2
7ff5adb98000-7ff5adb9a000 r--p 00046000 08:06 401693                     /usr/lib/x86_64-linux-gnu/libgssapi_krb5.so.2.2
7ff5adb9a000-7ff5adb9c000 rw-p 00048000 08:06 401693                     /usr/lib/x86_64-linux-gnu/libgssapi_krb5.so.2.2
7ff5adb9c000-7ff5addb7000 r-xp 00000000 08:06 16653641                   /lib/x86_64-linux-gnu/libcrypto.so.1.0.0
7ff5addb7000-7ff5adfb6000 ---p 0021b000 08:06 16653641                   /lib/x86_64-linux-gnu/libcrypto.so.1.0.0
7ff5adfb6000-7ff5adfd2000 r--p 0021a000 08:06 16653641                   /lib/x86_64-linux-gnu/libcrypto.so.1.0.0
7ff5adfd2000-7ff5adfde000 rw-p 00236000 08:06 16653641                   /lib/x86_64-linux-gnu/libcrypto.so.1.0.0
7ff5adfde000-7ff5adfe1000 rw-p 00000000 00:00 0 
7ff5adfe1000-7ff5ae03f000 r-xp 00000000 08:06 16653648                   /lib/x86_64-linux-gnu/libssl.so.1.0.0
7ff5ae03f000-7ff5ae23f000 ---p 0005e000 08:06 16653648                   /lib/x86_64-linux-gnu/libssl.so.1.0.0
7ff5ae23f000-7ff5ae243000 r--p 0005e000 08:06 16653648                   /lib/x86_64-linux-gnu/libssl.so.1.0.0
7ff5ae243000-7ff5ae249000 rw-p 00062000 08:06 16653648                   /lib/x86_64-linux-gnu/libssl.so.1.0.0
7ff5ae249000-7ff5ae263000 r-xp 00000000 08:06 396137                     /usr/lib/x86_64-linux-gnu/liburiparser.so.1.0.20
7ff5ae263000-7ff5ae462000 ---p 0001a000 08:06 396137                     /usr/lib/x86_64-linux-gnu/liburiparser.so.1.0.20
7ff5ae462000-7ff5ae463000 r--p 00019000 08:06 396137                     /usr/lib/x86_64-linux-gnu/liburiparser.so.1.0.20
7ff5ae463000-7ff5ae464000 rw-p 0001a000 08:06 396137                     /usr/lib/x86_64-linux-gnu/liburiparser.so.1.0.20
7ff5ae464000-7ff5ae46e000 r-xp 00000000 08:06 396135                     /usr/lib/x86_64-linux-gnu/libminizip.so.1.0.0
7ff5ae46e000-7ff5ae66d000 ---p 0000a000 08:06 396135                     /usr/lib/x86_64-linux-gnu/libminizip.so.1.0.0
7ff5ae66d000-7ff5ae66e000 r--p 00009000 08:06 396135                     /usr/lib/x86_64-linux-gnu/libminizip.so.1.0.0
7ff5ae66e000-7ff5ae66f000 rw-p 0000a000 08:06 396135                     /usr/lib/x86_64-linux-gnu/libminizip.so.1.0.0
7ff5ae66f000-7ff5ae7fd000 r-xp 00000000 08:06 396089                     /usr/lib/x86_64-linux-gnu/libgeos-3.5.0.so
7ff5ae7fd000-7ff5ae9fd000 ---p 0018e000 08:06 396089                     /usr/lib/x86_64-linux-gnu/libgeos-3.5.0.so
7ff5ae9fd000-7ff5aea08000 r--p 0018e000 08:06 396089                     /usr/lib/x86_64-linux-gnu/libgeos-3.5.0.so
7ff5aea08000-7ff5aea09000 rw-p 00199000 08:06 396089                     /usr/lib/x86_64-linux-gnu/libgeos-3.5.0.so
7ff5aea09000-7ff5aeaad000 r-xp 00000000 08:06 398563                     /usr/lib/x86_64-linux-gnu/libfreetype.so.6.12.1
7ff5aeaad000-7ff5aecac000 ---p 000a4000 08:06 398563                     /usr/lib/x86_64-linux-gnu/libfreetype.so.6.12.1
7ff5aecac000-7ff5aecb2000 r--p 000a3000 08:06 398563                     /usr/lib/x86_64-linux-gnu/libfreetype.so.6.12.1
7ff5aecb2000-7ff5aecb3000 rw-p 000a9000 08:06 398563                     /usr/lib/x86_64-linux-gnu/libfreetype.so.6.12.1
7ff5aecb3000-7ff5aecf0000 r-xp 00000000 08:06 401517                     /usr/lib/x86_64-linux-gnu/libfontconfig.so.1.9.0
7ff5aecf0000-7ff5aeeef000 ---p 0003d000 08:06 401517                     /usr/lib/x86_64-linux-gnu/libfontconfig.so.1.9.0
7ff5aeeef000-7ff5aeef1000 r--p 0003c000 08:06 401517                     /usr/lib/x86_64-linux-gnu/libfontconfig.so.1.9.0
7ff5aeef1000-7ff5aeef6000 rw-p 0003e000 08:06 401517                     /usr/lib/x86_64-linux-gnu/libfontconfig.so.1.9.0
7ff5aeef6000-7ff5aef48000 r-xp 00000000 08:06 401887                     /usr/lib/x86_64-linux-gnu/liblcms2.so.2.0.6
7ff5aef48000-7ff5af147000 ---p 00052000 08:06 401887                     /usr/lib/x86_64-linux-gnu/liblcms2.so.2.0.6
7ff5af147000-7ff5af148000 r--p 00051000 08:06 401887                     /usr/lib/x86_64-linux-gnu/liblcms2.so.2.0.6
7ff5af148000-7ff5af14c000 rw-p 00052000 08:06 401887                     /usr/lib/x86_64-linux-gnu/liblcms2.so.2.0.6
7ff5af14c000-7ff5af14d000 rw-p 00000000 00:00 0 
7ff5af14d000-7ff5af1b8000 r-xp 00000000 08:06 393417                     /usr/lib/x86_64-linux-gnu/libsuperlu.so.4.0.0
7ff5af1b8000-7ff5af3b7000 ---p 0006b000 08:06 393417                     /usr/lib/x86_64-linux-gnu/libsuperlu.so.4.0.0
7ff5af3b7000-7ff5af3b8000 r--p 0006a000 08:06 393417                     /usr/lib/x86_64-linux-gnu/libsuperlu.so.4.0.0
7ff5af3b8000-7ff5af3b9000 rw-p 0006b000 08:06 393417                     /usr/lib/x86_64-linux-gnu/libsuperlu.so.4.0.0
7ff5af3b9000-7ff5af403000 r-xp 00000000 08:06 393812                     /usr/lib/libarpack.so.2.0.0
7ff5af403000-7ff5af602000 ---p 0004a000 08:06 393812                     /usr/lib/libarpack.so.2.0.0
7ff5af602000-7ff5af603000 r--p 00049000 08:06 393812                     /usr/lib/libarpack.so.2.0.0
7ff5af603000-7ff5af604000 rw-p 0004a000 08:06 393812                     /usr/lib/libarpack.so.2.0.0
7ff5af604000-7ff5afbe4000 r-xp 00000000 08:06 527228                     /usr/lib/openblas-base/liblapack.so.3
7ff5afbe4000-7ff5afde4000 ---p 005e0000 08:06 527228                     /usr/lib/openblas-base/liblapack.so.3
7ff5afde4000-7ff5afde5000 r--p 005e0000 08:06 527228                     /usr/lib/openblas-base/liblapack.so.3
7ff5afde5000-7ff5afde7000 rw-p 005e1000 08:06 527228                     /usr/lib/openblas-base/liblapack.so.3
7ff5afde7000-7ff5afe42000 r-xp 00000000 08:06 527227                     /usr/lib/openblas-base/libblas.so.3
7ff5afe42000-7ff5b0042000 ---p 0005b000 08:06 527227                     /usr/lib/openblas-base/libblas.so.3
7ff5b0042000-7ff5b0047000 r--p 0005b000 08:06 527227                     /usr/lib/openblas-base/libblas.so.3
7ff5b0047000-7ff5b0048000 rw-p 00060000 08:06 527227                     /usr/lib/openblas-base/libblas.so.3
7ff5b0048000-7ff5b004d000 r-xp 00000000 08:06 401124                     /usr/lib/x86_64-linux-gnu/libXdmcp.so.6.0.0
7ff5b004d000-7ff5b024c000 ---p 00005000 08:06 401124                     /usr/lib/x86_64-linux-gnu/libXdmcp.so.6.0.0
7ff5b024c000-7ff5b024d000 r--p 00004000 08:06 401124                     /usr/lib/x86_64-linux-gnu/libXdmcp.so.6.0.0
7ff5b024d000-7ff5b024e000 rw-p 00005000 08:06 401124                     /usr/lib/x86_64-linux-gnu/libXdmcp.so.6.0.0
7ff5b024e000-7ff5b0250000 r-xp 00000000 08:06 401113                     /usr/lib/x86_64-linux-gnu/libXau.so.6.0.0
7ff5b0250000-7ff5b0450000 ---p 00002000 08:06 401113                     /usr/lib/x86_64-linux-gnu/libXau.so.6.0.0
7ff5b0450000-7ff5b0451000 r--p 00002000 08:06 401113                     /usr/lib/x86_64-linux-gnu/libXau.so.6.0.0
7ff5b0451000-7ff5b0452000 rw-p 00003000 08:06 401113                     /usr/lib/x86_64-linux-gnu/libXau.so.6.0.0
7ff5b0452000-7ff5b0483000 r-xp 00000000 08:06 393322                     /usr/lib/x86_64-linux-gnu/libexif.so.12.3.3
7ff5b0483000-7ff5b0683000 ---p 00031000 08:06 393322                     /usr/lib/x86_64-linux-gnu/libexif.so.12.3.3
7ff5b0683000-7ff5b0696000 r--p 00031000 08:06 393322                     /usr/lib/x86_64-linux-gnu/libexif.so.12.3.3
7ff5b0696000-7ff5b0697000 rw-p 00044000 08:06 393322                     /usr/lib/x86_64-linux-gnu/libexif.so.12.3.3
7ff5b0697000-7ff5b06a0000 r-xp 00000000 08:06 401907                     /usr/lib/x86_64-linux-gnu/libltdl.so.7.3.1
7ff5b06a0000-7ff5b089f000 ---p 00009000 08:06 401907                     /usr/lib/x86_64-linux-gnu/libltdl.so.7.3.1
7ff5b089f000-7ff5b08a0000 r--p 00008000 08:06 401907                     /usr/lib/x86_64-linux-gnu/libltdl.so.7.3.1
7ff5b08a0000-7ff5b08a1000 rw-p 00009000 08:06 401907                     /usr/lib/x86_64-linux-gnu/libltdl.so.7.3.1
7ff5b08a1000-7ff5b08b0000 r-xp 00000000 08:06 16650774                   /lib/x86_64-linux-gnu/libbz2.so.1.0.4
7ff5b08b0000-7ff5b0aaf000 ---p 0000f000 08:06 16650774                   /lib/x86_64-linux-gnu/libbz2.so.1.0.4
7ff5b0aaf000-7ff5b0ab0000 r--p 0000e000 08:06 16650774                   /lib/x86_64-linux-gnu/libbz2.so.1.0.4
7ff5b0ab0000-7ff5b0ab1000 rw-p 0000f000 08:06 16650774                   /lib/x86_64-linux-gnu/libbz2.so.1.0.4
7ff5b0ab1000-7ff5b0bd4000 r-xp 00000000 08:06 394454                     /usr/lib/x86_64-linux-gnu/libgnutls.so.30.6.2
7ff5b0bd4000-7ff5b0dd3000 ---p 00123000 08:06 394454                     /usr/lib/x86_64-linux-gnu/libgnutls.so.30.6.2
7ff5b0dd3000-7ff5b0dde000 r--p 00122000 08:06 394454                     /usr/lib/x86_64-linux-gnu/libgnutls.so.30.6.2
7ff5b0dde000-7ff5b0de0000 rw-p 0012d000 08:06 394454                     /usr/lib/x86_64-linux-gnu/libgnutls.so.30.6.2
7ff5b0de0000-7ff5b0de1000 rw-p 00000000 00:00 0 
7ff5b0de1000-7ff5b0e28000 r-xp 00000000 08:06 394722                     /usr/lib/x86_64-linux-gnu/libbluray.so.1.9.2
7ff5b0e28000-7ff5b1027000 ---p 00047000 08:06 394722                     /usr/lib/x86_64-linux-gnu/libbluray.so.1.9.2
7ff5b1027000-7ff5b1029000 r--p 00046000 08:06 394722                     /usr/lib/x86_64-linux-gnu/libbluray.so.1.9.2
7ff5b1029000-7ff5b102a000 rw-p 00048000 08:06 394722                     /usr/lib/x86_64-linux-gnu/libbluray.so.1.9.2
7ff5b102a000-7ff5b1074000 r-xp 00000000 08:06 394981                     /usr/lib/x86_64-linux-gnu/libgme.so.0.6.0
7ff5b1074000-7ff5b1274000 ---p 0004a000 08:06 394981                     /usr/lib/x86_64-linux-gnu/libgme.so.0.6.0
7ff5b1274000-7ff5b1277000 r--p 0004a000 08:06 394981                     /usr/lib/x86_64-linux-gnu/libgme.so.0.6.0
7ff5b1277000-7ff5b1278000 rw-p 0004d000 08:06 394981                     /usr/lib/x86_64-linux-gnu/libgme.so.0.6.0
7ff5b1278000-7ff5b12c2000 r-xp 00000000 08:06 397114                     /usr/lib/x86_64-linux-gnu/libmodplug.so.1.0.0
7ff5b12c2000-7ff5b14c2000 ---p 0004a000 08:06 397114                     /usr/lib/x86_64-linux-gnu/libmodplug.so.1.0.0
7ff5b14c2000-7ff5b14c3000 r--p 0004a000 08:06 397114                     /usr/lib/x86_64-linux-gnu/libmodplug.so.1.0.0
7ff5b14c3000-7ff5b14c4000 rw-p 0004b000 08:06 397114                     /usr/lib/x86_64-linux-gnu/libmodplug.so.1.0.0
7ff5b14c4000-7ff5b1603000 rw-p 00000000 00:00 0 
7ff5b1603000-7ff5b161e000 r-xp 00000000 08:06 402198                     /usr/lib/x86_64-linux-gnu/librtmp.so.1
7ff5b161e000-7ff5b181d000 ---p 0001b000 08:06 402198                     /usr/lib/x86_64-linux-gnu/librtmp.so.1
7ff5b181d000-7ff5b181e000 r--p 0001a000 08:06 402198                     /usr/lib/x86_64-linux-gnu/librtmp.so.1
7ff5b181e000-7ff5b181f000 rw-p 0001b000 08:06 402198                     /usr/lib/x86_64-linux-gnu/librtmp.so.1
7ff5b181f000-7ff5b1867000 r-xp 00000000 08:06 397299                     /usr/lib/x86_64-linux-gnu/libssh-gcrypt.so.4.4.1
7ff5b1867000-7ff5b1a67000 ---p 00048000 08:06 397299                     /usr/lib/x86_64-linux-gnu/libssh-gcrypt.so.4.4.1
7ff5b1a67000-7ff5b1a68000 r--p 00048000 08:06 397299                     /usr/lib/x86_64-linux-gnu/libssh-gcrypt.so.4.4.1
7ff5b1a68000-7ff5b1a69000 rw-p 00049000 08:06 397299                     /usr/lib/x86_64-linux-gnu/libssh-gcrypt.so.4.4.1
7ff5b1a69000-7ff5b1a83000 r-xp 00000000 08:06 393353                     /usr/lib/x86_64-linux-gnu/libcrystalhd.so.3.6
7ff5b1a83000-7ff5b1c82000 ---p 0001a000 08:06 393353                     /usr/lib/x86_64-linux-gnu/libcrystalhd.so.3.6
7ff5b1c82000-7ff5b1c83000 r--p 00019000 08:06 393353                     /usr/lib/x86_64-linux-gnu/libcrystalhd.so.3.6
7ff5b1c83000-7ff5b1c84000 rw-p 0001a000 08:06 393353                     /usr/lib/x86_64-linux-gnu/libcrystalhd.so.3.6
7ff5b1c84000-7ff5b1c91000 r-xp 00000000 08:06 393482                     /usr/lib/x86_64-linux-gnu/libgsm.so.1.0.12
7ff5b1c91000-7ff5b1e90000 ---p 0000d000 08:06 393482                     /usr/lib/x86_64-linux-gnu/libgsm.so.1.0.12
7ff5b1e90000-7ff5b1e91000 r--p 0000c000 08:06 393482                     /usr/lib/x86_64-linux-gnu/libgsm.so.1.0.12
7ff5b1e91000-7ff5b1e92000 rw-p 0000d000 08:06 393482                     /usr/lib/x86_64-linux-gnu/libgsm.so.1.0.12
7ff5b1e92000-7ff5b1ed7000 r-xp 00000000 08:06 393560                     /usr/lib/x86_64-linux-gnu/libmp3lame.so.0.0.0
7ff5b1ed7000-7ff5b20d7000 ---p 00045000 08:06 393560                     /usr/lib/x86_64-linux-gnu/libmp3lame.so.0.0.0
7ff5b20d7000-7ff5b20d8000 r--p 00045000 08:06 393560                     /usr/lib/x86_64-linux-gnu/libmp3lame.so.0.0.0
7ff5b20d8000-7ff5b20d9000 rw-p 00046000 08:06 393560                     /usr/lib/x86_64-linux-gnu/libmp3lame.so.0.0.0
7ff5b20d9000-7ff5b2107000 rw-p 00000000 00:00 0 
7ff5b2107000-7ff5b2129000 r-xp 00000000 08:06 393715                     /usr/lib/x86_64-linux-gnu/libopenjpeg.so.1.5.2
7ff5b2129000-7ff5b2328000 ---p 00022000 08:06 393715                     /usr/lib/x86_64-linux-gnu/libopenjpeg.so.1.5.2
7ff5b2328000-7ff5b2329000 r--p 00021000 08:06 393715                     /usr/lib/x86_64-linux-gnu/libopenjpeg.so.1.5.2
7ff5b2329000-7ff5b232a000 rw-p 00022000 08:06 393715                     /usr/lib/x86_64-linux-gnu/libopenjpeg.so.1.5.2
7ff5b232a000-7ff5b2373000 r-xp 00000000 08:06 402050                     /usr/lib/x86_64-linux-gnu/libopus.so.0.5.2
7ff5b2373000-7ff5b2572000 ---p 00049000 08:06 402050                     /usr/lib/x86_64-linux-gnu/libopus.so.0.5.2
7ff5b2572000-7ff5b2573000 r--p 00048000 08:06 402050                     /usr/lib/x86_64-linux-gnu/libopus.so.0.5.2
7ff5b2573000-7ff5b2574000 rw-p 00049000 08:06 402050                     /usr/lib/x86_64-linux-gnu/libopus.so.0.5.2
7ff5b2574000-7ff5b2645000 r-xp 00000000 08:06 393504                     /usr/lib/x86_64-linux-gnu/libschroedinger-1.0.so.0.11.0
7ff5b2645000-7ff5b2845000 ---p 000d1000 08:06 393504                     /usr/lib/x86_64-linux-gnu/libschroedinger-1.0.so.0.11.0
7ff5b2845000-7ff5b2847000 r--p 000d1000 08:06 393504                     /usr/lib/x86_64-linux-gnu/libschroedinger-1.0.so.0.11.0
7ff5b2847000-7ff5b2848000 rw-p 000d3000 08:06 393504                     /usr/lib/x86_64-linux-gnu/libschroedinger-1.0.so.0.11.0
7ff5b2848000-7ff5b2849000 rw-p 00000000 00:00 0 
7ff5b2849000-7ff5b2855000 r-xp 00000000 08:06 393718                     /usr/lib/x86_64-linux-gnu/libshine.so.3.0.1
7ff5b2855000-7ff5b2a54000 ---p 0000c000 08:06 393718                     /usr/lib/x86_64-linux-gnu/libshine.so.3.0.1
7ff5b2a54000-7ff5b2a55000 r--p 0000b000 08:06 393718                     /usr/lib/x86_64-linux-gnu/libshine.so.3.0.1
7ff5b2a55000-7ff5b2a56000 rw-p 0000c000 08:06 393718                     /usr/lib/x86_64-linux-gnu/libshine.so.3.0.1
7ff5b2a56000-7ff5b2a6d000 r-xp 00000000 08:06 402272                     /usr/lib/x86_64-linux-gnu/libspeex.so.1.5.0
7ff5b2a6d000-7ff5b2c6d000 ---p 00017000 08:06 402272                     /usr/lib/x86_64-linux-gnu/libspeex.so.1.5.0
7ff5b2c6d000-7ff5b2c6e000 r--p 00017000 08:06 402272                     /usr/lib/x86_64-linux-gnu/libspeex.so.1.5.0
7ff5b2c6e000-7ff5b2c6f000 rw-p 00018000 08:06 402272                     /usr/lib/x86_64-linux-gnu/libspeex.so.1.5.0
7ff5b2c6f000-7ff5b2c88000 r-xp 00000000 08:06 402311                     /usr/lib/x86_64-linux-gnu/libtheoradec.so.1.1.4
7ff5b2c88000-7ff5b2e87000 ---p 00019000 08:06 402311                     /usr/lib/x86_64-linux-gnu/libtheoradec.so.1.1.4
7ff5b2e87000-7ff5b2e88000 r--p 00018000 08:06 402311                     /usr/lib/x86_64-linux-gnu/libtheoradec.so.1.1.4
7ff5b2e88000-7ff5b2e89000 rw-p 00019000 08:06 402311                     /usr/lib/x86_64-linux-gnu/libtheoradec.so.1.1.4
7ff5b2e89000-7ff5b2ec7000 r-xp 00000000 08:06 402313                     /usr/lib/x86_64-linux-gnu/libtheoraenc.so.1.1.2
7ff5b2ec7000-7ff5b30c6000 ---p 0003e000 08:06 402313                     /usr/lib/x86_64-linux-gnu/libtheoraenc.so.1.1.2
7ff5b30c6000-7ff5b30c7000 r--p 0003d000 08:06 402313                     /usr/lib/x86_64-linux-gnu/libtheoraenc.so.1.1.2
7ff5b30c7000-7ff5b30c8000 rw-p 0003e000 08:06 402313                     /usr/lib/x86_64-linux-gnu/libtheoraenc.so.1.1.2
7ff5b30c8000-7ff5b30e6000 r-xp 00000000 08:06 393727                     /usr/lib/x86_64-linux-gnu/libtwolame.so.0.0.0
7ff5b30e6000-7ff5b32e5000 ---p 0001e000 08:06 393727                     /usr/lib/x86_64-linux-gnu/libtwolame.so.0.0.0
7ff5b32e5000-7ff5b32e6000 r--p 0001d000 08:06 393727                     /usr/lib/x86_64-linux-gnu/libtwolame.so.0.0.0
7ff5b32e6000-7ff5b32e7000 rw-p 0001e000 08:06 393727                     /usr/lib/x86_64-linux-gnu/libtwolame.so.0.0.0
7ff5b32e7000-7ff5b32eb000 rw-p 00000000 00:00 0 
7ff5b32eb000-7ff5b3315000 r-xp 00000000 08:06 402410                     /usr/lib/x86_64-linux-gnu/libvorbis.so.0.4.8
7ff5b3315000-7ff5b3514000 ---p 0002a000 08:06 402410                     /usr/lib/x86_64-linux-gnu/libvorbis.so.0.4.8
7ff5b3514000-7ff5b3515000 r--p 00029000 08:06 402410                     /usr/lib/x86_64-linux-gnu/libvorbis.so.0.4.8
7ff5b3515000-7ff5b3516000 rw-p 0002a000 08:06 402410                     /usr/lib/x86_64-linux-gnu/libvorbis.so.0.4.8
7ff5b3516000-7ff5b35a3000 r-xp 00000000 08:06 402412                     /usr/lib/x86_64-linux-gnu/libvorbisenc.so.2.0.11
7ff5b35a3000-7ff5b37a2000 ---p 0008d000 08:06 402412                     /usr/lib/x86_64-linux-gnu/libvorbisenc.so.2.0.11
7ff5b37a2000-7ff5b37be000 r--p 0008c000 08:06 402412                     /usr/lib/x86_64-linux-gnu/libvorbisenc.so.2.0.11
7ff5b37be000-7ff5b37bf000 rw-p 000a8000 08:06 402412                     /usr/lib/x86_64-linux-gnu/libvorbisenc.so.2.0.11
7ff5b37bf000-7ff5b39de000 r-xp 00000000 08:06 398679                     /usr/lib/x86_64-linux-gnu/libvpx.so.3.0.0
7ff5b39de000-7ff5b3bdd000 ---p 0021f000 08:06 398679                     /usr/lib/x86_64-linux-gnu/libvpx.so.3.0.0
7ff5b3bdd000-7ff5b3bdf000 r--p 0021e000 08:06 398679                     /usr/lib/x86_64-linux-gnu/libvpx.so.3.0.0
7ff5b3bdf000-7ff5b3be0000 rw-p 00220000 08:06 398679                     /usr/lib/x86_64-linux-gnu/libvpx.so.3.0.0
7ff5b3be0000-7ff5b3be3000 rw-p 00000000 00:00 0 
7ff5b3be3000-7ff5b3c0b000 r-xp 00000000 08:06 402423                     /usr/lib/x86_64-linux-gnu/libwavpack.so.1.1.7
7ff5b3c0b000-7ff5b3e0a000 ---p 00028000 08:06 402423                     /usr/lib/x86_64-linux-gnu/libwavpack.so.1.1.7
7ff5b3e0a000-7ff5b3e0b000 r--p 00027000 08:06 402423                     /usr/lib/x86_64-linux-gnu/libwavpack.so.1.1.7
7ff5b3e0b000-7ff5b3e0c000 rw-p 00028000 08:06 402423                     /usr/lib/x86_64-linux-gnu/libwavpack.so.1.1.7
7ff5b3e0c000-7ff5b3f34000 r-xp 00000000 08:06 394024                     /usr/lib/x86_64-linux-gnu/libx264.so.148
7ff5b3f34000-7ff5b4133000 ---p 00128000 08:06 394024                     /usr/lib/x86_64-linux-gnu/libx264.so.148
7ff5b4133000-7ff5b4134000 r--p 00127000 08:06 394024                     /usr/lib/x86_64-linux-gnu/libx264.so.148
7ff5b4134000-7ff5b4135000 rw-p 00128000 08:06 394024                     /usr/lib/x86_64-linux-gnu/libx264.so.148
7ff5b4135000-7ff5b41b0000 rw-p 00000000 00:00 0 
7ff5b41b0000-7ff5b4bbd000 r-xp 00000000 08:06 394025                     /usr/lib/x86_64-linux-gnu/libx265.so.79
7ff5b4bbd000-7ff5b4dbc000 ---p 00a0d000 08:06 394025                     /usr/lib/x86_64-linux-gnu/libx265.so.79
7ff5b4dbc000-7ff5b4dbf000 r--p 00a0c000 08:06 394025                     /usr/lib/x86_64-linux-gnu/libx265.so.79
7ff5b4dbf000-7ff5b4dc2000 rw-p 00a0f000 08:06 394025                     /usr/lib/x86_64-linux-gnu/libx265.so.79
7ff5b4dc2000-7ff5b4dcf000 rw-p 00000000 00:00 0 
7ff5b4dcf000-7ff5b4e70000 r-xp 00000000 08:06 394026                     /usr/lib/x86_64-linux-gnu/libxvidcore.so.4.3
7ff5b4e70000-7ff5b506f000 ---p 000a1000 08:06 394026                     /usr/lib/x86_64-linux-gnu/libxvidcore.so.4.3
7ff5b506f000-7ff5b5070000 r--p 000a0000 08:06 394026                     /usr/lib/x86_64-linux-gnu/libxvidcore.so.4.3
7ff5b5070000-7ff5b507a000 rw-p 000a1000 08:06 394026                     /usr/lib/x86_64-linux-gnu/libxvidcore.so.4.3
7ff5b507a000-7ff5b50e3000 rw-p 00000000 00:00 0 
7ff5b50e3000-7ff5b515a000 r-xp 00000000 08:06 394102                     /usr/lib/x86_64-linux-gnu/libzvbi.so.0.13.2
7ff5b515a000-7ff5b5359000 ---p 00077000 08:06 394102                     /usr/lib/x86_64-linux-gnu/libzvbi.so.0.13.2
7ff5b5359000-7ff5b5362000 r--p 00076000 08:06 394102                     /usr/lib/x86_64-linux-gnu/libzvbi.so.0.13.2
7ff5b5362000-7ff5b536e000 rw-p 0007f000 08:06 394102                     /usr/lib/x86_64-linux-gnu/libzvbi.so.0.13.2
7ff5b536e000-7ff5b5389000 r-xp 00000000 08:06 393729                     /usr/lib/x86_64-linux-gnu/libva.so.1.3900.0
7ff5b5389000-7ff5b5588000 ---p 0001b000 08:06 393729                     /usr/lib/x86_64-linux-gnu/libva.so.1.3900.0
7ff5b5588000-7ff5b5589000 r--p 0001a000 08:06 393729                     /usr/lib/x86_64-linux-gnu/libva.so.1.3900.0
7ff5b5589000-7ff5b558a000 rw-p 0001b000 08:06 393729                     /usr/lib/x86_64-linux-gnu/libva.so.1.3900.0
7ff5b558a000-7ff5b55a4000 r-xp 00000000 08:06 396931                     /usr/lib/x86_64-linux-gnu/libswresample-ffmpeg.so.1.2.101
7ff5b55a4000-7ff5b57a4000 ---p 0001a000 08:06 396931                     /usr/lib/x86_64-linux-gnu/libswresample-ffmpeg.so.1.2.101
7ff5b57a4000-7ff5b57a6000 r--p 0001a000 08:06 396931                     /usr/lib/x86_64-linux-gnu/libswresample-ffmpeg.so.1.2.101
7ff5b57a6000-7ff5b57a7000 rw-p 0001c000 08:06 396931                     /usr/lib/x86_64-linux-gnu/libswresample-ffmpeg.so.1.2.101
7ff5b57a7000-7ff5b57be000 r-xp 00000000 08:06 16651001                   /lib/x86_64-linux-gnu/libusb-1.0.so.0.1.0
7ff5b57be000-7ff5b59bd000 ---p 00017000 08:06 16651001                   /lib/x86_64-linux-gnu/libusb-1.0.so.0.1.0
7ff5b59bd000-7ff5b59be000 r--p 00016000 08:06 16651001                   /lib/x86_64-linux-gnu/libusb-1.0.so.0.1.0
7ff5b59be000-7ff5b59bf000 rw-p 00017000 08:06 16651001                   /lib/x86_64-linux-gnu/libusb-1.0.so.0.1.0
7ff5b59bf000-7ff5b59cc000 r-xp 00000000 08:06 402170                     /usr/lib/x86_64-linux-gnu/libraw1394.so.11.1.0
7ff5b59cc000-7ff5b5bcc000 ---p 0000d000 08:06 402170                     /usr/lib/x86_64-linux-gnu/libraw1394.so.11.1.0
7ff5b5bcc000-7ff5b5bcd000 r--p 0000d000 08:06 402170                     /usr/lib/x86_64-linux-gnu/libraw1394.so.11.1.0
7ff5b5bcd000-7ff5b5bce000 rw-p 0000e000 08:06 402170                     /usr/lib/x86_64-linux-gnu/libraw1394.so.11.1.0
7ff5b5bce000-7ff5b5c32000 r-xp 00000000 08:06 402090                     /usr/lib/x86_64-linux-gnu/libpcre16.so.3.13.2
7ff5b5c32000-7ff5b5e32000 ---p 00064000 08:06 402090                     /usr/lib/x86_64-linux-gnu/libpcre16.so.3.13.2
7ff5b5e32000-7ff5b5e33000 r--p 00064000 08:06 402090                     /usr/lib/x86_64-linux-gnu/libpcre16.so.3.13.2
7ff5b5e33000-7ff5b5e34000 rw-p 00065000 08:06 402090                     /usr/lib/x86_64-linux-gnu/libpcre16.so.3.13.2
7ff5b5e34000-7ff5b5e90000 r-xp 00000000 08:06 401766                     /usr/lib/x86_64-linux-gnu/libharfbuzz.so.0.10000.1
7ff5b5e90000-7ff5b6090000 ---p 0005c000 08:06 401766                     /usr/lib/x86_64-linux-gnu/libharfbuzz.so.0.10000.1
7ff5b6090000-7ff5b6091000 r--p 0005c000 08:06 401766                     /usr/lib/x86_64-linux-gnu/libharfbuzz.so.0.10000.1
7ff5b6091000-7ff5b6092000 rw-p 0005d000 08:06 401766                     /usr/lib/x86_64-linux-gnu/libharfbuzz.so.0.10000.1
7ff5b6092000-7ff5b61a1000 r-xp 00000000 08:06 16677693                   /lib/x86_64-linux-gnu/libglib-2.0.so.0.4800.2
7ff5b61a1000-7ff5b63a0000 ---p 0010f000 08:06 16677693                   /lib/x86_64-linux-gnu/libglib-2.0.so.0.4800.2
7ff5b63a0000-7ff5b63a1000 r--p 0010e000 08:06 16677693                   /lib/x86_64-linux-gnu/libglib-2.0.so.0.4800.2
7ff5b63a1000-7ff5b63a2000 rw-p 0010f000 08:06 16677693                   /lib/x86_64-linux-gnu/libglib-2.0.so.0.4800.2
7ff5b63a2000-7ff5b63a3000 rw-p 00000000 00:00 0 
7ff5b63a3000-7ff5b63f5000 r-xp 00000000 08:06 395713                     /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0.4800.2
7ff5b63f5000-7ff5b65f4000 ---p 00052000 08:06 395713                     /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0.4800.2
7ff5b65f4000-7ff5b65f5000 r--p 00051000 08:06 395713                     /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0.4800.2
7ff5b65f5000-7ff5b65f6000 rw-p 00052000 08:06 395713                     /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0.4800.2
7ff5b65f6000-7ff5b65fd000 r-xp 00000000 08:06 400025                     /usr/lib/x86_64-linux-gnu/libaec.so.0.0.3
7ff5b65fd000-7ff5b67fc000 ---p 00007000 08:06 400025                     /usr/lib/x86_64-linux-gnu/libaec.so.0.0.3
7ff5b67fc000-7ff5b67fd000 r--p 00006000 08:06 400025                     /usr/lib/x86_64-linux-gnu/libaec.so.0.0.3
7ff5b67fd000-7ff5b67fe000 rw-p 00007000 08:06 400025                     /usr/lib/x86_64-linux-gnu/libaec.so.0.0.3
7ff5b67fe000-7ff5b80b4000 r-xp 00000000 08:06 394645                     /usr/lib/x86_64-linux-gnu/libicudata.so.55.1
7ff5b80b4000-7ff5b82b3000 ---p 018b6000 08:06 394645                     /usr/lib/x86_64-linux-gnu/libicudata.so.55.1
7ff5b82b3000-7ff5b82b4000 r--p 018b5000 08:06 394645                     /usr/lib/x86_64-linux-gnu/libicudata.so.55.1
7ff5b82b4000-7ff5b82b5000 rw-p 018b6000 08:06 394645                     /usr/lib/x86_64-linux-gnu/libicudata.so.55.1
7ff5b82b5000-7ff5b85f1000 r-xp 00000000 08:06 396145                     /usr/lib/x86_64-linux-gnu/libmysqlclient.so.20.3.18
7ff5b85f1000-7ff5b87f0000 ---p 0033c000 08:06 396145                     /usr/lib/x86_64-linux-gnu/libmysqlclient.so.20.3.18
7ff5b87f0000-7ff5b87f4000 r--p 0033b000 08:06 396145                     /usr/lib/x86_64-linux-gnu/libmysqlclient.so.20.3.18
7ff5b87f4000-7ff5b8866000 rw-p 0033f000 08:06 396145                     /usr/lib/x86_64-linux-gnu/libmysqlclient.so.20.3.18
7ff5b8866000-7ff5b886c000 rw-p 00000000 00:00 0 
7ff5b886c000-7ff5b8a1d000 r-xp 00000000 08:06 395446                     /usr/lib/x86_64-linux-gnu/libxml2.so.2.9.3
7ff5b8a1d000-7ff5b8c1c000 ---p 001b1000 08:06 395446                     /usr/lib/x86_64-linux-gnu/libxml2.so.2.9.3
7ff5b8c1c000-7ff5b8c24000 r--p 001b0000 08:06 395446                     /usr/lib/x86_64-linux-gnu/libxml2.so.2.9.3
7ff5b8c24000-7ff5b8c26000 rw-p 001b8000 08:06 395446                     /usr/lib/x86_64-linux-gnu/libxml2.so.2.9.3
7ff5b8c26000-7ff5b8c27000 rw-p 00000000 00:00 0 
7ff5b8c27000-7ff5b8c91000 r-xp 00000000 08:06 393605                     /usr/lib/x86_64-linux-gnu/libcurl-gnutls.so.4.4.0
7ff5b8c91000-7ff5b8e90000 ---p 0006a000 08:06 393605                     /usr/lib/x86_64-linux-gnu/libcurl-gnutls.so.4.4.0
7ff5b8e90000-7ff5b8e93000 r--p 00069000 08:06 393605                     /usr/lib/x86_64-linux-gnu/libcurl-gnutls.so.4.4.0
7ff5b8e93000-7ff5b8e94000 rw-p 0006c000 08:06 393605                     /usr/lib/x86_64-linux-gnu/libcurl-gnutls.so.4.4.0
7ff5b8e94000-7ff5b8f02000 r-xp 00000000 08:06 16650950                   /lib/x86_64-linux-gnu/libpcre.so.3.13.2
7ff5b8f02000-7ff5b9102000 ---p 0006e000 08:06 16650950                   /lib/x86_64-linux-gnu/libpcre.so.3.13.2
7ff5b9102000-7ff5b9103000 r--p 0006e000 08:06 16650950                   /lib/x86_64-linux-gnu/libpcre.so.3.13.2
7ff5b9103000-7ff5b9104000 rw-p 0006f000 08:06 16650950                   /lib/x86_64-linux-gnu/libpcre.so.3.13.2
7ff5b9104000-7ff5b9683000 r-xp 00000000 08:06 396163                     /usr/lib/x86_64-linux-gnu/libspatialite.so.7.1.0
7ff5b9683000-7ff5b9882000 ---p 0057f000 08:06 396163                     /usr/lib/x86_64-linux-gnu/libspatialite.so.7.1.0
7ff5b9882000-7ff5b9883000 r--p 0057e000 08:06 396163                     /usr/lib/x86_64-linux-gnu/libspatialite.so.7.1.0
7ff5b9883000-7ff5b9885000 rw-p 0057f000 08:06 396163                     /usr/lib/x86_64-linux-gnu/libspatialite.so.7.1.0
7ff5b9885000-7ff5b9887000 rw-p 00000000 00:00 0 
7ff5b9887000-7ff5b99fc000 r-xp 00000000 08:06 394460                     /usr/lib/x86_64-linux-gnu/libdap.so.17.3.1
7ff5b99fc000-7ff5b9bfc000 ---p 00175000 08:06 394460                     /usr/lib/x86_64-linux-gnu/libdap.so.17.3.1
7ff5b9bfc000-7ff5b9c03000 r--p 00175000 08:06 394460                     /usr/lib/x86_64-linux-gnu/libdap.so.17.3.1
7ff5b9c03000-7ff5b9c04000 rw-p 0017c000 08:06 394460                     /usr/lib/x86_64-linux-gnu/libdap.so.17.3.1
7ff5b9c04000-7ff5b9c05000 rw-p 00000000 00:00 0 
7ff5b9c05000-7ff5b9c42000 r-xp 00000000 08:06 396083                     /usr/lib/x86_64-linux-gnu/libdapclient.so.6.1.1
7ff5b9c42000-7ff5b9e41000 ---p 0003d000 08:06 396083                     /usr/lib/x86_64-linux-gnu/libdapclient.so.6.1.1
7ff5b9e41000-7ff5b9e43000 r--p 0003c000 08:06 396083                     /usr/lib/x86_64-linux-gnu/libdapclient.so.6.1.1
7ff5b9e43000-7ff5b9e44000 rw-p 0003e000 08:06 396083                     /usr/lib/x86_64-linux-gnu/libdapclient.so.6.1.1
7ff5b9e44000-7ff5b9e71000 r-xp 00000000 08:06 393374                     /usr/lib/x86_64-linux-gnu/libpq.so.5.8
7ff5b9e71000-7ff5ba070000 ---p 0002d000 08:06 393374                     /usr/lib/x86_64-linux-gnu/libpq.so.5.8
7ff5ba070000-7ff5ba073000 r--p 0002c000 08:06 393374                     /usr/lib/x86_64-linux-gnu/libpq.so.5.8
7ff5ba073000-7ff5ba074000 rw-p 0002f000 08:06 393374                     /usr/lib/x86_64-linux-gnu/libpq.so.5.8
7ff5ba074000-7ff5ba07c000 r-xp 00000000 08:06 399886                     /usr/lib/x86_64-linux-gnu/libgif.so.7.0.0
7ff5ba07c000-7ff5ba27b000 ---p 00008000 08:06 399886                     /usr/lib/x86_64-linux-gnu/libgif.so.7.0.0
7ff5ba27b000-7ff5ba27c000 r--p 00007000 08:06 399886                     /usr/lib/x86_64-linux-gnu/libgif.so.7.0.0
7ff5ba27c000-7ff5ba27d000 rw-p 00008000 08:06 399886                     /usr/lib/x86_64-linux-gnu/libgif.so.7.0.0
7ff5ba27d000-7ff5ba29d000 r-xp 00000000 08:06 396157                     /usr/lib/libogdi.so.3.2
7ff5ba29d000-7ff5ba49c000 ---p 00020000 08:06 396157                     /usr/lib/libogdi.so.3.2
7ff5ba49c000-7ff5ba49d000 r--p 0001f000 08:06 396157                     /usr/lib/libogdi.so.3.2
7ff5ba49d000-7ff5ba49e000 rw-p 00020000 08:06 396157                     /usr/lib/libogdi.so.3.2
7ff5ba49e000-7ff5ba519000 r-xp 00000000 08:06 396132                     /usr/lib/libdfalt.so.0.0.0
7ff5ba519000-7ff5ba718000 ---p 0007b000 08:06 396132                     /usr/lib/libdfalt.so.0.0.0
7ff5ba718000-7ff5ba71a000 r--p 0007a000 08:06 396132                     /usr/lib/libdfalt.so.0.0.0
7ff5ba71a000-7ff5ba71b000 rw-p 0007c000 08:06 396132                     /usr/lib/libdfalt.so.0.0.0
7ff5ba71b000-7ff5ba744000 rw-p 00000000 00:00 0 
7ff5ba744000-7ff5ba767000 r-xp 00000000 08:06 396131                     /usr/lib/libmfhdfalt.so.0.0.0
7ff5ba767000-7ff5ba966000 ---p 00023000 08:06 396131                     /usr/lib/libmfhdfalt.so.0.0.0
7ff5ba966000-7ff5ba967000 r--p 00022000 08:06 396131                     /usr/lib/libmfhdfalt.so.0.0.0
7ff5ba967000-7ff5ba968000 rw-p 00023000 08:06 396131                     /usr/lib/libmfhdfalt.so.0.0.0
7ff5ba968000-7ff5ba969000 rw-p 00000000 00:00 0 
7ff5ba969000-7ff5baa6d000 r-xp 00000000 08:06 396147                     /usr/lib/x86_64-linux-gnu/libnetcdf.so.11.0.0
7ff5baa6d000-7ff5bac6d000 ---p 00104000 08:06 396147                     /usr/lib/x86_64-linux-gnu/libnetcdf.so.11.0.0
7ff5bac6d000-7ff5bacbb000 r--p 00104000 08:06 396147                     /usr/lib/x86_64-linux-gnu/libnetcdf.so.11.0.0
7ff5bacbb000-7ff5bacbd000 rw-p 00152000 08:06 396147                     /usr/lib/x86_64-linux-gnu/libnetcdf.so.11.0.0
7ff5bacbd000-7ff5bdccc000 rw-p 00000000 00:00 0 
7ff5bdccc000-7ff5bdd16000 r-xp 00000000 08:06 401858                     /usr/lib/x86_64-linux-gnu/libjasper.so.1.0.0
7ff5bdd16000-7ff5bdf15000 ---p 0004a000 08:06 401858                     /usr/lib/x86_64-linux-gnu/libjasper.so.1.0.0
7ff5bdf15000-7ff5bdf16000 r--p 00049000 08:06 401858                     /usr/lib/x86_64-linux-gnu/libjasper.so.1.0.0
7ff5bdf16000-7ff5bdf1a000 rw-p 0004a000 08:06 401858                     /usr/lib/x86_64-linux-gnu/libjasper.so.1.0.0
7ff5bdf1a000-7ff5bdf21000 rw-p 00000000 00:00 0 
7ff5bdf21000-7ff5bdf77000 r-xp 00000000 08:06 397482                     /usr/lib/x86_64-linux-gnu/libopenjp2.so.2.3.0
7ff5bdf77000-7ff5be177000 ---p 00056000 08:06 397482                     /usr/lib/x86_64-linux-gnu/libopenjp2.so.2.3.0
7ff5be177000-7ff5be178000 r--p 00056000 08:06 397482                     /usr/lib/x86_64-linux-gnu/libopenjp2.so.2.3.0
7ff5be178000-7ff5be179000 rw-p 00057000 08:06 397482                     /usr/lib/x86_64-linux-gnu/libopenjp2.so.2.3.0
7ff5be179000-7ff5be4ba000 r-xp 00000000 08:06 396165                     /usr/lib/x86_64-linux-gnu/libxerces-c-3.1.so
7ff5be4ba000-7ff5be6ba000 ---p 00341000 08:06 396165                     /usr/lib/x86_64-linux-gnu/libxerces-c-3.1.so
7ff5be6ba000-7ff5be6cc000 r--p 00341000 08:06 396165                     /usr/lib/x86_64-linux-gnu/libxerces-c-3.1.so
7ff5be6cc000-7ff5be6ee000 rw-p 00353000 08:06 396165                     /usr/lib/x86_64-linux-gnu/libxerces-c-3.1.so
7ff5be6ee000-7ff5be726000 r-xp 00000000 08:06 396143                     /usr/lib/x86_64-linux-gnu/libkmlengine.so.1.3.0
7ff5be726000-7ff5be926000 ---p 00038000 08:06 396143                     /usr/lib/x86_64-linux-gnu/libkmlengine.so.1.3.0
7ff5be926000-7ff5be927000 r--p 00038000 08:06 396143                     /usr/lib/x86_64-linux-gnu/libkmlengine.so.1.3.0
7ff5be927000-7ff5be928000 rw-p 00039000 08:06 396143                     /usr/lib/x86_64-linux-gnu/libkmlengine.so.1.3.0
7ff5be928000-7ff5be9d9000 r-xp 00000000 08:06 396141                     /usr/lib/x86_64-linux-gnu/libkmldom.so.1.3.0
7ff5be9d9000-7ff5bebd9000 ---p 000b1000 08:06 396141                     /usr/lib/x86_64-linux-gnu/libkmldom.so.1.3.0
7ff5bebd9000-7ff5bebdf000 r--p 000b1000 08:06 396141                     /usr/lib/x86_64-linux-gnu/libkmldom.so.1.3.0
7ff5bebdf000-7ff5bebe1000 rw-p 000b7000 08:06 396141                     /usr/lib/x86_64-linux-gnu/libkmldom.so.1.3.0
7ff5bebe1000-7ff5bebfa000 r-xp 00000000 08:06 396139                     /usr/lib/x86_64-linux-gnu/libkmlbase.so.1.3.0
7ff5bebfa000-7ff5bedf9000 ---p 00019000 08:06 396139                     /usr/lib/x86_64-linux-gnu/libkmlbase.so.1.3.0
7ff5bedf9000-7ff5bedfa000 r--p 00018000 08:06 396139                     /usr/lib/x86_64-linux-gnu/libkmlbase.so.1.3.0
7ff5bedfa000-7ff5bedfb000 rw-p 00019000 08:06 396139                     /usr/lib/x86_64-linux-gnu/libkmlbase.so.1.3.0
7ff5bedfb000-7ff5bee0c000 r-xp 00000000 08:06 396168                     /usr/lib/x86_64-linux-gnu/libodbcinst.so.2.0.0
7ff5bee0c000-7ff5bf00b000 ---p 00011000 08:06 396168                     /usr/lib/x86_64-linux-gnu/libodbcinst.so.2.0.0
7ff5bf00b000-7ff5bf00c000 r--p 00010000 08:06 396168                     /usr/lib/x86_64-linux-gnu/libodbcinst.so.2.0.0
7ff5bf00c000-7ff5bf00d000 rw-p 00011000 08:06 396168                     /usr/lib/x86_64-linux-gnu/libodbcinst.so.2.0.0
7ff5bf00d000-7ff5bf06e000 r-xp 00000000 08:06 396149                     /usr/lib/x86_64-linux-gnu/libodbc.so.2.0.0
7ff5bf06e000-7ff5bf26d000 ---p 00061000 08:06 396149                     /usr/lib/x86_64-linux-gnu/libodbc.so.2.0.0
7ff5bf26d000-7ff5bf26e000 r--p 00060000 08:06 396149                     /usr/lib/x86_64-linux-gnu/libodbc.so.2.0.0
7ff5bf26e000-7ff5bf275000 rw-p 00061000 08:06 396149                     /usr/lib/x86_64-linux-gnu/libodbc.so.2.0.0
7ff5bf275000-7ff5bf276000 rw-p 00000000 00:00 0 
7ff5bf276000-7ff5bf289000 r-xp 00000000 08:06 396085                     /usr/lib/x86_64-linux-gnu/libepsilon.so.1.0.0
7ff5bf289000-7ff5bf488000 ---p 00013000 08:06 396085                     /usr/lib/x86_64-linux-gnu/libepsilon.so.1.0.0
7ff5bf488000-7ff5bf489000 r--p 00012000 08:06 396085                     /usr/lib/x86_64-linux-gnu/libepsilon.so.1.0.0
7ff5bf489000-7ff5bf48e000 rw-p 00013000 08:06 396085                     /usr/lib/x86_64-linux-gnu/libepsilon.so.1.0.0
7ff5bf48e000-7ff5bf4ba000 r-xp 00000000 08:06 396090                     /usr/lib/x86_64-linux-gnu/libgeos_c.so.1.9.0
7ff5bf4ba000-7ff5bf6ba000 ---p 0002c000 08:06 396090                     /usr/lib/x86_64-linux-gnu/libgeos_c.so.1.9.0
7ff5bf6ba000-7ff5bf6bb000 r--p 0002c000 08:06 396090                     /usr/lib/x86_64-linux-gnu/libgeos_c.so.1.9.0
7ff5bf6bb000-7ff5bf6bc000 rw-p 0002d000 08:06 396090                     /usr/lib/x86_64-linux-gnu/libgeos_c.so.1.9.0
7ff5bf6bc000-7ff5bf6c5000 r-xp 00000000 08:06 396087                     /usr/lib/x86_64-linux-gnu/libfreexl.so.1.1.0
7ff5bf6c5000-7ff5bf8c4000 ---p 00009000 08:06 396087                     /usr/lib/x86_64-linux-gnu/libfreexl.so.1.1.0
7ff5bf8c4000-7ff5bf8c5000 r--p 00008000 08:06 396087                     /usr/lib/x86_64-linux-gnu/libfreexl.so.1.1.0
7ff5bf8c5000-7ff5bf8c6000 rw-p 00009000 08:06 396087                     /usr/lib/x86_64-linux-gnu/libfreexl.so.1.1.0
7ff5bf8c6000-7ff5bfb01000 r-xp 00000000 08:06 395731                     /usr/lib/x86_64-linux-gnu/libpoppler.so.58.0.0
7ff5bfb01000-7ff5bfd01000 ---p 0023b000 08:06 395731                     /usr/lib/x86_64-linux-gnu/libpoppler.so.58.0.0
7ff5bfd01000-7ff5bfd1e000 r--p 0023b000 08:06 395731                     /usr/lib/x86_64-linux-gnu/libpoppler.so.58.0.0
7ff5bfd1e000-7ff5bfd45000 rw-p 00258000 08:06 395731                     /usr/lib/x86_64-linux-gnu/libpoppler.so.58.0.0
7ff5bfd45000-7ff5bfd9c000 r-xp 00000000 08:06 396155                     /usr/lib/x86_64-linux-gnu/libproj.so.9.1.0
7ff5bfd9c000-7ff5bff9c000 ---p 00057000 08:06 396155                     /usr/lib/x86_64-linux-gnu/libproj.so.9.1.0
7ff5bff9c000-7ff5bff9d000 r--p 00057000 08:06 396155                     /usr/lib/x86_64-linux-gnu/libproj.so.9.1.0
7ff5bff9d000-7ff5bffa0000 rw-p 00058000 08:06 396155                     /usr/lib/x86_64-linux-gnu/libproj.so.9.1.0
7ff5bffa0000-7ff5bffa7000 r-xp 00000000 08:06 393354                     /usr/lib/libarmadillo.so.6.500.5
7ff5bffa7000-7ff5c01a6000 ---p 00007000 08:06 393354                     /usr/lib/libarmadillo.so.6.500.5
7ff5c01a6000-7ff5c01a7000 r--p 00006000 08:06 393354                     /usr/lib/libarmadillo.so.6.500.5
7ff5c01a7000-7ff5c01a8000 rw-p 00007000 08:06 393354                     /usr/lib/libarmadillo.so.6.500.5
7ff5c01a8000-7ff5c01ad000 r-xp 00000000 08:06 400987                     /usr/lib/x86_64-linux-gnu/libIlmThread-2_2.so.12.0.0
7ff5c01ad000-7ff5c03ad000 ---p 00005000 08:06 400987                     /usr/lib/x86_64-linux-gnu/libIlmThread-2_2.so.12.0.0
7ff5c03ad000-7ff5c03ae000 r--p 00005000 08:06 400987                     /usr/lib/x86_64-linux-gnu/libIlmThread-2_2.so.12.0.0
7ff5c03ae000-7ff5c03af000 rw-p 00006000 08:06 400987                     /usr/lib/x86_64-linux-gnu/libIlmThread-2_2.so.12.0.0
7ff5c03af000-7ff5c03ca000 r-xp 00000000 08:06 400979                     /usr/lib/x86_64-linux-gnu/libIex-2_2.so.12.0.0
7ff5c03ca000-7ff5c05c9000 ---p 0001b000 08:06 400979                     /usr/lib/x86_64-linux-gnu/libIex-2_2.so.12.0.0
7ff5c05c9000-7ff5c05cc000 r--p 0001a000 08:06 400979                     /usr/lib/x86_64-linux-gnu/libIex-2_2.so.12.0.0
7ff5c05cc000-7ff5c05cd000 rw-p 0001d000 08:06 400979                     /usr/lib/x86_64-linux-gnu/libIex-2_2.so.12.0.0
7ff5c05cd000-7ff5c05d8000 r-xp 00000000 08:06 401861                     /usr/lib/x86_64-linux-gnu/libjbig.so.0
7ff5c05d8000-7ff5c07d7000 ---p 0000b000 08:06 401861                     /usr/lib/x86_64-linux-gnu/libjbig.so.0
7ff5c07d7000-7ff5c07d8000 r--p 0000a000 08:06 401861                     /usr/lib/x86_64-linux-gnu/libjbig.so.0
7ff5c07d8000-7ff5c07db000 rw-p 0000b000 08:06 401861                     /usr/lib/x86_64-linux-gnu/libjbig.so.0
7ff5c07db000-7ff5c07fc000 r-xp 00000000 08:06 16650888                   /lib/x86_64-linux-gnu/liblzma.so.5.0.0
7ff5c07fc000-7ff5c09fb000 ---p 00021000 08:06 16650888                   /lib/x86_64-linux-gnu/liblzma.so.5.0.0
7ff5c09fb000-7ff5c09fc000 r--p 00020000 08:06 16650888                   /lib/x86_64-linux-gnu/liblzma.so.5.0.0
7ff5c09fc000-7ff5c09fd000 rw-p 00021000 08:06 16650888                   /lib/x86_64-linux-gnu/liblzma.so.5.0.0
7ff5c09fd000-7ff5c0a0d000 r-xp 00000000 08:06 401425                     /usr/lib/x86_64-linux-gnu/libdrm.so.2.4.0
7ff5c0a0d000-7ff5c0c0d000 ---p 00010000 08:06 401425                     /usr/lib/x86_64-linux-gnu/libdrm.so.2.4.0
7ff5c0c0d000-7ff5c0c0e000 r--p 00010000 08:06 401425                     /usr/lib/x86_64-linux-gnu/libdrm.so.2.4.0
7ff5c0c0e000-7ff5c0c0f000 rw-p 00011000 08:06 401425                     /usr/lib/x86_64-linux-gnu/libdrm.so.2.4.0
7ff5c0c0f000-7ff5c0c14000 r-xp 00000000 08:06 401164                     /usr/lib/x86_64-linux-gnu/libXxf86vm.so.1.0.0
7ff5c0c14000-7ff5c0e13000 ---p 00005000 08:06 401164                     /usr/lib/x86_64-linux-gnu/libXxf86vm.so.1.0.0
7ff5c0e13000-7ff5c0e14000 r--p 00004000 08:06 401164                     /usr/lib/x86_64-linux-gnu/libXxf86vm.so.1.0.0
7ff5c0e14000-7ff5c0e15000 rw-p 00005000 08:06 401164                     /usr/lib/x86_64-linux-gnu/libXxf86vm.so.1.0.0
7ff5c0e15000-7ff5c0e36000 r-xp 00000000 08:06 402537                     /usr/lib/x86_64-linux-gnu/libxcb.so.1.1.0
7ff5c0e36000-7ff5c1035000 ---p 00021000 08:06 402537                     /usr/lib/x86_64-linux-gnu/libxcb.so.1.1.0
7ff5c1035000-7ff5c1036000 r--p 00020000 08:06 402537                     /usr/lib/x86_64-linux-gnu/libxcb.so.1.1.0
7ff5c1036000-7ff5c1037000 rw-p 00021000 08:06 402537                     /usr/lib/x86_64-linux-gnu/libxcb.so.1.1.0
7ff5c1037000-7ff5c103b000 r-xp 00000000 08:06 402505                     /usr/lib/x86_64-linux-gnu/libxcb-dri2.so.0.0.0
7ff5c103b000-7ff5c123a000 ---p 00004000 08:06 402505                     /usr/lib/x86_64-linux-gnu/libxcb-dri2.so.0.0.0
7ff5c123a000-7ff5c123b000 r--p 00003000 08:06 402505                     /usr/lib/x86_64-linux-gnu/libxcb-dri2.so.0.0.0
7ff5c123b000-7ff5c123c000 rw-p 00004000 08:06 402505                     /usr/lib/x86_64-linux-gnu/libxcb-dri2.so.0.0.0
7ff5c123c000-7ff5c1253000 r-xp 00000000 08:06 402509                     /usr/lib/x86_64-linux-gnu/libxcb-glx.so.0.0.0
7ff5c1253000-7ff5c1452000 ---p 00017000 08:06 402509                     /usr/lib/x86_64-linux-gnu/libxcb-glx.so.0.0.0
7ff5c1452000-7ff5c1454000 r--p 00016000 08:06 402509                     /usr/lib/x86_64-linux-gnu/libxcb-glx.so.0.0.0
7ff5c1454000-7ff5c1455000 rw-p 00018000 08:06 402509                     /usr/lib/x86_64-linux-gnu/libxcb-glx.so.0.0.0
7ff5c1455000-7ff5c158a000 r-xp 00000000 08:06 396347                     /usr/lib/x86_64-linux-gnu/libX11.so.6.3.0
7ff5c158a000-7ff5c178a000 ---p 00135000 08:06 396347                     /usr/lib/x86_64-linux-gnu/libX11.so.6.3.0
7ff5c178a000-7ff5c178b000 r--p 00135000 08:06 396347                     /usr/lib/x86_64-linux-gnu/libX11.so.6.3.0
7ff5c178b000-7ff5c178f000 rw-p 00136000 08:06 396347                     /usr/lib/x86_64-linux-gnu/libX11.so.6.3.0
7ff5c178f000-7ff5c1790000 r-xp 00000000 08:06 396357                     /usr/lib/x86_64-linux-gnu/libX11-xcb.so.1.0.0
7ff5c1790000-7ff5c198f000 ---p 00001000 08:06 396357                     /usr/lib/x86_64-linux-gnu/libX11-xcb.so.1.0.0
7ff5c198f000-7ff5c1990000 r--p 00000000 08:06 396357                     /usr/lib/x86_64-linux-gnu/libX11-xcb.so.1.0.0
7ff5c1990000-7ff5c1991000 rw-p 00001000 08:06 396357                     /usr/lib/x86_64-linux-gnu/libX11-xcb.so.1.0.0
7ff5c1991000-7ff5c1996000 r-xp 00000000 08:06 401128                     /usr/lib/x86_64-linux-gnu/libXfixes.so.3.1.0
7ff5c1996000-7ff5c1b95000 ---p 00005000 08:06 401128                     /usr/lib/x86_64-linux-gnu/libXfixes.so.3.1.0
7ff5c1b95000-7ff5c1b96000 r--p 00004000 08:06 401128                     /usr/lib/x86_64-linux-gnu/libXfixes.so.3.1.0
7ff5c1b96000-7ff5c1b97000 rw-p 00005000 08:06 401128                     /usr/lib/x86_64-linux-gnu/libXfixes.so.3.1.0
7ff5c1b97000-7ff5c1b99000 r-xp 00000000 08:06 401122                     /usr/lib/x86_64-linux-gnu/libAborted (core dumped)
```

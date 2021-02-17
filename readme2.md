# Single Shot multibox Detectors (SSD):

**Note: SSD module is generalized on VGG-net.**  

* Unlike Faster RCNN which deals with detection process in two phases after feature extractor layer i.e. **RPN(Regional Proposal Network)** + **Detection Layer**, SSD framework does that in a single go.

* Here the output from feature extractor is further reduced to smaller dimensions in four phases. The output at these 4 phases including the output from the feature extractor output are passed on to a 3 x 3 x **variable size channels respective to a phase** .   

**Note 1: Please open it in new tab**  

<div align="center">
![export](/uploads/6ffa1be2b3300acf92a6573ba7a8b5f3/export.png)
</div>

* So if you see te image above, consider phase 1 and in that phase the output goes through 3 x 3 X (4 x (classes+4)) convolution, in this convolution the first **4** in **(4 x (classes+4))** denotes the number of default boxes for which the convolution generates offsets.

* Second **4** in the same expression denotes the co-ordinates for the detected object. For example consider there are 3 objects to be classified then the output would look like **4 x (3 + 4) = 4 x 7**, 3 in the given expression denotes binary label for the object confidence in the given box.

* At the output fro the intermediate phases(approx. 8732 detections/class) are given to the NMS module which in turn removes redundant boxes and gives refined output of all detections in an image.

<div align="center">
![ssd](/uploads/67899ed8c78bbf6131f331e964827dae/ssd.png)
</div>

* Same pipeline is followed for **ssd** and **ssdlite** for object detection purpose.

* The only difference between ssd and ssdlite, **ssdlite** uses depthwise seperabale convolutions in phase convolutions module. 

### Source:[Link](https://arxiv.org/pdf/1512.02325.pdf)

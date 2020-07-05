# Week's Review

### 29th June:

1. I started out with converting mobilenet's architecture into arm compute library Graph API format.

2. Firstly I followed the steps in already implemented MobileNet's classification caffemodel.

3. I listed down each and every process involved in it's method and mapped those methods into armcl's implementation.

4. Then I arranged every task according to it's complexity and started to execute.

5. I documented all of those process which were required in arm cl's mobilenet classification.


### 30th June:

1. I found out there were some mistakes while implementing the process in some critical parts in which peer's documentation also helped.

2. I removed all thos mistakes and updated my documentation with simplified summary.

### 1st July:

1. After MobileNet I moved towards converting FaceNet's inferencing architecture into arm compute's Graph API format.

2. I got clear with concepts involved in FaceNet's architecture and decided it's flow in arm compute library's implementation.

3. I extracted weights from already implemented facenet's caffemodel and made updates in arm compute library's inception resnet's v1 architecture accordingly.

4. But was unable to decide how to process output of inception resnet v1.

### 2nd July:

1. I found a method from where I can extract the outputs of the architecture and store it in numpy format

2. I decided to use SVM classifier for classifiying extracted unique outputs from the architecture.

3. I found out that there are implentations of SVM in C++ also but they required Vector format to process the inputs and armcl has only support for numpy.

### 3rd July:

1. Since the whole pipleline was ready , I wanted test the process on Raspberry Pi's Virtual enviornment.

2. I tried to simulate RPi enviornment on **Qemu** but faced some errors intitally due to it's enviorment's size.

3. Found a method to extend that enviornment and the issue of space was solved.


### 4th July:

1. I built Compute Library on RPi's enviornment which initally gave many errors of virtual memory and sometimes it would freeze due to heavy load.

2. Then I came to know that there's an alternate method of cross compilation in which I can compile the whole library on my host system and only transfer necessary files to RPi's envoirnment.

3. But still im facing some errors in the envoirnmnet which are version specific, still searching them and I hope to find the solutions soon.


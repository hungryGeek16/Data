## Weeks Review

### 22nd June:

1. My last week's task ended with training MobileNetV2 which classifies grocery objects and model was performing moderately. I got clear with caffe's methods of training.

2. After MobileNetV2, it was decided that I will train EfficientNet's variant on the same grocery data ans see results.

3. I trained on same grocery data on EfficientNet-B0's architecture which gave poor results. 

4. I thought by increasing the number of Samples and classes might solve the issues.

### 23rd June:

1. I updated the grocery dataset which was earlier created with 5 labels by adding 2 more to it.

2. I was using MobileNetV2 solver configurations on EfficientNet because both use the same principle of depthwise configuration.

3. Trained the model by using this configuration, but even then issue of poor result was not solved.

4. I updated some of the configurations inside solver.prototxt, even these changes turned futile.

5. Here I understood that identification of grocery items is not an easy task, it requires lot of research and experience to solve this problem.

### 24th June:

1. Since I used transfer learning while training MobileNetV2, I tried to understand transfer learning in more detailed manner.

2. I got to know that there are two types of transfer learning methods while training a model.

3. I simulated both of those methods on simpler classifier to make my concepts more clear.

### 25th June:

1. I gave a final try on creating an approach for classifying grocery objects.

2. Read many articles, research paper and blogs, but the problem itslef had many constraints. 

3. I tried to do classification on a simpler model as compared to the previous ones.

4. But then it was concluded that this problem needs more study and time which can be alloted in the long run.

### 26th June:

1. I was introduced to arm compute library which is a framework for deep learning.

2. I studied the file structure,functions and utilities in it and I found that library to be very instresting.

3. I understood the basic terms and concepts which it is using through the documentation and code comments present.

### 27th June:

1. After doing this I came back to image preprocessing concepts.

2. I studied about different predifined filters which are present for image processing





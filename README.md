# Data

#### b. Other replacements of tf core functions.

* There's a function called tf.gather which would take a list of indices and an array as input parameters and would output only the indices which are mentioned in the passed list arranged in the same way as of lust.

* Example: refer the image below

<div align='center'>
![dnd](/uploads/1c289d1449b8ca702bcada9fe8f286f6/dnd.png)
</div>

* Hence in numpy this would be implemented without a function, just pass the list of indices through numpy array and store them in new variable.

```python

import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9])

indices = [1,2,5]

v = a[indices]

print("Second array:",v) # This would give [2,3,6] as output.

```

* [tf.math.softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) function can replaced by [scipy.special.softmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html)


* These changes were made in two files called inference.py and utils.py. The tf and caffe versions are attached below:  
1.  **TF**:[inference.py](/uploads/1bfe13ef9e1a9218c52afe4b0010aa46/inference.py)[utils.py](/uploads/2cc75482a8b10d9322455b7ef541387e/utils.py)  
2.  **Caffe**:[inference.py](/uploads/cea125986fe7162dddd4611c5fd5b509/inference.py)[utils.py](/uploads/b481a5490bbecf65fe8efc4f9619daef/utils.py)  

#### NOTE: The above alternative functions are compatible with the tf library 2.3.0 , please check the documentation of other versions of tf and use accordingly.

### Step 3: Verifying outputs.

* Here after changing the functions totally, a test image was passed through both the networks and these two networks produced similar outputs.

<div align='center'>
![det](/uploads/d076cb5721e6f9b323093a7a90965a8b/det.png)
</div>

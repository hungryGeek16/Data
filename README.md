# Data

### Implement post-processing functions which are compatible for caffe.

#### a. Numpy replacements of tf core functions:

* In post processing, tf uses it's core functions for which alternative implementation must be found out.

* Most of it's functions can be replaced by numpy **utilities** as it is without even changing their parameters.

* Refer the mapping below:

| Sr.No | Tf function | Numpy alternative |
| ------ | ------ | ------ |
| **1** | [tf.concat()](https://www.tensorflow.org/api_docs/python/tf/concat) | [np.concatenate()](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)|
| **2** | [tf.split()](https://www.tensorflow.org/api_docs/python/tf/split) | [np.split()](https://numpy.org/doc/stable/reference/generated/numpy.split.html)|
| **3** | [tf.broadcast_to()](https://www.tensorflow.org/api_docs/python/tf/broadcast_to) |[np.broadcast_to()](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.broadcast_to.html)|
| **4** | [tf.expand_dims()](https://www.tensorflow.org/api_docs/python/tf/expand_dims)| [np.expand_dims()](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)|
| **5** | [tf.minimum()](https://www.tensorflow.org/api_docs/python/tf/math/minimum),[tf.maximum()](https://www.tensorflow.org/api_docs/python/tf/math/maximum)| [np.minimum()](https://numpy.org/doc/stable/reference/generated/numpy.minimum.html),[np.maximum()](https://numpy.org/doc/stable/reference/generated/numpy.maximum.html)|
| **6** | [tf.clip_by_value()](https://www.tensorflow.org/api_docs/python/tf/clip_by_value)| [np.clip()](https://numpy.org/doc/stable/reference/generated/numpy.split.html)|
| **7** | [tf.math.exp()](https://www.tensorflow.org/api_docs/python/tf/math/exp)| [np.exp()](https://numpy.org/doc/stable/reference/generated/numpy.exp.html)|
| **8** | [tf.ones_like()](https://www.tensorflow.org/api_docs/python/tf/ones_like)| [np.ones_like()](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html)|
| **9** | [tf.math.reduce_any()](https://www.tensorflow.org/api_docs/python/tf/math/reduce_any)| [np.any()](https://numpy.org/doc/stable/reference/generated/numpy.any.html)|
| **10** | [tf.where()](https://www.tensorflow.org/api_docs/python/tf/where)| [np.where()](https://numpy.org/doc/stable/reference/generated/numpy.where.html)|
| **11** | [tf.cast()](https://www.tensorflow.org/api_docs/python/tf/cast)| [np.ndarray.astype()](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)|


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

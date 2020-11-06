### Log2:

1. Tried to print a layer's output by changing a line in the cpp file **FROM**:

```cpp
graph << InputLayer(input_descriptor,get_input_accessor(common_params, std::move(preprocessor))); 
```

TO:

```cpp
graph << InputLayer(input_descriptor,get_input_accessor(common_params, std::move(preprocessor)))
      << OutputLayer(get_print_output_accessor());
```      
Link to get_print_output_accessor():[Here](https://github.com/ARM-software/ComputeLibrary/blob/master/utils/GraphUtils.h#L651)

2. But this gave an error which was way bigger to fit in the terminal, so I tried to search in the issues of ARMCL where I thought I could find a method to print the output tensor from each layer.

3. So in this [issue](https://github.com/ARM-software/ComputeLibrary/issues/633#issuecomment-465583101), it's mentioned that we should compile armcl library with debug=1 ,which would allow us to print values.

4. I compiled the whole library again with this command:

```cpp
scons arch=arm64-v8a benchmark=1 benchmark_tests=1 opencl=0 neon=1 debug=1 cppthreads=1 benchmark_tests=1 os=linux -j3 Werror=0
```
5. Then agian transported the compiled library to rpi:

6. But this time it gave me segmentation fault

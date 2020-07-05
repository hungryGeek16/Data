After building Compute Library , wanted to cross verify 

* Transferred required build files into RPi's enviornment.The command below does that.
```cpp

scp -P 5022 build/example/graph_alexnet build/libarm_compute.so build/libarm_compute_core.so build/libarm_compute_graph.so pi@127.0.0.1:Desktop
```

*  After transferring, stored shared objects in **build** folder inside Desktop folder and alexnet.cpp inside **build/examples/**.

*  Ran these commands after storing.

```cpp
export LD_LIBRARY_PATH=build/ 

PATH_ASSETS=/home/pi/assets_alexnet 

./build/examples/graph_alexnet 0 $PATH_ASSETS  $PATH_ASSETS/go_kart.ppm $PATH_ASSETS/labels.txt

```

* By running the above commands, the enviornment gave the below error.



* Hence found a solution from this [site](https://github.com/doublethinkco/cpp-ethereum-cross/issues/79#issuecomment-287878665) which said to update RPi's kernel by updating gcc.

* Followed these steps in RPi's enviornment:

1. Edit the package source file with sudo nano /etc/apt/sources.list
2. Replace jessie with stretch in that file to point to the testing repository that has the gcc-5 package
```cpp
sudo apt-get update to refresh the list of packages
sudo apt-get install gcc-5
```



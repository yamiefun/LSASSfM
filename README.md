# Large Scale Ambiguous Scene Structure from Motion (LSASSfM)

*This work is developed based on [DAGSfM](https://github.com/AIBluefisher/DAGSfM) and [COLMAP](https://github.com/colmap/colmap).*

## 1. Overview of LSASSfM
**[Paper](https://dl.acm.org/doi/abs/10.1145/3550082.3564199)**

In recent years, 3D reconstruction methods have been used in many different fields, for example, augmented reality, autonomous navigation, etc. The most famous algorithm is structure-from-motion (SfM). SfM takes 2D images in different view point in the scene as input, extracts feature points in images, and figure out position of those points and camera poses in 3D space. Although SfM method has been widely used, it often fails to reconstruct ambiguous or repeated structures due to large amount of false feature matching. [Cui et al. 2020] claimed that the image which has the largest number of feature matching is most likely the correct matched image, and used this theory to filter out more outliers, made their method more robust against repeated structures. [Su et al. 2020] is the first work that introduced additional inertial measurement unit (IMU) information and successfully reconstruct ambiguous scenes. However, their method is still unable to work well on large-scale indoor scene, which is still a big challenge due to the severely ambiguous and inefficient problem.

To resolve the problem mentioned above, we propose a method to reconstruct large-scale indoor scene by building a correct view graph with the help of some additional information. The experiments show that our method can successfully reconstruct largescale scenes with highly ambiguous and repeated structures.

If you use this project for your research, please cite:
```
@inproceedings{10.1145/3550082.3564199,
    author = {Yu, Chia Hung and Chen, Kuan Wen},
    title = {Robust and Efficient Structure-from-Motion Method for Ambiguous Large-Scale Indoor Scene},
    year = {2022},
    isbn = {9781450394628},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3550082.3564199},
    doi = {10.1145/3550082.3564199},
    booktitle = {SIGGRAPH Asia 2022 Posters},
    articleno = {25},
    numpages = {2},
    location = {Daegu, Republic of Korea},
    series = {SA '22}
}
@misc{chen2019graphbased,
    title={Graph-Based Parallel Large Scale Structure from Motion},
    author={Yu Chen and Shuhan Shen and Yisong Chen and Guoping Wang},
    year={2019},
    eprint={1912.10659},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@inproceedings{schoenberger2016sfm,
    author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
    title={Structure-from-Motion Revisited},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2016},
}
```

## 2. How to Build

### 2.1 Required

```
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev
```

Then install [ceres-solver](http://ceres-solver.org/)
```sh
sudo apt-get install libatlas-base-dev libsuitesparse-dev
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout $(git describe --tags) # Checkout the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make
sudo make install
```

Build our LSASSfM
```sh
git clone https://github.com/yamiefun/LSASSfM.git
cd LSASSfM
mkdir build
cd build
cmake ..
make -j8
sudo make install
```

## 3. Usage
TODO

## Licence

```
Copyright (c) 2022, Henry Yu.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
      its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Author: Henry Yu (yamiefun@gmail.com)
```

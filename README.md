# ICME2023 《FlowText: Synthesizing Realistic Scene Text Video with Optical Flow Estimation》
 ## Get Started
 To setup the environment of FlowText, we need to set up the environment of segmentation model [Mask2fomer](https://github.com/facebookresearch/Mask2Former), depth estimation model [Monodepth2](https://github.com/nianticlabs/monodepth2), optical flow estimation model [GMA](https://github.com/zacjiang/GMA), and synthesis engine [SynthText](https://github.com/ankush-me/SynthText). We use `conda` to manage our dependencies. Our developers use `CUDA 11.1` to do experiments. You can specify the appropriate `cudatoolkit` version to install on your machine in the `environment.yml` file, and then run the following to create the `conda` environment:
 ```
 conda env create -f environment.yml
 ```

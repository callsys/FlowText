# ICME2023 《FlowText: Synthesizing Realistic Scene Text Video with Optical Flow Estimation》
 ## Get Started
 ### Environment Setup
 FlowText is based on the segmentation model [Mask2fomer](https://github.com/facebookresearch/Mask2Former), depth estimation model [Monodepth2](https://github.com/nianticlabs/monodepth2), optical flow estimation model [GMA](https://github.com/zacjiang/GMA), synthesis engine [SynthText](https://github.com/ankush-me/SynthText). To setup the environment of FlowText, we use `conda` to manage our dependencies. Our developers use `CUDA 11.1` to do experiments. You can specify the appropriate `cudatoolkit` version to install on your machine in the `requirements.txt` file, and then run the following commands to install FlowText:
 ```
conda create -n flowtext python=3.8
conda activate flowtext

git clone https://github.com/callsys/FlowText
cd FlowText
pip install -r requirements.txt
 
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
 
cd segmentation/mask2former/modeling/pixel_decoder/ops/
sh make.sh
 ```
### Download Models

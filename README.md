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
To run FlowText, you need to download some files, which mainly contain the font file for the synthesized text, the text source, and the weight of the models. Once you have downloaded the files, link them to the FlowText directory:
```
ln -s path/to/files FlowText/data
```
### Generate Synthetic Videos with Full Annotations
Generate Synthetic video with demo video `assets/demo.mp4` and output to result to `assets`:
```
python gen.py
```
Generate Synthetic video with given video `video.mp4`, frame range `start,end,interval`, save path `save` and random seed `seed`:
```
python gen.py --video video.mp4 --range start,end,interval --save save --seed seed
```
For example:
```
python gen.py --video assets/demo.mp4 --range 0,400,5 --save assets/result --seed 16
```
### Output Format


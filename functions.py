import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import sys
import json
from torchvision import transforms as T
sys.path.insert(0,'depth')
sys.path.insert(0,'segmentation')
sys.path.insert(0,'segmentation/demo_video')
sys.path.insert(0,'segmentation/demo')
sys.path.insert(0,'flow')
sys.path.insert(0,'flow/core')

import torch
from torchvision import transforms, datasets
import torch.nn.functional as F

import depth.networks as networks
from depth.layers import disp_to_depth
from depth.utils import download_model_if_doesnt_exist
import matplotlib.pyplot as plt
import cv2
import shutil
from segmentation.demo.demo import setup_cfg, VisualizationDemo, read_image
from segmentation.demo_video.demo import setup_cfg as vsetup_cfg
from segmentation.demo_video.demo import VisualizationDemo as vVisualizationDemo
from segmentation.demo_video.demo import autocast

import tqdm
import time
from flow.evaluate_single import RAFTGMA, load_image, InputPadder
from params import params

class segmentation_args():
    def __init__(self):
        # self.config_file = './segmentation/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml'
        # self.opts = ['MODEL.WEIGHTS','./segmentation/ckpoints/model_final_6b4a3a.pkl']
        self.config_file = './segmentation/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml'
        self.opts = ['MODEL.WEIGHTS', 'data/ckpts/segmentation/model_final_f07440.pkl']
        self.confidence_threshold = 0.5
        self.output = None
        self.video_output = None
        self.webcam = False

class video_segmentation_args():
    def __init__(self):
        self.config_file = './segmentation/configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml'
        self.opts = ['MODEL.WEIGHTS','data/ckpts/segmentation/model_final_4da256.pkl']
        self.confidence_threshold = 0.5
        self.output = None
        self.video_output = None
        self.webcam = False

class flow_args():
    def __init__(self):
        super().__init__()
        self.mixed_precision = False
        self.model = 'data/ckpts/flow/gma-kitti.pth' # 'flow/checkpoints/gma-sintel.pth'
        self.model_name = 'GMA'
        self.num_heads = 1
        self.path = None
        self.position_and_content = False
        self.position_only = False
        self.dropout = 0







def depth_es(image, rg=(0.1,100)):
    # model_name = 'mono+stereo_640x192'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # download_model_if_doesnt_exist(model_name)
    # model_path = os.path.join("models", model_name)
    # print("-> Loading model from ", model_path)
    encoder_path = os.path.join("data/ckpts/depth", "encoder.pth")
    depth_decoder_path = os.path.join("data/ckpts/depth", "depth.pth")

    # LOADING PRETRAINED MODEL
    # print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if isinstance(image,str):
        # Only testing on a single image
        paths = [image]
    elif isinstance(image,list):
        paths = image
    else:
        raise ValueError

    # print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    depths = []
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # scaled_disp, depth = disp_to_depth(disp_resized, 0.1, 100)
            rg = params['depth']['range']
            scaled_disp, depth = disp_to_depth(disp_resized, rg[0], rg[1])
            depths.append(depth.cpu().numpy()[0][0])
            # depths.append(scaled_disp.cpu().numpy()[0][0])
            # depths.append(scaled_disp.cpu().numpy()[0][0])
        if len(depths)==1:
            return depths[0]
        else:
            return depths

def segmentation_es(image):
    args = segmentation_args()
    if isinstance(image,list):
        args.input = image
    else:
        args.input = [image]

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    segs = []
    areas = []
    labels = []
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)

        # panoptic_seg = predictions['panoptic_seg']
        # seg = panoptic_seg[0].cpu().numpy()
        # area = np.array([el['area'] for el in predictions['panoptic_seg'][1]])
        # label = np.array([el['id'] for el in predictions['panoptic_seg'][1]])

        seg = predictions['sem_seg'].softmax(0).argmax(0).cpu().numpy()

        label = np.unique(seg)

        area = np.zeros_like(label)

        for i,l in enumerate(label):
            area[i] = (seg==l).sum()
        segs.append(seg)
        areas.append(area)
        labels.append(label)
    if len(segs)==1:
        return segs[0], areas[0], labels[0]
    else:
        return segs, areas, labels

def video_segmentation_es(image):
    # mp.set_start_method("spawn", force=True)
    args = video_segmentation_args()

    if isinstance(image,list):
        args.input = image
    else:
        args.input = [image]

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        vid_frames = []
        for path in args.input:
            img = read_image(path, format="BGR")
            vid_frames.append(img)

        with autocast():
            predictions, visualized_output = demo.run_on_video(vid_frames)
        T = len(vid_frames)
        segs = []
        labels = []
        areas = []
        label = np.arange(1,1+len(predictions['pred_labels'])).astype(np.int64)
        for t in range(T):
            scores = predictions['pred_scores']
            seg = [el[t].numpy() for el in predictions['pred_masks']]
            lst = [(el1, el2) for el1,el2 in zip(scores,seg)]
            lst = sorted(lst,key=lambda x:x[0])
            canvas = np.zeros_like(seg[0]).astype(np.int64)
            for idx,(el1,el2) in enumerate(lst):
                canvas[el2>0] = idx+1
            area = np.zeros_like(label)
            for idx,l in enumerate(label):
                area[idx] = (canvas==l).sum()
            segs.append(canvas)
            labels.append(label)
            areas.append(area)
        return segs, areas, labels

def mixin_segmentation_es(images, key):
    segs = []
    areas = []
    labels = []

    # panoptic segmentation for the key frame
    imagek = images[key]
    args = segmentation_args()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    img = read_image(imagek, format="BGR")
    predictions, visualized_output = demo.run_on_image(img)

    pseg = predictions['sem_seg'].softmax(0).argmax(0).cpu().numpy()

    plabel = np.unique(pseg)

    nplabel = 0
    npseg = np.zeros_like(pseg).astype(np.float32)
    for label in plabel:
        binary = (pseg==label).astype(np.int64)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            npseg = cv2.fillPoly(npseg, [contour], nplabel)
            nplabel = nplabel + 1

    nplabel = np.unique(npseg)
    nparea = np.zeros_like(nplabel)

    for i, l in enumerate(nplabel):
        nparea[i] = (npseg == l).sum()

    pseg = npseg
    parea = nparea
    plabel = nplabel





    # video instance segmentation for all frames
    args = video_segmentation_args()
    cfg = vsetup_cfg(args)
    demo = vVisualizationDemo(cfg)


    vid_frames = []
    for path in images:
        img = read_image(path, format="BGR")
        vid_frames.append(img)

    with autocast():
        predictions, visualized_output = demo.run_on_video(vid_frames)
    T = len(vid_frames)
    label = np.arange(1001, 1001 + len(predictions['pred_labels'])).astype(np.int64)
    all_label = np.concatenate([plabel,label],0)
    for t in range(T):
        scores = predictions['pred_scores']
        seg = [el[t].numpy() for el in predictions['pred_masks']]
        lst = [(el1, el2, el3) for el1, el2, el3 in zip(scores, seg, label)]
        lst = sorted(lst, key=lambda x: x[0])
        if t==key:
            canvas = pseg
        else:
            canvas = np.zeros_like(seg[0]).astype(np.int64)
        for el1, el2, el3 in lst:
            canvas[el2 > 0] = el3
        area = np.zeros_like(all_label)
        for idx, l in enumerate(all_label):
            area[idx] = (canvas == l).sum()
        segs.append(canvas)
        labels.append(all_label)
        areas.append(area)


    return segs, areas, labels

def flow_es(images):
    args = flow_args()
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    l = 640

    with torch.no_grad():

        images = sorted(images)

        flows = []
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            # print(f"Reading in images at {imfile1} and {imfile2}")

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, _, h_org, w_org = image1.shape
            ratio = l/w_org
            h_new = int(h_org*ratio)
            w_new = int(w_org*ratio)
            image1 = F.interpolate(image1,(h_new , w_new),mode='nearest')
            image2 = F.interpolate(image2, (h_new, w_new), mode='nearest')

            flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)
            # print(f"Estimating optical flow...")
            flow_up = F.interpolate(flow_up,(h_org,w_org),mode='bilinear')/ratio

            flows.append(flow_up[0].cpu().numpy())
        return flows

def video2images(video ,path = None, range=None, size=(320,320)):
    if len(range) == 3:
        st = range[0]
        ed = range[1]
        itv = range[-1]
    else:
        st = range[0]
        ed = range[1]
        itv = 1

    if not video.endswith('mp4'):
        assert path is not None
        if not os.path.exists(path):
            os.mkdir(path)
        frames = sorted([os.path.join(video,el) for el in os.listdir(video) if el.endswith('jpg')])
        for i,frame in enumerate(frames):
            if i<st or i >=ed or (i-st)%itv!=0:
                continue
            name = os.path.join(path, '{:0>8d}.jpg'.format((i-st)//itv))
            shutil.copy(frame, name)

        return


    if path is None:
        save_dir = video[:-4]
    else:
        save_dir = path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)



    video = cv2.VideoCapture(video)  # 打开视频文件
    fps = int(video.get(cv2.CAP_PROP_FPS))  # 获取帧率

    count = 0
    while True:
        exist, frame = video.read()  # 读取一帧数据
        if not exist:
            break
        if count>=st and count<ed and (count-st)%itv==0:
            org_h, org_w, _ = frame.shape
            new_h, new_w = size
            ratio = max(new_h / org_h, new_w / org_w)
            new_h = int(org_h * ratio)//64*64
            new_w = int(org_w * ratio)//64*64
            new_frame = np.array(T.Resize((new_h, new_w))(pil.fromarray(frame)))
            name = os.path.join(save_dir,'{:0>8d}.jpg'.format((count-st)//itv))
            cv2.imwrite(name,new_frame)
        count += 1
    video.release()  # 关闭视频

def video_length(video):
    if os.path.isdir(video):
        frames = [el for el in os.listdir(video) if el.endswith('jpg') or el.endswith('png')]
        return len(frames)
    else:
        video = cv2.VideoCapture(video)  # 打开视频文件
        return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

def vis_sample(path = 'tmp/sample',pltann=True, is_video=True):

    if is_video:
        images = sorted([os.path.join(path,el) for el in os.listdir(path) if el.endswith('jpg')])
        h, w = cv2.imread(images[0]).shape[:2]
        save_path = os.path.join(path,'viz_ann.mp4') if pltann else os.path.join(path,'viz.mp4')
        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 8, (w, h))
        with open(os.path.join(path,'ann.json'), 'r') as fr:
            anns = json.load(fr)
        for image,ann in zip(images,anns):
            img = cv2.imread(image)
            if pltann:
                lst = list(ann['instances'].values())
                for tmp in lst:
                    bbox = tmp['coords']
                    text = tmp['text']
                    cv2.polylines(img,[np.array(bbox).astype(np.int32)],True,color=[0,255,0],thickness=2)
                    cv2.putText(img,text,np.array(bbox).mean(0).astype(np.int32),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)
            # plt.imshow(img)
            # plt.show()
            video.write(img)
        video.release()
    else:
        save_path = os.path.join(path, 'viz_ann.jpg') if pltann else os.path.join(path, 'viz.jpg')
        image = [os.path.join(path,el) for el in os.listdir(path) if el.endswith('jpg')][0]
        with open(os.path.join(path,'ann.json'), 'r') as fr:
            ann = json.load(fr)[0]
        img = cv2.imread(image)
        if pltann:
            lst = list(ann['instances'].values())
            for tmp in lst:
                bbox = tmp['coords']
                text = tmp['text']
                cv2.polylines(img, [np.array(bbox).astype(np.int32)], True, color=[0, 255, 0], thickness=2)
                cv2.putText(img, text, np.array(bbox).mean(0).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)
        cv2.imwrite(save_path,img)

def num_samples(path = 'tmp/synthtextvid_vatex'):
    files = [os.path.join(path,el) for el in os.listdir(path)]
    return [el for el in files if os.path.exists(os.path.join(el,'ann.json'))], [el for el in files if not os.path.exists(os.path.join(el,'ann.json'))]





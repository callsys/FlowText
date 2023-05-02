'''
FlowText generator
'''
import os
import argparse
import random
from synthgen import *
from functions import depth_es, mixin_segmentation_es, flow_es, video2images, vis_sample
from PIL import Image
import json
import shutil
from shapely.geometry import Polygon
import cv2
import traceback
import tqdm

meta = {
    'SECS_PER_IMG': 5,  # max time per image in seconds
    'DATA_PATH': 'data',
    'ntry': 1,
    'NUM_REP': 3,
}


def main(video='tmp/demo.mp4', rg=[100, 150], save='tmp/sample'):
    try:
        save_ann = os.path.join(save,'ann.json')
        if os.path.exists(save_ann):
            print(f'{save} already exists')
            return
        t1 = time.time()
        # video to image list
        if os.path.exists(save):
            shutil.rmtree(save)
        video2images(video, path=save, range=rg)
        images = sorted([os.path.join(save, el) for el in os.listdir(save)])

        # random sample a key frame the paint the init text
        # key = 4
        key = len(images)//2
        # key = random.randint(1, len(images) - 2)
    except:
        traceback.print_exc()
        print(f'fail to generate for {video} in range {str(rg)}')
        if os.path.exists(save):
            shutil.rmtree(save)
        return

    try:
        t = 0
        while t < 4:
            print('model inference ...')
            # flow,depth,segmentation estimation

            RV10 = RendererV10(meta['DATA_PATH'], max_time=meta['SECS_PER_IMG'])
            imgs = [np.array(Image.open(el)) for el in images]
            flows = flow_es(images)
            depthk = depth_es(images[key])
            depths = [depthk]*len(imgs)
            segs, areas, labels = mixin_segmentation_es(images, key)

            print('text painting')
            # paint text for all frames
            res = RV10.render_text(imgs, flows, depths, segs, areas, labels, key)

            if len(res) == 0:
                t = t + 1
                continue
            else:
                break

        # no valid paint
        if len(res) == 0:
            print(f'no results.')
            if os.path.exists(save):
                shutil.rmtree(save)
            return

        # write painted images
        out_imgs = [el['img'] for el in res]
        for image, img in zip(images, out_imgs):
            cv2.imwrite(image, img[:,:,::-1])
        h, w, _ = img.shape


        # write annotations
        anns = []
        ids = list(range(len(res[0]['wordBB'])))
        txts = ' '.join(res[0]['txt']).split()
        for res_, image in zip(res, images):
            ann = dict()
            ann['wordBB'] = res_['wordBB']
            ann['words'] = dict(zip(ids,' '.join(res_['txt']).split()))
            instances = dict()

            for id,txt in zip(ids,txts):
                try:
                    box = ann['wordBB'][id]
                except:
                    print(id)
                box = np.array(box).astype(np.int32)

                try:
                    mask = Polygon(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.int32))
                    poly = Polygon(np.array(box).astype(np.int32))
                    npoly = poly.intersection(mask)

                    hw = cv2.minAreaRect(box)[1]
                    minhw = min(hw)
                    asp = max(hw) / (min(hw) + 1e-6)
                    if npoly.area < 10 or minhw < 5:
                        continue
                    coords = box.tolist()

                    instances[id] = {'text':txt,'coords':coords}
                except:
                    continue
            ann['instances'] = instances

            ann['img'] = image
            ann['key'] = images[key]
            ann.pop('wordBB')
            anns.append(ann)

        save_ann = os.path.join(save, 'ann.json')
        with open(save_ann, 'w') as fw:
            json.dump(anns, fw)
        t2 = time.time()
        print('successful paint a clip with {:.2f}s'.format(t2 - t1))
        vis_sample(save, pltann=False)
        vis_sample(save, pltann=True)
    except:
        traceback.print_exc()
        print(f'fail to paint.')
        if os.path.exists(save):
            shutil.rmtree(save)
        print(images)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='assets/demo.mp4', help='background video')
    parser.add_argument('--range', default='0,500,5', help='range of the video')
    parser.add_argument('--save', default='', help='save path')
    parser.add_argument('--seed', default='16', help='random seed')
    args = parser.parse_args()

    video = args.video


    save = args.save
    if len(save)==0:
        save = video.split('.')[0] + '_' + args.range + '_' + args.seed

    rg = args.range
    rg = rg.split(',')
    if len(rg)==0:
        rg = None
    elif len(rg)==2:
        rg = [int(rg[0]),int(rg[1])]
    elif len(rg)==3:
        rg = [int(rg[0]),int(rg[1]),int(rg[2])]
    else:
        raise ValueError("Invalid range input.")

    seed = int(args.seed)
    np.random.seed(seed)
    random.seed(seed)

    main(video, rg, save)


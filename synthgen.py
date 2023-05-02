"""
FlowText Engine by Yuzhong Zhao
"""

from __future__ import division
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import synth_utils as su
import text_utils as tu
from colorize_poisson import Colorize4
from common import *
import traceback, itertools
import torch.nn.functional as F
import torch
from shapely.geometry import Polygon
from scipy.linalg import sqrtm, inv
import time
import copy
import math
from params import params
from itertools import *

class TextRegions(object):
    """
    Get region from segmentation which are good for placing
    text.
    """
    minWidth = 30 #px
    minHeight = 30 #px
    minAspect = 0.3 # w > 0.3*h
    maxAspect = 7
    minArea = 100 # number of pix
    pArea = 0.60 # area_obj/area_minrect >= 0.6

    # RANSAC planar fitting params:
    dist_thresh = 0.10 # m
    num_inlier = 90
    ransac_fit_trials = 100
    min_z_projection = 0.25

    minW = 20

    @staticmethod
    def filter_rectified(mask):
        """
        mask : 1 where "ON", 0 where "OFF"
        """
        wx = np.median(np.sum(mask,axis=0))
        wy = np.median(np.sum(mask,axis=1))
        return wx>TextRegions.minW and wy>TextRegions.minW

    @staticmethod
    def get_hw(pt,return_rot=False):
        pt = pt.copy()
        R = su.unrotate2d(pt)
        mu = np.median(pt,axis=0)
        pt = (pt-mu[None,:]).dot(R.T) + mu[None,:]
        h,w = np.max(pt,axis=0) - np.min(pt,axis=0)
        if return_rot:
            return h,w,R
        return h,w
 
    @staticmethod
    def filter(seg,area,label):
        """
        Apply the filter.
        The final list is ranked by area.
        """
        good = label[area > TextRegions.minArea]
        area = area[area > TextRegions.minArea]
        filt,R = [],[]
        for idx,i in enumerate(good):
            mask = seg==i
            xs,ys = np.where(mask)

            coords = np.c_[xs,ys].astype('float32')
            rect = cv2.minAreaRect(coords)          
            #box = np.array(cv2.cv.BoxPoints(rect))
            box = np.array(cv2.boxPoints(rect))
            h,w,rot = TextRegions.get_hw(box,return_rot=True)

            f = (h > TextRegions.minHeight 
                and w > TextRegions.minWidth
                and TextRegions.minAspect < w/h < TextRegions.maxAspect
                and area[idx]/w*h > TextRegions.pArea)
            filt.append(f)
            R.append(rot)

        # filter bad regions:
        filt = np.array(filt)
        area = area[filt]
        R = [R[i] for i in range(len(R)) if filt[i]]

        # sort the regions based on areas:
        aidx = np.argsort(-area)
        good = good[filt][aidx]
        R = [R[i] for i in aidx]
        filter_info = {'label':good, 'rot':R, 'area': area[aidx]}
        return filter_info

    @staticmethod
    def sample_grid_neighbours(mask,nsample,step=3):
        """
        Given a HxW binary mask, sample 4 neighbours on the grid,
        in the cardinal directions, STEP pixels away.
        """
        if 2*step >= min(mask.shape[:2]):
            return #None

        y_m,x_m = np.where(mask)
        mask_idx = np.zeros_like(mask,'int32')
        for i in range(len(y_m)):
            mask_idx[y_m[i],x_m[i]] = i

        xp,xn = np.zeros_like(mask), np.zeros_like(mask)
        yp,yn = np.zeros_like(mask), np.zeros_like(mask)
        xp[:,:-2*step] = mask[:,2*step:]
        xn[:,2*step:] = mask[:,:-2*step]
        yp[:-2*step,:] = mask[2*step:,:]
        yn[2*step:,:] = mask[:-2*step,:]
        valid = mask&xp&xn&yp&yn

        ys,xs = np.where(valid)
        N = len(ys)
        if N==0: #no valid pixels in mask:
            return #None
        nsample = min(nsample,N)
        idx = np.random.choice(N,nsample,replace=False)
        # generate neighborhood matrix:
        # (1+4)x2xNsample (2 for y,x)
        xs,ys = xs[idx],ys[idx]
        s = step
        X = np.transpose(np.c_[xs,xs+s,xs+s,xs-s,xs-s][:,:,None],(1,2,0))
        Y = np.transpose(np.c_[ys,ys+s,ys-s,ys+s,ys-s][:,:,None],(1,2,0))
        sample_idx = np.concatenate([Y,X],axis=1)
        mask_nn_idx = np.zeros((5,sample_idx.shape[-1]),'int32')
        for i in range(sample_idx.shape[-1]):
            mask_nn_idx[:,i] = mask_idx[sample_idx[:,:,i][:,0],sample_idx[:,:,i][:,1]]
        return mask_nn_idx

    @staticmethod
    def filter_depth(xyz,seg,regions):
        plane_info = {'label':[],
                      'coeff':[],
                      'support':[],
                      'rot':[],
                      'area':[]}
        for idx,l in enumerate(regions['label']):
            mask = seg==l
            pt_sample = TextRegions.sample_grid_neighbours(mask,TextRegions.ransac_fit_trials,step=3)
            if pt_sample is None:
                continue #not enough points for RANSAC
            # get-depths
            pt = xyz[mask]
            plane_model = su.isplanar(pt, pt_sample,
                                     TextRegions.dist_thresh,
                                     TextRegions.num_inlier,
                                     TextRegions.min_z_projection)
            if plane_model is not None:
                plane_coeff = plane_model[0]
                if np.abs(plane_coeff[2])>TextRegions.min_z_projection:
                    plane_info['label'].append(l)
                    plane_info['coeff'].append(plane_model[0])
                    plane_info['support'].append(plane_model[1])
                    plane_info['rot'].append(regions['rot'][idx])
                    plane_info['area'].append(regions['area'][idx])

        return plane_info

    @staticmethod
    def get_regions(xyz,seg,area,label):
        regions = TextRegions.filter(seg,area,label)
        # fit plane to text-regions:
        regions = TextRegions.filter_depth(xyz,seg,regions)
        return regions

def rescale_frontoparallel(p_fp,box_fp,p_im):
    """
    The fronto-parallel image region is rescaled to bring it in 
    the same approx. size as the target region size.

    p_fp : nx2 coordinates of countour points in the fronto-parallel plane
    box  : 4x2 coordinates of bounding box of p_fp
    p_im : nx2 coordinates of countour in the image

    NOTE : p_fp and p are corresponding, i.e. : p_fp[i] ~ p[i]

    Returns the scale 's' to scale the fronto-parallel points by.
    """
    l1 = np.linalg.norm(box_fp[1,:]-box_fp[0,:])
    l2 = np.linalg.norm(box_fp[1,:]-box_fp[2,:])

    n0 = np.argmin(np.linalg.norm(p_fp-box_fp[0,:][None,:],axis=1))
    n1 = np.argmin(np.linalg.norm(p_fp-box_fp[1,:][None,:],axis=1))
    n2 = np.argmin(np.linalg.norm(p_fp-box_fp[2,:][None,:],axis=1))

    lt1 = np.linalg.norm(p_im[n1,:]-p_im[n0,:])
    lt2 = np.linalg.norm(p_im[n1,:]-p_im[n2,:])

    s =  max(lt1/l1,lt2/l2)
    if not np.isfinite(s):
        s = 1.0
    return s

def get_text_placement_mask(xyz,mask,plane,pad=2,viz=False):
    """
    Returns a binary mask in which text can be placed.
    Also returns a homography from original image
    to this rectified mask.

    XYZ  : (HxWx3) image xyz coordinates
    MASK : (HxW) : non-zero pixels mark the object mask
    REGION : DICT output of TextRegions.get_regions
    PAD : number of pixels to pad the placement-mask by
    """
    contour,hier = cv2.findContours(mask.copy().astype('uint8'),
                                    mode=cv2.RETR_CCOMP,
                                    method=cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contour = [np.squeeze(c).astype('float') for c in contour]
    #plane = np.array([plane[1],plane[0],plane[2],plane[3]])
    H,W = mask.shape[:2]

    # bring the contour 3d points to fronto-parallel config:
    pts,pts_fp = [],[]
    center = np.array([W,H])/2
    # n_front = np.array([0.0, 0.0, 1.0])
    n_front = np.array([0.0,0.0,-1.0])
    for i in range(len(contour)):
        cnt_ij = contour[i]
        xyz = su.DepthCamera.plane2xyz(center, cnt_ij, plane)
        R = su.rot3d(plane[:3],n_front)
        xyz = xyz.dot(R.T)
        pts_fp.append(xyz[:,:2])
        pts.append(cnt_ij)

    # unrotate in 2D plane:
    rect = cv2.minAreaRect(pts_fp[0].copy().astype('float32'))
    box = np.array(cv2.boxPoints(rect))
    R2d = su.unrotate2d(box.copy())
    box = np.vstack([box,box[0,:]]) #close the box for visualization

    mu = np.median(pts_fp[0],axis=0)
    pts_tmp = (pts_fp[0]-mu[None,:]).dot(R2d.T) + mu[None,:]
    boxR = (box-mu[None,:]).dot(R2d.T) + mu[None,:]
    
    # rescale the unrotated 2d points to approximately
    # the same scale as the target region:
    s = rescale_frontoparallel(pts_tmp,boxR,pts[0])
    boxR *= s
    for i in range(len(pts_fp)):
        pts_fp[i] = s*((pts_fp[i]-mu[None,:]).dot(R2d.T) + mu[None,:])

    # paint the unrotated contour points:
    minxy = -np.min(boxR,axis=0) + pad//2
    ROW = np.max(ssd.pdist(np.atleast_2d(boxR[:,0]).T))
    COL = np.max(ssd.pdist(np.atleast_2d(boxR[:,1]).T))

    place_mask = 255*np.ones((int(np.ceil(COL))+pad, int(np.ceil(ROW))+pad), 'uint8')

    pts_fp_i32 = [(pts_fp[i]+minxy[None,:]).astype('int32') for i in range(len(pts_fp))]
    cv2.drawContours(place_mask,pts_fp_i32,-1,0,
                     thickness=cv2.FILLED,
                     lineType=8,hierarchy=hier)
    
    if not TextRegions.filter_rectified((~place_mask).astype('float')/255):
        return

    # calculate the homography
    H,_ = cv2.findHomography(pts[0].astype('float32').copy(),
                             pts_fp_i32[0].astype('float32').copy(),
                             method=0)

    Hinv,_ = cv2.findHomography(pts_fp_i32[0].astype('float32').copy(),
                                pts[0].astype('float32').copy(),
                                method=0)
    if viz:
        plt.subplot(1,2,1)
        plt.imshow(mask)
        plt.subplot(1,2,2)
        plt.imshow(~place_mask)
        for i in range(len(pts_fp_i32)):
            plt.scatter(pts_fp_i32[i][:,0],pts_fp_i32[i][:,1],
                        edgecolors='none',facecolor='g',alpha=0.5)
        plt.show()

    return place_mask,H,Hinv

class RendererV10(object):

    def __init__(self, data_dir, max_time=None):
        self.text_renderer = tu.RenderFont(data_dir)
        self.colorizer = Colorize4(data_dir)

        self.min_char_height = 8  # px
        self.min_asp_ratio = 0.4  #

        self.max_text_regions = 7

        self.max_time = max_time

    def filter_regions(self, regions, filt):
        """
        filt : boolean list of regions to keep.
        """
        idx = np.arange(len(filt))[filt]
        for k in regions.keys():
            regions[k] = [regions[k][i] for i in idx]
        return regions

    def max_connected_region(self, mask):
        binary = mask.astype(np.int64)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # 找到最大区域并填充
        area = []

        for j in range(len(contours)):
            area.append(cv2.contourArea(contours[j]))

        max_idx = np.argmax(area)

        canvas = np.zeros_like(mask).astype(np.float32)
        mask = cv2.fillPoly(canvas, [contours[max_idx]], 1)

        return mask

    def filter_for_placement(self, xyz, seg, regions):
        filt = np.zeros(len(regions['label'])).astype('bool')
        masks, Hs, Hinvs = [], [], []
        for idx, l in enumerate(regions['label']):
            tmp = self.max_connected_region(seg == l)
            res = get_text_placement_mask(xyz, tmp, regions['coeff'][idx], pad=2)
            if res is not None:
                mask, H, Hinv = res
                masks.append(mask)
                Hs.append(H)
                Hinvs.append(Hinv)
                filt[idx] = True
        regions = self.filter_regions(regions, filt)
        regions['place_mask'] = masks
        regions['homography'] = Hs
        regions['homography_inv'] = Hinvs

        return regions

    def warpHomography(self, src_mat, H, dst_size):
        dst_mat = cv2.warpPerspective(src_mat, H, dst_size,
                                      flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return dst_mat

    def homographyBB(self, bbs, H, offset=None):
        """
        Apply homography transform to bounding-boxes.
        BBS: 2 x 4 x n matrix  (2 coordinates, 4 points, n bbs).
        Returns the transformed 2x4xn bb-array.

        offset : a 2-tuple (dx,dy), added to points before transfomation.
        """
        eps = 1e-16
        # check the shape of the BB array:
        t, f, n = bbs.shape
        assert (t == 2) and (f == 4)

        # append 1 for homogenous coordinates:
        bbs_h = np.reshape(np.r_[bbs, np.ones((1, 4, n))], (3, 4 * n), order='F')
        if offset != None:
            bbs_h[:2, :] += np.array(offset)[:, None]

        # perpective:
        bbs_h = H.dot(bbs_h)
        bbs_h /= (bbs_h[2, :] + eps)

        bbs_h = np.reshape(bbs_h, (3, 4, n), order='F')
        return bbs_h[:2, :, :]

    def bb_filter(self, bb0, bb, text):
        """
        Ensure that bounding-boxes are not too distorted
        after perspective distortion.

        bb0 : 2x4xn martrix of BB coordinates before perspective
        bb  : 2x4xn matrix of BB after perspective
        text: string of text -- for excluding symbols/punctuations.
        """
        h0 = np.linalg.norm(bb0[:, 3, :] - bb0[:, 0, :], axis=0)
        w0 = np.linalg.norm(bb0[:, 1, :] - bb0[:, 0, :], axis=0)
        hw0 = np.c_[h0, w0]

        h = np.linalg.norm(bb[:, 3, :] - bb[:, 0, :], axis=0)
        w = np.linalg.norm(bb[:, 1, :] - bb[:, 0, :], axis=0)
        hw = np.c_[h, w]

        # remove newlines and spaces:
        text = ''.join(text.split())
        assert len(text) == bb.shape[-1]

        alnum = np.array([ch.isalnum() for ch in text])
        hw0 = hw0[alnum, :]
        hw = hw[alnum, :]

        min_h0, min_h = np.min(hw0[:, 0]), np.min(hw[:, 0])
        asp0, asp = hw0[:, 0] / hw0[:, 1], hw[:, 0] / hw[:, 1]
        asp0, asp = np.median(asp0), np.median(asp)

        asp_ratio = asp / asp0
        is_good = (min_h > self.min_char_height
                   and asp_ratio > self.min_asp_ratio
                   and asp_ratio < 1.0 / self.min_asp_ratio)
        return is_good

    def get_min_h(selg, bb, text):
        # find min-height:
        h = np.linalg.norm(bb[:, 3, :] - bb[:, 0, :], axis=0)
        # remove newlines and spaces:
        text = ''.join(text.split())
        assert len(text) == bb.shape[-1]

        alnum = np.array([ch.isalnum() for ch in text])
        h = h[alnum]
        return np.min(h)

    def feather(self, text_mask, min_h):
        # determine the gaussian-blur std:
        if min_h <= 15:
            bsz = 0.25
            ksz = 1
        elif 15 < min_h < 30:
            bsz = max(0.30, 0.5 + 0.1 * np.random.randn())
            ksz = 3
        else:
            bsz = max(0.5, 1.5 + 0.5 * np.random.randn())
            ksz = 5
        return cv2.GaussianBlur(text_mask, (ksz, ksz), bsz)

    def mask_sample(self, mask):
        h, w = mask.shape
        x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        locs_grid = np.stack([x, y], -1)

        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.dilate(mask, kernel, 1)
        sample_mask = mask > 0
        coords = locs_grid[sample_mask]
        return coords

    def mask_sample_torch(self, mask, device):
        h, w = mask.shape
        y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        xy = torch.stack([x, y], -1).to(device)

        # kernel = torch.ones((10,10),torch.uint8)
        # mask = cv2.dilate(mask,kernel,1)
        coords = xy[mask > 0]
        return coords

    def mask2wbb(self, mask):
        cc = np.stack(np.where(mask > 0), 1)
        if len(cc) < 1:
            return np.zeros((4, 2))
        try:
            poly = Polygon(cc)
            box = np.moveaxis(np.array(poly.convex_hull.boundary.xy), 0, 1)[::-1]
            box = box[:, ::-1]
        except:
            rect = cv2.minAreaRect(cc.copy())
            box = np.array(cv2.boxPoints(rect))
            box = box[:, ::-1]

        return box

    def next_wbb(self, mask, wbb, seg_mask=None, M=None):
        next_wbbs = []
        for wbb_ in wbb:
            wbb_mask = cv2.fillPoly(np.zeros_like(mask), [wbb_.astype(np.int64)], 1)
            mask_ = wbb_mask * mask
            if M is not None:
                mask_ = cv2.warpPerspective(mask_, M.T, (mask_.shape[1], mask_.shape[0]))
            if seg_mask is not None:
                mask_ = mask_ * seg_mask
            next_wbb = self.mask2wbb(mask_)
            next_wbbs.append(next_wbb)

        return next_wbbs

    def flows_propagation(self, flows, seg_masks):
        _, h, w = flows[0].shape
        x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        xy = np.stack([x, y], 0)
        pflows = [np.zeros_like(flows[0])]
        for flow, mask in zip(flows, seg_masks):
            lflow = pflows[-1]
            sp = xy + lflow
            sp[0] = sp[0] / w
            sp[1] = sp[1] / h

            # the out of scope points and the out of mask points
            valid = ((sp > 0) & (sp < 1)).all(0)
            flow[:, ~mask] = -1e5

            flow = torch.from_numpy(flow[None])
            sp = torch.from_numpy(sp.reshape(2, -1).astype(np.float32)).permute(1, 0)[None]
            pflow = point_sample(flow, sp)

            pflow = pflow[0].reshape(2, h, w).numpy()

            pflow = lflow + pflow
            pflow[:, ~valid] = -1e5
            pflows.append(pflow)
        return pflows[1:]

    def flows_propagation_torch(self, flows, seg_masks):
        device = flows[0].device
        _, h, w = flows[0].shape
        y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        xy = torch.stack([x, y], 0).to(device)
        pflows = [torch.zeros_like(flows[0])]
        for i, flow in enumerate(flows):
            mask = seg_masks[i]
            pmask = seg_masks[i + 1]

            lflow = pflows[-1]
            sp = xy + lflow
            sp[0] = sp[0] / w
            sp[1] = sp[1] / h

            # the out of scope points and the out of mask points
            valid = ((sp > 0) & (sp < 1)).all(0)
            flow[:, ~mask] = -1e6

            sp = sp.reshape(2, -1).to(torch.float32).permute(1, 0)[None]
            pflow = point_sample(flow[None], sp)

            pflow = pflow[0].reshape(2, h, w)

            pflow = lflow + pflow
            pflow[:, ~valid] = -1e6
            # pflow[:,~pmask] = -1e5

            # moving out of next mask points
            psp = xy + pflow
            psp[0] = psp[0] / w
            psp[1] = psp[1] / h
            psp = psp.reshape(2, -1).to(torch.float32).permute(1, 0)[None]
            pvalid = point_sample(pmask[None, None].to(torch.float32), psp).reshape(h, w) > 0.99
            pflow[:, ~pvalid] = -1e6

            pflows.append(pflow)
        return pflows[1:]

    def wrap_trans(self, H, dx, dy):
        H2 = np.eye(3)
        H2[:2, :2] = H

        H1 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
        H3 = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]])
        nH = H1 @ H2 @ H3
        return nH

    def check_M(self, M):
        if np.max(np.abs(M)) > 640:
            # print('invalid transforms')
            # print(M)
            return False
        return True

    def place_text(self, imgs, flows, depths, segs, label, collision_mask, H, Hinv, key):
        t1 = time.time()
        rgb = imgs[key]  # the first img

        x = cv2.Sobel(rgb, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(rgb, cv2.CV_16S, 0, 1)
        abax = cv2.convertScaleAbs(x)
        abay = cv2.convertScaleAbs(y)
        edge_map = cv2.addWeighted(abax, 0.5, abay, 0.5, 0).sum(-1)

        # 控制形变
        times = 20
        max_medge = 1e6
        max_edge = 1e5
        # max_medge = 100
        # max_edge = 100
        min_area = 100
        text_dis = 1
        cr = 1
        abscr = 1e6
        method = 0

        # 控制贴图位置
        target_result = None
        for i in range(times):

            font = self.text_renderer.font_state.sample()
            font = self.text_renderer.font_state.init_font(font)

            try:
                # 保证文本粘贴位置偏向于中心区域
                # kernel = np.ones((2, 2), np.uint8)
                # tmp_collision_mask = cv2.dilate(collision_mask, kernel)
                render_res = self.text_renderer.render_sample(font, collision_mask)
            except:
                continue
            if render_res is None:  # rendering not successful
                continue
                # return #None
            else:
                text_mask, loc, bb, text = render_res

            # # update the collision mask with text:
            text_mask_org = copy.deepcopy(text_mask)
            # collision_mask += (255 * (text_mask>0)).astype('uint8')

            bb_orig = bb.copy()

            dx = bb.mean(1).mean(1)[0]
            dy = bb.mean(1).mean(1)[1]

            ct = np.array([dx, dy, 1]) @ Hinv.T
            ct[:2] = ct[:2] / ct[2]

            x1 = np.array([dx + 10, dy, 1]) @ Hinv.T
            x1[:2] = x1[:2] / x1[2]

            y1 = np.array([dx, dy + 10, 1]) @ Hinv.T
            y1[:2] = y1[:2] / y1[2]

            vx = (x1 - ct)[:2]
            vy = (y1 - ct)[:2]

            # 处理镜像
            H2 = np.eye(2)
            if np.linalg.det(H[:2, :2]) < 0:
                trans = np.array([[1, 0], [0, -1]])
                H2 = H2 @ trans
                vx = vx @ trans
                vy = vy @ trans

            # 处理翻转文本
            trans_lst = [np.array([[1, 0], [0, 1]]), np.array([[-1, 0], [0, -1]])]
            for trans in trans_lst:
                if (vx @ trans)[0] > 0 and (vy @ trans)[1] > 0:
                    H2 = H2 @ trans
                    vx = vx @ trans
                    vy = vy @ trans
                    break

            # 如果文本非正向，重新粘贴
            try:
                assert vx[0] > 0 and vy[1] > 0
                ovx = np.array([1, 0])
                nvx = vx / np.linalg.norm(vx)
                assert ovx @ nvx > 0.5
            except:
                # print(vx,vy)
                continue

            nH = self.wrap_trans(H2, dx, dy)

            H = nH @ H
            Hinv = Hinv @ inv(nH)

            text_mask = self.warpHomography(text_mask, H, rgb.shape[:2][::-1])

            # text mask reach the boundary
            if (text_mask[2:-2, 2:-2] > 0).sum() != (text_mask > 0).sum():
                continue

            bb = self.homographyBB(bb, Hinv)

            if not self.bb_filter(bb_orig, bb, text):
                # warn("bad charBB statistics")
                continue
                # return #None

            # get the minimum height of the character-BB:
            min_h = self.get_min_h(bb, text)

            # feathering:
            key_mask = self.feather(text_mask, min_h)

            wbb = [el.astype(np.int64) for el in
                   list(np.moveaxis(self.char2wordBB(bb, text), [0, 1, 2], [2, 1, 0]))]
            tmp_mask = cv2.fillPoly(np.zeros_like(edge_map).astype(np.uint8), wbb, 1)
            if tmp_mask.sum() < min_area:
                medge = 1e6
                maxedge = 1e6
            else:
                maxedge = (tmp_mask * edge_map).max()
                medge = (tmp_mask * edge_map).sum() / tmp_mask.sum()

            if medge < max_medge and maxedge < max_edge and (self.shelter_mask * (key_mask>0)).sum()<1:
                target_result = dict(key_mask=key_mask, text=text, bb=bb, medge=medge, maxedge=maxedge, min_h=min_h,
                                     text_mask_org=text_mask_org)
                break

        if target_result is None:
            return
        key_mask = target_result['key_mask']
        text = target_result['text']
        bb = target_result['bb']
        min_h = target_result['min_h']
        text_mask_org = target_result['text_mask_org']

        # 保存每一帧的贴图
        masks = [key_mask]

        # 用于估计的采样点
        kcs = self.mask_sample(key_mask)

        # 词级别的bbox
        wbbs = [list(np.moveaxis(self.char2wordBB(bb, text), [0, 1, 2], [2, 1, 0]))]
        key_wbb = self.next_wbb(key_mask, wbbs[0])
        wbbs[0] = key_wbb

        # 掩盖运动出目标区域的采样点
        # seg_masks = segs == label
        if label >= 1000:
            seg_masks = [el == label for el in segs]
            # seg_masks = [el == label for el in segs]
        else:
            seg_masks = [el < 1000 for el in segs]

        kernel = np.ones((2, 2), np.uint8)
        erode_seg_masks = [cv2.erode(el.astype(np.float32), kernel) for el in seg_masks]

        device = torch.device('cuda')
        flows = [torch.from_numpy(el).to(device) for el in flows]
        erode_seg_masks = [torch.from_numpy(el).to(device).to(torch.bool) for el in erode_seg_masks]

        forward_flows = flows[key:]

        forward_pflows = self.flows_propagation_torch(forward_flows, erode_seg_masks[key:])

        backward_flows = [-el for el in flows[:key]]
        backward_flows.reverse()

        backward_pflows = self.flows_propagation_torch(backward_flows, erode_seg_masks[:key + 1][::-1])

        _, h, w = forward_flows[0].shape
        kcs = torch.from_numpy(kcs).to(torch.float32).to(device)
        nkcs = copy.deepcopy(kcs)
        nkcs[..., 0] = kcs[..., 0] / w
        nkcs[..., 1] = kcs[..., 1] / h
        nkcs = nkcs[None]

        mindet = 1
        for idx, pflow in enumerate(forward_pflows):
            _, h, w = pflow.shape
            input = pflow[None]  # flow field
            output = point_sample(input, nkcs)

            valid_out = output[0].sum(0) > -1e4
            if valid_out.sum() < 10:
                next_mask = np.zeros_like(key_mask)
                next_wbb = [np.zeros_like(el) for el in key_wbb]
            else:

                npcs = kcs[valid_out] + output[:, :, valid_out][0].permute(1, 0)

                movement = npcs.norm(dim=1)
                meanm = movement.mean()
                stdm = movement.std()
                validm = (movement < meanm + cr * stdm) & (movement > meanm - cr * stdm) & (
                            movement < meanm + abscr) & (movement > meanm - abscr)
                npoints = npcs.shape[0]
                pcs = torch.cat([kcs[valid_out], torch.ones((npoints, 1)).to(device)], 1).to(torch.float64)[validm]
                npcs = torch.cat([npcs, torch.ones((npoints, 1)).to(device)], 1).to(torch.float64)[validm]

                try:
                    M = cv2.findHomography(pcs.cpu().numpy().astype('float32').copy(),
                                       npcs.cpu().numpy().astype('float32').copy(),
                                       method=method)[0].T
                    if not self.check_M(M):
                        return
                except:
                    return

                next_mask = cv2.warpPerspective(key_mask, M.T, (key_mask.shape[1], key_mask.shape[0]))
                if params['method']['shelter']:
                    seg_mask = seg_masks[key + idx + 1]
                else:
                    seg_mask = np.ones_like(seg_masks[key + idx + 1])
                next_mask = next_mask * seg_mask
                if (next_mask > 0).sum() < min_area:
                    next_mask = np.zeros_like(key_mask)
                    next_wbb = [np.zeros_like(el) for el in key_wbb]
                else:
                    next_wbb = self.next_wbb(copy.deepcopy(key_mask), key_wbb, seg_mask, M)

            masks.append(next_mask)
            wbbs.append(next_wbb)

        for idx, pflow in enumerate(backward_pflows):
            _, h, w = pflow.shape
            input = pflow[None]  # flow field
            output = point_sample(input, nkcs)

            valid_out = output[0].sum(0) > -1e4
            if valid_out.sum() < 10:
                next_mask = np.zeros_like(key_mask)
                next_wbb = [np.zeros_like(el) for el in key_wbb]
            else:

                npcs = kcs[valid_out] + output[:, :, valid_out][0].permute(1, 0)

                movement = npcs.norm(dim=1)
                meanm = movement.mean()
                stdm = movement.std()
                validm = (movement < meanm + cr * stdm) & (movement > meanm - cr * stdm) & (
                            movement < meanm + abscr) & (movement > meanm - abscr)
                npoints = npcs.shape[0]
                pcs = torch.cat([kcs[valid_out], torch.ones((npoints, 1)).to(device)], 1).to(torch.float64)[validm]
                npcs = torch.cat([npcs, torch.ones((npoints, 1)).to(device)], 1).to(torch.float64)[validm]

                try:
                    M = cv2.findHomography(pcs.cpu().numpy().astype('float32').copy(),
                                           npcs.cpu().numpy().astype('float32').copy(),
                                           method=method)[0].T
                    if not self.check_M(M):
                        return
                except:
                    return

                next_mask = cv2.warpPerspective(key_mask, M.T, (key_mask.shape[1], key_mask.shape[0]))

                if params['method']['shelter']:
                    seg_mask = seg_masks[key - idx - 1]
                else:
                    seg_mask = np.ones_like(seg_masks[key - idx - 1])
                next_mask = next_mask * seg_mask
                if (next_mask > 0).sum() < min_area:
                    next_mask = np.zeros_like(key_mask)
                    next_wbb = [np.zeros_like(el) for el in key_wbb]
                else:
                    next_wbb = self.next_wbb(copy.deepcopy(key_mask), key_wbb, seg_mask, M)

            masks.insert(0, next_mask)
            wbbs.insert(0, next_wbb)

        t5 = time.time()

        if params['method']['postprocess'] == 'hw':
            min_h = 15
            # 删除持续时间过短的文本
            valid_wbbs = [False] * len(wbbs)
            wh_wbbs = []
            for wbb in wbbs:
                try:
                    wh = cv2.minAreaRect(np.concatenate(wbb, 0).astype(np.int64))[1]
                except:
                    wh = (0, 0)
                wh_wbbs.append(wh)
            kwh = wh_wbbs[key]
            for i, wh in enumerate(wh_wbbs[key:]):
                if (min(wh) > min_h) & (max(wh) / (min(wh) + 1e6) < 10):
                    valid_wbbs[key + i] = True
                else:
                    break
            for i, wh in enumerate(wh_wbbs[:key][::-1]):
                if (min(wh) > min_h) & (max(wh) / (min(wh) + 1e6) < 10):
                    valid_wbbs[key - 1 - i] = True
                else:
                    break

            # valid_wbbs = [el for el in wbbs if not el[0].sum() < 1]
            if sum(valid_wbbs) < 3:
                return

            valid_masks = []
            wbbs = [el if is_valid else [np.array((0, 2), np.int64)] * len(el) for el, is_valid in
                    zip(wbbs, valid_wbbs)]
            for mask, is_valid in zip(masks, valid_wbbs):
                if is_valid:
                    valid_masks.append(mask)
                else:
                    valid_masks.append(np.zeros_like(mask))
            masks = valid_masks
        else:
            valid_wbbs = [el for el in wbbs if not el[0].sum() < 1]
            if len(valid_wbbs) < 3:
                return

        try:
            # pad_imgs = imgs
            # pad_masks = masks
            pad_imgs = [np.pad(el, ((50,50), (50, 50),(0,0)), 'edge') for el in imgs]
            pad_masks = [np.pad(el, ((50,50), (50, 50)), 'constant') for el in masks]
            imgs_final = self.colorizer.batch_color(pad_imgs, pad_masks, np.array([min_h]))
            imgs_final = [el[50:-50,50:-50] for el in imgs_final]
        except:
            return

        # update collision_mask
        kernel = np.ones((text_dis, text_dis), np.uint8)
        text_mask_org = cv2.dilate(text_mask_org.astype(np.float32), kernel)
        collision_mask += (255 * (text_mask_org > 0)).astype('uint8')

        if not params['method']['overlap']:
            self.shelter_mask = self.shelter_mask + (key_mask>0).astype(np.int64)

        print('paint text with {:.2f}s'.format(t5 - t1))

        return imgs_final, text, wbbs, collision_mask

    def save(self, masks):
        import os
        save_dir = 'cache/pipeline/text_mask'
        for i, mask in enumerate(masks):
            save_img = os.path.join(save_dir, '{:0>8d}.jpg')
            cv2.imwrite(save_img.format(i), mask)
        return

    def get_num_text_regions(self, nregions):
        # return nregions
        nmax = min(self.max_text_regions, nregions)
        if np.random.rand() < 0.10:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(5.0, 1.0)
        return int(np.ceil(nmax * rnd))

    def char2wordBB(self, charBB, text):
        """
        Converts character bounding-boxes to word-level
        bounding-boxes.

        charBB : 2x4xn matrix of BB coordinates
        text   : the text string

        output : 2x4xm matrix of BB coordinates,
                 where, m == number of words.
        """
        wrds = text.split()
        bb_idx = np.r_[0, np.cumsum([len(w) for w in wrds])]
        wordBB = np.zeros((2, 4, len(wrds)), 'float32')

        for i in range(len(wrds)):
            cc = charBB[:, :, bb_idx[i]:bb_idx[i + 1]]

            # fit a rotated-rectangle:
            # change shape from 2x4xn_i -> (4*n_i)x2
            cc = np.squeeze(np.concatenate(np.dsplit(cc, cc.shape[-1]), axis=1)).T.astype('float32')
            rect = cv2.minAreaRect(cc.copy())
            box = np.array(cv2.boxPoints(rect))

            # find the permutation of box-coordinates which
            # are "aligned" appropriately with the character-bb.
            # (exhaustive search over all possible assignments):
            cc_tblr = np.c_[cc[0, :],
                            cc[-3, :],
                            cc[-2, :],
                            cc[3, :]].T
            perm4 = np.array(list(itertools.permutations(np.arange(4))))
            dists = []
            for pidx in range(perm4.shape[0]):
                d = np.sum(np.linalg.norm(box[perm4[pidx], :] - cc_tblr, axis=1))
                dists.append(d)
            wordBB[:, :, i] = box[perm4[np.argmin(dists)], :].T

        return wordBB

    def render_text(self, src_imgs, flows, depths, segs, areas, labels, key):
        """
        rgb   : HxWx3 image rgb values (uint8)
        depth : HxW depth values (float)
        seg   : HxW segmentation region masks
        area  : number of pixels in each region
        label : region labels == unique(seg) / {0}
               i.e., indices of pixels in SEG which
               constitute a region mask
        ninstance : no of times image should be
                    used to place text.

        @return:
            res : a list of dictionaries, one for each of
                  the image instances.
                  Each dictionary has the following structure:
                      'img' : rgb-image with text on it.
                      'bb'  : 2x4xn matrix of bounding-boxes
                              for each character in the image.
                      'txt' : a list of strings.

                  The correspondence b/w bb and txt is that
                  i-th non-space white-character in txt is at bb[:,:,i].

            If there's an error in pre-text placement, for e.g. if there's
            no suitable region for text placement, an empty list is returned.
        """
        seg = segs[key]
        area = areas[key]
        label = labels[key]
        depth = depths[key]

        self.shelter_mask = np.zeros_like(seg)

        try:
            # depth -> xyz
            xyz = su.DepthCamera.depth2xyz(depth)

            # find text-regions:
            regions = TextRegions.get_regions(xyz, seg, area, label)

            # find the placement mask and homographies:
            regions = self.filter_for_placement(xyz, seg, regions)

            # finally place some text:
            nregions = len(regions['place_mask'])
            if nregions < 1:  # no good region to place text on
                return []
        except:
            # failure in pre-text placement
            # import traceback
            traceback.print_exc()
            return []

        res = []
        place_masks = copy.deepcopy(regions['place_mask'])

        idict_list = []
        idict_format = {'img': [], 'charBB': None, 'wordBB': None, 'txt': None}

        # m = self.get_num_text_regions(nregions)# np.arange(nregions)#min(nregions, 5*ninstance*self.max_text_regions))
        m = nregions
        reg_idx = np.arange(min(2 * m, nregions))
        np.random.shuffle(reg_idx)
        reg_idx = reg_idx[:m]

        placed = False
        imgs = [el.copy() for el in src_imgs]
        itexts = []
        ibbs = []

        # process regions:
        num_txt_regions = len(reg_idx)
        NUM_REP = params['method']['region_reuse']  # re-use each region three times:
        # NUM_REP = meta['NUM_REP']
        reg_range = np.arange(NUM_REP * num_txt_regions) % num_txt_regions
        for idx in reg_range:
            ireg = reg_idx[idx]
            try:
                i = 0
                while i < 1:

                    txt_render_res = self.place_text(imgs,
                                                     copy.deepcopy(flows),
                                                     copy.deepcopy(depths),
                                                     copy.deepcopy(segs),
                                                     regions['label'][ireg],
                                                     place_masks[ireg],
                                                     regions['homography'][ireg],
                                                     regions['homography_inv'][ireg],
                                                     key)
                    if txt_render_res is not None:
                        break
                    i = i + 1
            except TimeoutException as msg:
                print(msg)
                continue
            except:
                traceback.print_exc()
                # some error in placing text on the region
                continue

            if txt_render_res is not None:
                placed = True
                imgs, text, bb, collision_mask = txt_render_res
                # update the region collision mask:
                place_masks[ireg] = collision_mask
                # store the result:
                itexts.append(text)
                ibbs.append(bb)

        ibbs = list(zip(*ibbs))

        if placed:
            for img, ibb in zip(imgs, ibbs):
                # at least 1 word was placed in this instance:
                idict = copy.deepcopy(idict_format)
                idict['img'] = img
                idict['txt'] = itexts
                wordBB = []
                for el in ibb:
                    wordBB.extend(el)
                idict['wordBB'] = wordBB

                idict_list.append(idict)
        return idict_list

def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def motion_blur(image, degree=12, angle=45):
  image = np.array(image)
  # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
  M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle+45, 1)
  motion_blur_kernel = np.diag(np.ones(degree))
  motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
  motion_blur_kernel = motion_blur_kernel / degree
  blurred = cv2.filter2D(image, -1, motion_blur_kernel)
  # convert to uint8
  cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
  blurred = np.array(blurred, dtype=np.uint8)
  return blurred




# -*- coding: utf-8 -*-
import os
import sys, getopt
from pycocotools.coco import COCO, maskUtils
import cv2
import numpy as np
from tqdm import tqdm

def main():
    jsonfile = '../data/train/annotations/train.json'
    dataset_dir = '../data/train/images/'
    save_dir = '../data/train/pred_train/'
    os.makedirs(save_dir, exist_ok=True)

    coco = COCO(jsonfile)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds(catIds=catIds)  # 图片id，许多值

    for i in tqdm(range(len(imgIds))):
        # if i%100 == 0:
        #     print(i,"/", len(imgIds))
        img = coco.loadImgs(imgIds[i])[0]
        cvImage = cv2.imread(os.path.join(dataset_dir, img['file_name']), -1)
        # cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
        # cvImage = cv2.cvtColor(cvImage, cv2.COLOR_GRAY2BGR)

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        polygons = []
        color = []
        for ann in anns:
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                        poly_list = poly.tolist()
                        polygons.append(poly_list)
                else:
                    exit()
                    print("-------------")
                    # mask
                    t = imgIds[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                    m = maskUtils.decode(rle)

                    if ann['iscrowd'] == 1:
                        color_mask = np.array([255, 0, 0])
                    if ann['iscrowd'] == 0:
                        color_mask = np.array([255, 255, 0])

                    mask = m.astype(np.bool)
                    cvImage[mask] = cvImage[mask] * 0.7 + color_mask * 0.3

        point_size = 2
        thickness = 2
        for key in range(len(polygons)):
            ndata = polygons[key]
            for k in range(len(ndata)):
                data = ndata[k]
                cv2.circle(cvImage, (int(data[0]), int(data[1])), point_size, (0, 255, 255), thickness)
        cv2.imwrite(os.path.join(save_dir, img['file_name']), cvImage)


if __name__ == "__main__":
   main()
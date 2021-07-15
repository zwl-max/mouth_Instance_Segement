# -*- coding: utf-8 -*-
import os
import sys, getopt
from pycocotools.coco import COCO
import cv2
import numpy as np
from tqdm import tqdm
from mmdet.core import mouth_classes

def main():
    jsonfile = '../data/train/annotations/train.json'
    dataset_dir = '../data/train/images/'
    save_dir = '../data/train/check/'
    os.makedirs(save_dir, exist_ok=True)

    coco = COCO(jsonfile)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds(catIds=catIds)  # 图片id，许多值

    for i in tqdm(range(len(imgIds))):
        img = coco.loadImgs(imgIds[i])[0]
        cvImage = cv2.imread(os.path.join(dataset_dir, img['file_name']), -1)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for anno in anns:
            x1, y1, w, h = anno['bbox']
            x2 = x1 + w
            y2 = y1 + h
            cat_id = anno['category_id'] - 1
            cv2.rectangle(cvImage, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(cvImage, mouth_classes()[cat_id],
                        (int(x1), int(y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=1
                        )
            cv2.imwrite(os.path.join(save_dir, img['file_name']), cvImage)

if __name__ == '__main__':
    main()
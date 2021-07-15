"""
get semantic segmentation annotations from coco data set.
"""
from PIL import Image
import imgviz
import numpy as np
import argparse
import os, cv2
import tqdm
from pycocotools.coco import COCO
import shutil

np.set_printoptions(threshold=np.inf)
num_classes = 5 + 1  # 因为mouth包括其他部分

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)

def save_gray_mask(mask, save_path):
    mask[mask == 0] = 255  # 背景0 转变为 255
    mask[mask != 255] -= 1
    mask[mask >= num_classes] = 255
    global seglabel
    seglabel = max(len(np.unique(mask)), seglabel)
    cv2.imwrite(save_path, mask)

def main(args):
    annotation_file = os.path.join(args.input_dir, 'annotations', '{}.json'.format(args.split))
    os.makedirs(os.path.join(args.input_dir, 'SegmentationClass2'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'SegmentationClassGray'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'JPEGImages'), exist_ok=True)

    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = np.zeros_like(coco.annToMask(anns[0]))
        for j in range(len(anns)):
            if anns[j]['category_id'] == 2:
                mask += coco.annToMask(anns[j]) * anns[j]['category_id']
                # mask = coco.annToMask(anns[0]) * anns[0]['category_id']
                # for i in range(len(anns) - 1):
                #     mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
                # img_origin_path = os.path.join(args.input_dir, 'images', img['file_name'])
                # img_output_path = os.path.join(args.input_dir, 'JPEGImages', img['file_name'])
                seg_output_path = os.path.join(args.input_dir, 'SegmentationClass2',
                                               img['file_name'].replace('.jpg', '.png'))
                # gray_seg_output_path = os.path.join(args.input_dir, 'SegmentationClassGray',
                #                                img['file_name'].replace('.jpg', '.png'))
                # shutil.copy(img_origin_path, img_output_path)
                save_colored_mask(mask, seg_output_path)         # 调速板模式
                # save_gray_mask(mask, gray_seg_output_path)       # 灰度模式
    print('seg label', seglabel)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../data/train", type=str,
                        help="input dataset directory")
    parser.add_argument("--split", default="train", type=str,
                        help="train or val")
    return parser.parse_args()


if __name__ == '__main__':
    # label_map = []
    # for i in range(1, 10):
    #     mask = np.ones((32, 32)) * i
    #     label_map.append(mask)
    # label_map = np.concatenate(label_map, axis=-1)
    # save_colored_mask(label_map, 'color.png')
    # exit()
    seglabel = 0
    args = get_args()
    main(args)

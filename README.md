## 团队介绍：
队伍名称：mask \
队伍成员： 张卫良（江南大学研二）研究方向：2D目标检测

### B榜结果： 0.68726892619， 排名第二

## 环境信息
- sys.platform: linux(Ubuntu 7.5.0-3ubuntu1~18.04) 
- Python: 3.7.4 (default, Aug 13 2019, 20:35:49) [GCC 7.3.0] 
- GPU 0: GeForce RTX 2080 Ti 
- PyTorch: 1.6.0+cu101 
- TorchVision: 0.7.0+cu101 
- CUDA  10.1 
- CuDNN 7.6.3 
- OpenCV: 4.5.2 
- MMCV: 1.2.4 
- MMDetection: 2.11.0+41bb93f

## 最终的配置文件
- [swin-tiny](code/mouth_configs/cascade_mask_rcnn_swin_tiny.py)
- [swin-small](code/mouth_configs/cascade_mask_rcnn_swin_small.py)
- [swin-base](code/mouth_configs/cascade_mask_rcnn_swin_base.py)
- [swa-swin-tiny](code/swa_configs/swa_cascade_mask_rcnn_swin_tiny_fpn.py)
- [swa-swin-small](code/swa_configs/swa_cascade_mask_rcnn_swin_small_fpn.py)
- [swa-swin-base](code/swa_configs/swa_cascade_mask_rcnn_swin_base_fpn.py)

## 测试
- `sh main.sh `
- 得到B榜的测试结果 prediction_result/result.json

## 训练
- `sh train.sh`

## 解决方案（实验迭代过程）：
### 基于[swin-transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) 进行算法迭代
注意：测试采用的是多尺度测试(TTA)
- img_scale=[(1333, 480), (1333, 640), (1333, 800)] + hflip(水平翻转)


1. baseline: cascade_mask_rcnn_swin_small_fpn + ms(480-672) + nms(0.5) + epoch:12
    - 其中， ms指的是多尺度训练， nms(0.5): 使用nms, iou_thre等于0.5
- A榜结果：0.67286122814 

2. 基于baseline: 为了提高模型的召回率，用soft_nms替换nms, 其中iou_thre:0.5， 以及, 为了使模型对小物体有更好的性能，增大多
尺度训练的范围 ms(480-800)
- A榜结果：0.68311035012， 提升1.1%

3. 基于2
    - 因为该任务的评价指标是 AP(IoU=0.5:0.05:0.95)， 所以要求更精确的检测结果。 
基于此， 调整cas_iou(0.5, 0.6, 0.7) --> cas_iou(0.55, 0.65, 0.75)
- A榜结果：0.68516692196， 提升0.2%

4. 基于3
    - 考虑到[FPN](https://arxiv.org/abs/1612.03144) (特征金字塔网络)只有一条从上到下的路径，对于不同层的特征融合能力比较差；
      为了增强不同层之间的信息融合，在FPN的基础上，增加一条从下到上的路径，即[PAFPN](https://arxiv.org/abs/1803.01534)
- A榜结果：0.68235109092，  降低0.3%， 结果反而降低了。。。 说明在neck上做改进，并不能带来提升。

5. 基于3
    - 为了让模型更充分的利用训练数据， 增加训练轮数，epoch:20
- A榜结果：0.68555370639  提升0.04%， 提升微乎其微。。。

6. 基于3
    - swin_base模型bbox_head部分的回归损失使用的是giou loss,  所以依次尝试iou loss, smooth_l1 loss。
- 通过实验，iou loss使结果降低， 而smooth l1 loss使精度提升了0.1%， 0.68665009    

7. 基于6
    - rpn部分的MaxIoUAssigner换成[ATSSAssigner](https://arxiv.org/abs/1912.02424)
- A榜结果：0.68672692813  提升很小。。。

8. 基于7
    - 为了使每个roi获得全局上下文信息， 加入gc(global context)。
- A榜结果：0.68421612819  降了0.35%， 一味的加入上下文信息，对此任务并没有用。

9. 基于7
    - 根据论文[SWA](https://arxiv.org/abs/2012.12645) 中的介绍， 这是一种无痛涨点法。 采用余弦退火学习率额外
      训练模型12个epoch，然后平均每个epoch训练得到的weights作为最终的模型。  
      swa + epoch:12
- A榜结果：0.68918544278  提升0.24%, 果然有用。

开始尝试其他模型：(为了模型融合)
10. cascade_mask_rcnn_swin_tiny_fpn + casiou(0.55-0.75) + ms(480-800) + AdamW(0.0001) + weight_decay(0.05) + fp16 
+ bs:2 + soft_nms(0.5) + smoothl1 loss + atss(k=9) + epoch:12
- A榜结果：0.69098631098  惊讶~~~， swin-tiny这么强的吗。。。

11. 基于10.
    - 既然，swin-tiny这么强， 那试试swa + epoch:12
- A榜结果：0.68884048720  结果反而降了0.21%， 可见， swa也不是万能的， 还是要看任务以及模型。。

12. 尝试swin-base
cascade_mask_rcnn_swin_base_fpn + casiou(0.55-0.75) + ms(480-800) + AdamW(0.0001) + weight_decay(0.05) + fp16 + bs:1 
+ soft_nms(0.5) + smoothl1 loss + atss(k=9) + epoch:12
- A榜结果：0.68463508458

13. 基于12
    - swa + epoch:12
- A榜结果：0.68434781735
  
单模最高的是swin-tiny, A榜达到0.69098631098

为了提高模型的鲁棒性， 进行模型融合
- a. 
    - swin_tiny 0.69098631098
    - swin_small(swa)  0.68918544278
    - 采用soft_nms(0.5)方式
    - A榜结果：0.69504450825   提升0.41%
- b.
    - swin_tiny 0.69098631098
    - swin_small(swa)  0.68918544278
    - swin_base  0.68463508458
    - 采用soft_nms(0.5)方式
    - A榜结果：0.69234975154 
- c.
    - swin_tiny 0.69098631098   
    - swin_small(swa)  0.68918544278
    - 融合方式： wbf: [1, 1]   iou_thr:0.6   conf_type:avg
    - A榜结果： 0.67128827358
- d.
    - swin_tiny 0.69098631098   
    - swin_small(swa)  0.68918544278
    - 融合方式：nmw: [1, 1]   iou_thr: 0.5
    - A榜结果： 0.69068316921
- e.
    - swin_tiny        0.69098631098 
    - swin_small(swa)  0.68918544278
    - swin_base(swa)   0.68434781735
    - 融合方式：soft_nms(0.5)
    - A榜结果： 0.69531461879（A榜最高）
    — B榜结果： 0.68663301
      
为了进一步提高精度，采用4尺度测试 + hflip(水平翻转)
- test scale: [(1333, 480), (1333, 576), (1333, 704), (1333, 800)]   
    - swin_tiny        0.69098631098 
    - swin_small(swa)  0.68918544278
    - swin_base(swa)   0.68434781735
    - 融合方式：soft_nms(0.5)
    — B榜结果： 0.68726892619  最终的B榜成绩
    
### [coco模型权重下载链接](code/download_weight.sh)

## 参考链接
[实例分割模型融合](https://github.com/boliu61/open-images-2019-instance-segmentation/blob/master/mmdetection/tools/ensemble_test.py)

## 代码目录
```
/data 
├── raw_data            (数据集)
|   |—— train     (训练集目录)    
|   |—— test      (测试集目录）   
├── user_data           (用户中间数据目录)
├── prediction_result   (预测结果输出文件夹)
├── code                (代码文件夹）
├── main.sh             (预测脚本)
|—— train.sh            (训练脚本）
|—— run.sh              (启动预测脚本）
└── README.md

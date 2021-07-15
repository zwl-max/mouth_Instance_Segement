python tools/train.py mouth_configs/cascade_mask_rcnn_swin_base.py --no-validate
sleep 1

echo "开始swa training"
python tools/train.py swa_configs/swa_cascade_mask_rcnn_swin_base_fpn.py --no-validate

echo "训练完成"
mv /data/user_data/swa_cascade_mask_rcnn_swin_base_fpn_atss_smoothl1_casiou0.55-0.75_e12/swa_epoch_12.pth /data/code/ensemble_configs/swa_swin_base-aea9dc1f.pth
python tools/train.py mouth_configs/cascade_mask_rcnn_swin_tiny.py --no-validate
sleep 1

echo "训练完成"
mv /data/user_data/cascade_mask_rcnn_swin_tiny_fpn_atss_smoothl1_casiou0.55-0.75_e12/epoch_12.pth /data/code/ensemble_configs/swin_tiny_e12-f4677e00.pth
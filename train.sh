echo "编译apex"
cd /data/code/apex
python setup.py install
sleep 1

#echo "开始下载coco模型权重"
#cd /data/code
#sh download_weight.sh
#sleep 1

echo "开始训练第一个模型"
cd /data/code
sh train_model1.sh
sleep 1

echo "开始训练第二个模型"
cd /data/code
sh train_model2.sh
sleep 1

echo "开始训练第三个模型"
cd /data/code
sh train_model3.sh

echo "所有模型训练完成，可以开始预测！！"
echo "预测部分请执行 sh main.sh!!!"

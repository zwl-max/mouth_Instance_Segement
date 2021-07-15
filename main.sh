echo "编译apex"
cd /data/code/apex
python setup.py install
sleep 1

echo "生成testA.json"
cd /data/code/
python tools/generate_test_json.py
sleep 1

echo "开始预测"
python tools/test_ensemble.py
echo "预测完成！"

cp /data/prediction_result/result.segm.json /data/prediction_result/result.json
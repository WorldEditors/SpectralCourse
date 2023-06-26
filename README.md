# 高阶谱分析课程实验使用说明
# 下载相关的代码依赖
```script
pip install -r requirements.txt
```
# 下载RadioML 2018.01A数据集
https://www.kaggle.com/datasets/pinxau1000/radioml2018

# 对数据集进行统计
```script
python data_processing.py -raw_data [目录]/GOLD_XYZ_OSC.0001_1024.hdf5 -task statistics
```

# 对数据集进行拆分
```script
python data_processing.py -raw_data [目录]/GOLD_XYZ_OSC.0001_1024.hdf5 -task data_preprocessing --generate_dir ./data_tst
```

# 对数据集进行训练和评测
```script
bash run_train.py
bash run_test.py
```

# SimAttention
Contrastive Learning,  Patch Attention,  Point Cloud

Version 01 运行指南：(Single GPU Version, Network Structure 1 without MLP layer)

- 下载ModelNet数据集，并存为modelnet40_normal_resampled
- 下载地址： https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
- 修改train_version_0.py中19行数据的地址，也就是上面这个文件的解压包地址
- 运行train_version_0.py

Version 02 运行指南：(Multi-GPU Version, Network Structure 1 without MLP layer)
- 同上数据准备
- 可能需要修改train_multi_gpus_v1.py中142行数据地址
- 149行，就是用到的GPU个数，大于2即可，default后面可以修改
- 直接运行即可脚本即可

Version 02 PLUS: 更新了model中一个小部分，不需要改参数之类的
- 同Version02准备，但是model.py更新了SimAttention_7，需要对应更新model脚本
- train_multi_gpus_v1.py第10, 81行，SimAttention_5改成SimAttention_7
- bs设置为16，运行50个epoch，保存结果作为下一步的输入；

Version 03 运行指南：多GPU-用来训练分类网络，和评估工作
- 同上数据准备，运行train_multi_cls.py
- 在network里面新增shape_classifier脚本文件
- 273行存储的是上一次训练出来的模型参数，需要根据存储位置修改
- 其他基本不变，bs设置为32，epoch设置为50

Version 04 运行指南：
- 项目中增加脚本文件cube_model.py 这个和model同一层级
- 在utils文件夹中新增cubes.py脚本文件，和crops.py同一层级
- 在train_multi_gpus_v1.py中修改
  - from cube_model import SimAttention_All_Cubes
  - from utils.cubes import b_get_cubes
  - model初始化的时候 model = SimAttention_All_Cubes(aug_method, b_FPS, b_get_cubes, online_encoder, crossed_method)
  - 看一下单epoch运行的时间

Version 05 运行指南：
- 项目增加文件knn_model.py文件位置和model.py在同一层；
- utils文件夹中的crops.py有部分函数更新，直接下载最新版的即可；
- 在train_multi_gpus_v1.py中修改：
  - from knn_model import SimAttention_KNN
  - model在初始化的时候：model = SimAttention_KNN(aug_method, b_FPS, new_k_patch, online_encoder, crossed_method)

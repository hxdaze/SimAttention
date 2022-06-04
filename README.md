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

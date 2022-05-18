# SimAttention
Contrastive Learning,  Patch Attention,  Point Cloud

Version 01 运行指南：(Single GPU Version, Network Structure 1 without MLP layer)

- 下载ModelNet数据集，并存为modelnet40_normal_resampled
- 下载地址： https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
- 修改train_version_0.py中19行数据的地址，也就是上面这个文件的解压包地址
- 运行train_version_0.py

Version 02 运行指南：(Multi-GPU Version, Network Structure 1 without MLP layer)
- 同上数据准备
- 修改train_multi_gpus_v1.py中122行数据地址，同上
- 命令行命令：python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_gpus_v1.py

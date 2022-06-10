因为代码主要还是在服务器上运行，和本地有许多不同之处，所以在此记录。

1. SimAttention在import的时候需要去掉；
2. to(device) 改成 .cuda()
3. 用到permute的地方，容易引发DDP警告，后面需要加上.contiguous()，来保证不会损坏性能（实验表明，加上之后loss会下降更快）

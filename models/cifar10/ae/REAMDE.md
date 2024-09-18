|model name|cycle mse auc-roc|
|--|--|
|MemAEV1|0.566|
|MemAEV2|0.558|
## 
通过实验已经证明，增加动态优化器使实验效果更加好，原memaev2验证

[史上最全学习率调整策略lr_scheduler](https://zhuanlan.zhihu.com/p/538447997)
## mem_ae
### MemAEV7
损失函数: l1loss + ssim_loss
重构误差: mse_scores, ssim_scores, ssim_l1_scores

结果来看，ssim作为损失函数，mse重构误差有较好效果，但是ssim重构误差效果比较差，看来ssim虽然作为人类视觉的第一感受，但是不能很好区分开异常数据和正常数据的重构误差，
## aev2_v2
aev2_v2里面的卷积网络修改初始化权重，批正则化，affine=False
## 20230817
aev1: 原有Deep SVDD神经网络
aev2: 原有记忆模块神经网络模型
aev3: OCGAN-Pytorch版本里面的自编码器网络

aev[0-9]下划线之后是在原有神经网络上的改进类型
##
ae

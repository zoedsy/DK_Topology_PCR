# Distilling Knowledge from Topological Representations 

This is the official Pytorch implementation of "Distilling Knowledge from Topological Representations for Pathological Complete Response Prediction" (MICCAI 2022 early accept).



### Overview

![image-20221002225508913](C:\Users\zoedusy\AppData\Roaming\Typora\typora-user-images\image-20221002225508913.png)



### Dataset

please download the dataset through this https://wiki.cancerimagingarchive.net/display/Public/ISPY1#20643859f2ec9d7881eb4a408ae1f347ea462beb

### How to start

1. download the dataset and preprocess the dataset
2. extract 3D betti curves of the dataset and normalize them
3. edit the root name in the codebase and put both the dataset and extracted betti curve in the root which has been edited
4. pip install -r requirements.txt
5. python cv_densenet_kd_1.py

### Citation

If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{du2022distilling,
  title={Distilling Knowledge from Topological Representations for Pathological Complete Response Prediction},
  author={Du, Shiyi and Lao, Qicheng and Kang, Qingbo and Li, Yiyue and Jiang, Zekun and Zhao, Yanfeng and Li, Kang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={56--65},
  year={2022},
  organization={Springer}
}
```


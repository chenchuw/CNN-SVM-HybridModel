# Boston University SPRG2021 EC503 A1 Final Project
Team member: Bingquan Cai, Chuwei Chen, Jiawei Zhao, Xiaowei Ge, Yuxiang Wan

---
## Reasoning
Recognizing the cell identity is usually through slow and costly sequencing-based methods. Image-based methods for cell identification are cost-efficient and fast, but usually difficult to realize by relating the 2D shape information to identity. By applying the machine learning methods, we could extract features more efficiently and improve the image-based methods classification precision. 

## Introduction
Inspired by the method, region-based convolutional neural network (R-CNN)<sup>1,2</sup>, we would like to apply CNN-SVM for feature extraction and classification. Our goal is to realize the algorithm in "Handwriting recognition on form documents using convolutional neural networks and support vector machines (CNN-SVM)"<sup>4</sup> by following the tasks listed below. By transferring the learning ability of the algorithm, we aim to extend this method for bacteria species identification based on the shapes. The backup plan is using k-nearest neighbors (kNN) with metric learning, such as large margin nearest neighbor<sup>5,7</sup>, which also shows classification ability, especially expertise on adaptation to different datasets.

## Resource & Reference
[1] Girshick, Ross, et al. "Rich feature hierarchies for accurate object detection and semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. Available: https://arxiv.org/abs/1311.2524

[2] Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems 28 (2015). Available: https://arxiv.org/abs/1506.01497

[3] Niu, Xiao-Xiao, and Ching Y. Suen. "A novel hybrid CNN–SVM classifier for recognizing handwritten digits." Pattern Recognition 45.4 (2012): 1318-1325. 
Available: https://www.sciencedirect.com/science/article/abs/pii/S0031320311004006

[4] Fanany, Mohamad Ivan. "Handwriting recognition on form document using convolutional neural network and support vector machines (CNN-SVM)." 2017 5th international conference on information and communication technology (ICoIC7). IEEE, 2017. 
Available: https://www.researchgate.net/publication/316350087_Handwriting_Recognition_on_Form_Document_Using_Convolutional_Neural_Network_and_Support_Vector_Machines_CNN-SVM

[5] Weinberger, Kilian Q., and Lawrence K. Saul. "Distance metric learning for large margin nearest neighbor classification." Journal of machine learning research 10.2 (2009). 
Available: https://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf

[6] Kulis, Brian. "Metric learning: A survey." Foundations and Trends® in Machine Learning 5.4 (2013): 287-364. Available: https://people.bu.edu/bkulis/pubs/ftml_metric_learning.pdf

[7] Abu Alfeilat, Haneen Arafat, et al. "Effects of distance measure choice on k-nearest neighbor classifier performance: a review." Big data 7.4 (2019): 221-248.
Available: https://www.liebertpub.com/doi/pdfplus/10.1089/big.2018.0175

[8] Li, Chao, et al. "Using the K-nearest neighbor algorithm for the classification of lymph node metastasis in gastric cancer." Computational and mathematical methods in medicine 2012 (2012).
Available: https://downloads.hindawi.com/journals/cmmm/2012/876545.pdf

[9] Xuzhenqi, “Xuzhenqi/CNN: This is a Matlab-code implementation of Convolutional Neural Network,” GitHub. [Online]. 
Available: https://github.com/xuzhenqi/cnn. [Accessed: 03-May-2022]. 

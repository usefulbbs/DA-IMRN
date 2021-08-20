# DA-IMRN
The fusion of spectral-spatial information has emerged as the main stream of hyperspectral image classification, especially with the advance of deep learning. However, due to the insufficient samples, the recent feature fusion methods might loss the joint interactions, impeding the further improvement of classification accuracy. In this paper, we propose a dual-attention-guided interactive multi-scale residual network (DA-IMRN) to explore the joint spectral-spatial information and assign the label for each pixel of hyperspectral image without information leakage. First, two branches focusing on the spatial and spectral information separately are employed for feature extraction. A bidirectional-attention mechanism, including spatial-channel attention and spectral attention is utilized to guide the interactive feature learning between two branches, and promote refined feature maps. In addition, we extract deep multi-scale features corresponding to multiple receptive fields from limited samples via a multi-scale spectral/spatial residual block, and improve the segmentation performance. Experimental results demonstrate that the proposed method outperforms state-of-the-art methods with overall accuracy of 91.26%, 93.33%, and 82.38%, and the average accuracy of 94.22%, 89.61%, and 80.35% over three benchmark datasets, including Salinas Valley, Pavia University and Indian Pines dataset, respectively. It illustrates that the attention-guided multi-scale feature learning framework is able to effectively explore the joint spectral-spatial information.

The main contribution of DA-IMRN is mainly in three folds. 
1) We propose a dual-attention-guided interactive feature learning strategy, including spatial and channel attention module (SCAM), as well as spectral attention module (SAM). It interactively extracts joint spectral-spatial information and perform feature fusion to realize end-to-end HSI pixel-level classification. By adjusting the weights of feature maps from three different dimensions, the bidirectional attention guides feature extraction in an effective direction.

2) We introduce a multi-scale spectral/spatial residual block (MSRB) for semantic segmentation. It utilizes different kernel sizes in the convolution layer to extract the features corresponding to multiple receptive fields, and provides abundant information for pixel-level classification.

3) We evaluate the significance of the proposed modules and their performance over three popular benchmark datasets. The extensive experimental results demonstrate the proposed DA-IMRN outperforms state-of-the-arts HSI semantic segmentation methods published in recent two years. The related codes are freely available for users at the following website: https://github.com/usefulbbs/DA-IMRN. 


Please find the following scripts used in this study.

dataset_generation: Code for dataset Partition, from HSI to training/validation/test/leaked image>>>training/validation/test patches.

main_ori: Main function.

model_IP: Model used for Indian Pines dataset.

model_PU: Model used for Pavia University dataset.

model_SV: Model used for Salinas Valley dataset.

Data: Part of the source data.

predata folder： Randomly generated data set used in this experiment. You can generate new predata using the code of dataset_generation.

Any probelm, please feel free to contact me at liangzou@cumt.edu.cn

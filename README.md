# DA-IMRN
Abstract：The fusion of spectral-spatial information has emerged as the main stream of hyperspectral image classification, especially with the advance of deep learning. However, due to the insufficient samples, the recent feature fusion methods might loss the joint interactions, impeding the further improvement of classification accuracy. In this paper, we propose a dual-attention-guided interactive multi-scale residual network (DA-IMRN) to explore the joint spectral-spatial information and assign the label for each pixel of hyperspectral image without information leakage. First, two branches focusing on the spatial and spectral information separately are employed for feature extraction. A bidirectional-attention mechanism, including spatial-channel attention and spectral attention is utilized to guide the interactive feature learning between two branches, and promote refined feature maps. In addition, we extract deep multi-scale features corresponding to multiple receptive fields from limited samples via a multi-scale spectral/spatial residual block, and improve the segmentation performance. Experimental results demonstrate that the proposed method outperforms state-of-the-art methods with overall accuracy of 91.26%, 93.33%, and 82.38%, and the average accuracy of 94.22%, 89.61%, and 80.35% over three benchmark datasets, including Salinas Valley, Pavia University and Indian Pines dataset, respectively. It illustrates that the attention-guided multi-scale feature learning framework is able to effectively explore the joint spectral-spatial information.



dataset_generation: Code to divide the data set.

main_ori: Main function.

model_IP: Model used for Indian Pines dataset.

model_PU: Model used for Pavia University dataset.

model_SV: Model used for Salinas Valley dataset.

data: Part of the source data.

predata folder： Randomly generated data set used in this experiment. You can generate new predata through the code of dataset_generation.

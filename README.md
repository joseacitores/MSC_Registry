# Enabling the creation of X-ray patient registries through Deep Neural Networks


## Abstract

The objective is to implement a neural network based on previous models like the ResNet/ EfficientNet/ Vision Transformers in order to classify x-rays in three different sets of categories; nine different body part categories, the presence or absence of an instrument, and the presence or absence of a fracture. The datasets consists of 45000 x-rays from the nine different classes, labeled by physicians, for the body part classification matter as well as for the instrument detection. A hip X-ray dataset was used for fracture detection, consisting of 468 patients and also labeled by physicians. The methods used to solve this task consist of the implementation of three different neural networks, the use of transfer learning from the ImageNet dataset, and the fine-tuning of the networks to fit them to our problems. The data underwent normalization and augmentation for both datasets, as well as concatenation for the fracture dataset. After collecting the results, the final model is chosen based on the metrics collected after the testing stage. The Results show great performance for both problems that were trained on the larger dataset. The body part classification problem was solved by the three models almost. Equally, the best model being Efficient Net with a logarithmic loss of 0.1053 and a Cohen Kappa score of 0.9940. For instrument detection, the best model was ResNet101, achieving an AUC of 0.99 with a 95 CI of 0.95-1. Finally, in the proposed proof of concept for fracture detection, the results did not surpass the ones of a professional radiologist, achieving a sensitivity of 0.86, a specificity of 1, and an AUC of 0.93. In conclusion, the results show that the creation of an automated pipeline for the creation of x-rays patient registries is possible and achievable with a low error rate; the main limitation is the lack of labeled data to aid the creation of the given pipeline. Therefore the main challenge is the collaboration of medical staff for the creation of an initial database that can help to complete the work that is often overlooked and avoided. Enabling the expansion of scope for many possible applications of AI in the medical field.


Model comp and its relatives are files where each cell ca be run in order to follow a pipeline of training a model and storing the model and results in different files. path_dir variable should contain the main folder of the project with the following data structure:

.\
├── images\
├── models\
├── plots\
├── probs\
├── test\
├── train\
└── validation\


mean_std_dataset presents a function to calculate the mean and the standard deviation of an image dataset in order to be able to normalize it. 

utils is composed by several functions that allow to extract different metrics and plots

# Code for *Rethinking multiple instance learning for whole slide image classification: a bag-level classifier is a good instance-level teacher*

*This branch holds the confidence-based embedder finetuning method for ICMIL, which is even more effective and elegant. Click [here](https://github.com/Dootmaan/ICMIL) for the original version of ICMIL.* 

## 1. Brief Introduction and Prerequisites

In this preview version of ICMIL code, sensitive information and code comments has been removed to keep our submission anonymous. Some modifications (e.g., path to dataset, path to pretrained weights) to the code by yourself is necessary for running ICMIL correctly.

This repo contains the following codes that may be helpful for understanding our idea:

- Patch tiling code which is directly borrowed from official DTFD-MIL repo. It can be found at ./utils/gen_patch_noLabel_stride_MultiProcessing_multiScales.py
- The pickle dataset file for verifying baseline models (The pickle dataset is provided in the [official DTFD-MIL repo](https://github.com/hrzhang1123/DTFD-MIL)). When using this pickle dataset, our implementation shows the same performance as the official implementations do. It can be found at ./dataset/PickleDataset.py. Substitute the dataset class in the step1_train_classifier_xxxx.py to use this pickle dataset.
- Embedding visulization code which can be found at ./utils/plot_tsne_for_visulization.py

It should also be noted that, [the official code of DTFD-MIL](https://github.com/hrzhang1123/DTFD-MIL) requires using **PyTorch 1.4.0** for training, otherwise an Exception will be raised during training. However, the other baselines do not have such prerequistes. In our experiments, we set a independent anaconda environment with PyTorch 1.4.0 for DTFD-MIL related experiments. Other experiments are conducted with Python 3.9.12 and PyTorch 1.12.1. You need to have a GPU with >12GB video memory for the training, or otherwise you can reduce the default training batch size in step2.

Some required libraries:

- scikit-learn
- numpy
- opencv

  > In this branch, both aggregators and classifiers in DTFD-MIL are used to generate a more accurate attention score in Embedder Phase. BETA is set to 2 for this method. You can find the new weights in ./utils for verification or further finetuning. If you need the weights for other MIL backbones or you want the ICMIL-finetuned ResNet50, please contact me through email.
  >

## 2. Prepare your dataset

Please use the unpreprocessed [official release of Camelyon16](https://camelyon17.grand-challenge.org/Data/) for the most accurate verification of ICMIL. The release  can be found here:

    **GigaScience Database:** http://gigadb.org/dataset/100439

    **AWS Registry of Open Data**: https://registry.opendata.aws/camelyon/

    **Baidu Pan:** https://pan.baidu.com/s/1UW_HLXXjjw5hUvBIUYPgbA

If you would like to try ICMIL on your own WSI dataset, please make sure they follow the same file structure of Camelyon16, otherwise you may also need to write your own dataset.py for compatibility.

## 3. Run this code

The current version of code only use one single GPU for training. If you would like to use multiple GPUs for larger batch size, please remember to convert each model with torch.nn.DataParallel().

* Step1: generate the instance embeddings using the initial ResNet50 weights with ``CUDA_VISIBLE_DEVICES=[your_available_device_ids] python3 -u step0_patch2feature.py >step0_patch2feature_[method_name].log 2>&1 &``
* Step2: train a bag-level classifier with ``CUDA_VISIBLE_DEVICES=[your_available_device_ids] python3 -u step1_train_classifier_[method_name].py >step1_train_classifier_[method_name].log 2>&1 &``
* Step3: use ICMIL to finetune the embeder used in Step1 with the help of the classifier in Step2 by running  ``CUDA_VISIBLE_DEVICES=[your_available_device_ids] python3 -u step2_teacher_student_[method_name]_distillation.py >step2_teacher_student_[method_name]_distillation.log 2>&1 &``
* After having the finetuned embedder, return to **Step1** and generate the instance embeddings with the **new ResNet50 weights**.

The saved log files or weights during ICMIL training can be found directly in the folder. You can also change the saving directory by yourself at the end of each code file. After you run the three steps, you will have a new ResNet50 tailored for the WSI dataset, which can be used again in Step1 for better embeddings.

## 4. Some pretrained weights

The max supplementary material file size limit is 100Mb as is shown in CMT. Therefore, we carefully selected the following weights for your evaluation. Also, the full pretrained weights (including the full MIL weights and the corresponding ResNet50 weights after ICMIL) will be open along with the code upon paper acceptance.

- MaxPool_after_ICMIL_model_best.pth (16.8Mb): This is the Max Pool weights after ICMIL, which achieves a 85.2% AUC on Camelyon16 after one ICMIL iteration.
- AB-MIL_model_best_resnet1024_(also_for_visualization).pth (20.8Mb): This is the ABMIL weights after one ICMIL iteration. It achieves 90.00% AUC on the instance embeddings generated by ICMIL (before ICMIL the AUC is 85.4%).
- DTFD_after_ICMIL_model_best.pth (24.8Mb): This is the DTFD-MIL weights after ICMIL, which achieves a 93.7% AUC/87.0% F1/90.6% Acc on Camelyon16 after one ICMIL iteration.

These pretrained weights can be found in ./utils/ with the corresponding filenames. It should be noted that these MIL weights should also be used with the corresponding instance embeddings by ICMIL.

MMRSSeg
Multimodal Remote Sensing Image Semantic Segmentation

This repository, MMRSSeg, provides implementations and related resources for our research on multimodal remote sensing image semantic segmentation.

The project focuses on effectively exploiting complementary information from different sensing modalities, including RGB, NIR, DSM, and SAR, to improve semantic understanding of high-resolution remote sensing images.

Currently, this repository includes the following works:

🛰️ CSFAFormer: Category-Selective Feature Aggregation Transformer for Multimodal Remote Sensing Image Semantic Segmentation

🛰️ C²AHSegFormer: Cross-Modal Class-Distribution Alignment Hierarchical Transformer for Semantic Segmentation of Remote Sensing Images

📖 Introduction

Multimodal remote sensing data provide complementary spectral, structural, and elevation information, offering significant potential for improving semantic segmentation performance. 

However, effectively exploiting heterogeneous information across different modalities remains challenging due to modality discrepancies, category ambiguity, and inconsistent semantic responses.

To address these challenges, we propose two multimodal semantic segmentation frameworks:

🛰️ CSFAFormer

CSFAFormer (Category-Selective Feature Aggregation Transformer) is a multimodal semantic segmentation framework designed to effectively integrate complementary information from different remote sensing modalities.

🛰️ C²AHSegFormer

C²AHSegFormer (Cross-Modal Class-Distribution Alignment Hierarchical Transformer) is designed to further address the semantic inconsistency problem among different modalities.

🛠️ Framework

Our training and testing pipeline is built upon the excellent UNetFormer / GeoSeg framework developed by LiBo Wang.

We sincerely thank the author for sharing the implementation and providing a solid foundation for remote sensing semantic segmentation research.https://github.com/WangLibo1995/GeoSeg

🚀 Getting Started

The execution commands and environment configuration can be found in:

Terminal_config.txt

Please refer to this file for detailed instructions on training and testing the models.

The relevant source code is currently being organized and will be released progressively.

📂 Dataset Preparation

Since the tool code used to generate patch images from the original images is relatively complex, 

we have uploaded the pre-processed patch data to Baidu Cloud for your convenience.

（File_name：Cut_Muti_Modal_datasets.rarLink: https://pan.baidu.com/s/17CD2siDwyO7CyxpgC-m2tw?pwd=bvkq password: bvkq）

We tried uploading it to Google Drive, but due to payment issues, we were unsuccessful. For now, we can only provide the download link from Baidu Cloud.

📄 Citation

If you find this project useful for your research, please consider citing our papers:

Ni Y, Xue D, Chi W, et al. CSFAFormer: Category-Selective Feature Aggregation Transformer for Multimodal Remote Sensing Image Semantic Segmentation[J]. Information Fusion, 2025: 103786.
https://doi.org/10.1016/j.inffus.2025.103786

Ni Y, Liu J, Chi W, et al. C2AHSegFormer: cross-modal class-distribution alignment hierarchical transformer for semantic segmentation of remote sensing images[J]. Expert Systems with Applications, 2026: 134139.
10.1016/j.eswa.2026.134139

🔗 Related Project

We have also released several related works and implementations focusing on single-modal remote sensing image semantic segmentation.

https://github.com/EvilGhostY/SMRSSeg

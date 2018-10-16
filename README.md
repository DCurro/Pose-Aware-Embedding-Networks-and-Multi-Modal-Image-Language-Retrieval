# Pose-Aware Embedding Networks and Multi-Modal Image-Language Retrieval

The provided code is intended to support the exploration of embedding spaces, and multi-modal embedding spaces, especially for language.

## No Training Required

The dataset and pretrained models are available for [download]().

## Training from Scratch

To train these models from scratch:
1. Download the [dataset]().
1. Set the **path_dataset** variable in the path_manage.py.example file
1. remove the ".example" suffix from the path_manage.py.example file so that it is named path_manage.py
1.
1.
1.
1.
1.

# Masters Thesis Summary

A thesis presented to Ryerson University in partial fulfillment of the requirements for the degree of Master of Science in the program of Computer Science.

Domenico Curro
MSc. Computer Science.

## Abstract

Inspired by recent work in human pose metric learning this thesis explores a family of pose-aware embedding networks designed for the purpose of image similarity retrieval. Circumventing the need for direct human joint localization, a series of CNN embedding networks are trained to respect a variety of Euclidean and language-primitive metric spaces. Querying with imagery alone presents certain limitations and thus this thesis proposes a multi-modal image-language embedding space, extending the current model to allow for language-primitive queries. This additional language mode provides the benefit of improving retrieval quality by 3% to 14% under the hit@k metric. Finally, two approaches are constructed to address the issues of conducting partial language-primitive queries, with the former generating maximally likely descriptors and the latter exploiting the network’s tendency to factorize the embedding space into (mostly) linearly separable sub-spaces. These two approaches improve upon recall by 13% and 17% over the provided baselines.


## Similarity Metrics

This work explores the retreival of images across three spatial and one language-primitive similarity metric.

### Spatial Metrics

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/poses.png" width="200">
</p>

2D similarity is calculated by first aligning the two poses, and then taking their mean per-joint distance in image space.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/2d_similarity.png" width="200">
</p>

3D similarity is calculated by first aligning the two poses, about their volume centers, and then taking their mean per-joint distance in metric space.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/3d_similarity.png" width="200">
</p>

Procrustes similarity is calculated by first aligning, and rotating the two poses such that they maximally align, and then taking their mean per-joint distance in metric space.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/procrustes_similarity.png" width="200">
</p>

### Language Metric

Language is framed as a set of primitive statements [&#91;2&#93;](http://www.cs.toronto.edu/~fleet/research/Papers/posebits_cvpr2014.pdf), such as "left elbow is (not) bent." There are three kinds of language-primitives: joint angle, joint-pair distance, and joint-pair relative distance. Joint angle and joint-pair distance are determine by a predefined threshold, where the joint is "bent" if it is before the threshold, and two joints are "far" if they are beyond the threshold. Joint-pair relative distances tell if you a joint is "beyond" a relative joint with respect to the torso-center.

Any one pose can then be described as a vector of binary values (a posebyte, composed of posebits).

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/Posebits.png" width="350">
</p>

## Querying with Images

The network is used is a modified version of the [&#91;3&#93;](http://www.robots.ox.ac.uk/~vgg/research/deep_eval/), as outlined in Kwak et al. [&#91;1&#93;](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kwak_Thin-Slicing_for_Pose_CVPR_2016_paper.pdf).

The modified network is initialized with the convolutional weights of the original network. The final fully connected layers are removed and replaced. Finally, the network is concluded with an l2-normalization layer, such that the final embedding vector lays on a unit-hypershere.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/Thin-Slicing.png" width="300">
</p>

## Querying with Language

Querying with images alone is limited. You might not have the images you want to begin a search. Alternatively, language descriptors should be easier to generate. A network archetecture and two approaches are defined to accomodate for language-based queries.

To query with language, a new mode is trained to map from posebyte language descriptor to the the trained embedding space. More specifically, first the image network is trained (1+3). When complete, the language stream is trained (2+3). While training this new mode, the shared embedding space (3) remain unaltered. This process learns a new mapping onto the existing embedding space.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/Thin-Slicing_multimodal.png" width="500">
</p>

### Conditional Queries

The first approach assumes that the relationship between each posebit is governed by a Normal Distribution. Conditionin the Normal Distribution on a desired set of conditions, maximum-likelihood estimation can be used to resolve the remaining bits. This entails simply taking the mean of the conditonal Normal Distribution. Consider the toy example of predicting the value of **bit a**, when **bit b** is one:

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/conditional_posebyte.png" width="400">
</p>

With a subset of desired conditions, like "right knee is bent; left knee is bent," (indicated in red) we can then resolve a new posebyte w.
<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/conditional_posebyte_resolution.png" width="300">
</p>

### Origin Queries by Warping

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/MaskedEmbeddings.png" width="800">
</p>

## Dataset and Pretrained Models

The dataset, available [here]() is composed of images, with corresponding 2D, 3D, and language-primitive descriptors. For example:

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/dataset_annotations.png" width="600">
</p>

## Refrences

[1] S. Kwak, M. Cho, and I. Laptev. Thin-slicing for pose: Learning to understand pose without explicit pose estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4938–4947. IEEE, 2016.

[2] G. Pons-Moll, D. J. Fleet, and B. Rosenhahn. Posebits for monocular human pose estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2337–2344, 2014.

[3] K. Chatfield, K. Simonyan, A. Vedaldi, and A. Zisserman. Return of the devil in the details: Delving deep into convolutional nets. In Proceedings of the British Machine Vision Conference, 2014.

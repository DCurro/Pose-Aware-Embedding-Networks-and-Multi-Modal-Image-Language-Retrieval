# Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval

Some background text about how this code supports my thesis (some link here).

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

Language is framed as [a set of primitive statements](http://www.cs.toronto.edu/~fleet/research/Papers/posebits_cvpr2014.pdf), such as "left elbow is (not) bent." There are three kinds of language-primitives: joint angle, joint-pair distance, and joint-pair relative distance. Joint angle and joint-pair distance are determine by a predefined threshold, where the joint is "bent" if it is before the threshold, and two joints are "far" if they are beyond the threshold. Joint-pair relative distances tell if you a joint is "beyond" a relative joint with respect to the torso-center.

Any one pose can then be described as a vector of binary values (a posebyte, composed of posebits).

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/Posebits.png" width="350">
</p>

## Querying with Images

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/Thin-Slicing.png" width="400">
</p>

## Querying with Language

## Conditional Queries

## Origin Queries by Warping

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/MaskedEmbeddings.png" width="800">
</p>

## Dataset and Pretrained Models

Link to the dataset (). Statement about the dataset.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/dataset_annotations.png" width="600">
</p>


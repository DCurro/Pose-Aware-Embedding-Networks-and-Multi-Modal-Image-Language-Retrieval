# Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval

Some background text about how this code supports my thesis (some link here).

## Abstract

Inspired by recent work in human pose metric learning this thesis explores a family of pose-aware embedding networks designed for the purpose of image similarity retrieval. Circumventing the need for direct human joint localization, a series of CNN embedding networks are trained to respect a variety of Euclidean and language-primitive metric spaces. Querying with imagery alone presents certain limitations and thus this thesis proposes a multi-modal image-language embedding space, extending the current model to allow for language-primitive queries. This additional language mode provides the benefit of improving retrieval quality by 3% to 14% under the hit@k metric. Finally, two approaches are constructed to address the issues of conducting partial language-primitive queries, with the former generating maximally likely descriptors and the latter exploiting the network’s tendency to factorize the embedding space into (mostly) linearly separable sub-spaces. These two approaches improve upon recall by 13% and 17% over the provided baselines.


## Similarity Metrics

This work explore three spatial and one language similarity metrics.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/poses.png" width="300">
</p>

### Spatial Metrics

2D similarity is calculated by first aligning the two poses, and then taking their mean per-joint distance in image space.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/2d_similarity.png" width="300">
</p>

3D similarity is calculated by first aligning the two poses, about their volume centers, and then taking their mean per-joint distance in metric space.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/3d_similarity.png" width="300">
</p>

Procrustes similarity is calculated by first aligning, and rotating the two poses such that they maximally align, and then taking their mean per-joint distance in metric space.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/procrustes_similarity.png" width="300">
</p>

### Language Metric


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


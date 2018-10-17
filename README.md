# Pose-Aware Embedding Networks and Multi-Modal Image-Language Retrieval

The provided code is intended to support the exploration of embedding spaces, and multi-modal embedding spaces, especially for language.

The dataset and pretrained models are available for [download]().

This code is written in python 3.7.0, and uses PyTorch 0.4.1.

## Setting up

1. Install the python dependencies `pip install -r requirements.txt`.
1. Download the [dataset]().
1. Set the `path_dataset` variable in the `path_manager.py.example` file.
1. Raname `path_manager.py.example` to `path_manager.py`.

## Just Evaluation; No Training Required

Evaluating the pose-aware embeddings:
1. Copy the desired `precomputed_embeddings/image/*mode*/embedding_0.py` to the corresponding `experiments/image/*mode*/embeddings` folder.
1. Run the desired evaulation located in the respective eval folder.

Evaluating the language mode:
1. Copy the `precomputed_embeddings/language/embedding_valtest_0.py` to the corresponding `experiments/language` embeddings folder.
1. Run the desired evaulation located in the respective eval folder.

Evaluating the conditional posebytes:
1. Copy the `precomputed_embeddings/language_conditional/embedding_conditional.py` to the corresponding `experiments/language_conditional/` embeddings folder.
1. Run the desired evaulation located in the respective eval folder.

Evaluating the pose-aware masks:
1. Copy the contents of `precomputed_masks/` to the corresponding `experiments/language_mask/masks/` folder.
1. Run the desired evaulation located in the respective eval folder.

## Training from Scratch

To train and evalute these pose-aware embeddings:
1. Run the desired model `experiments/*mode*/train.py` script.
1. Once complete, run `experiments/*mode*/embed.py`.
1. Once complete, run the desired evaluation script in `experiments/*mode*/eval/`.

To train and evaluate the language mode:
1. Run the `experiments/language/train.py`.
1. Once complete, run `experiments/language/embed.py`.
1. Once complete, run the desired evaluation script in `experiments/language/eval/`.

To produce and evaluate the conditional posebytes:
1. Run the `experiments/language_conditional/generate_single_question_posebyte.py`.
1. Once complete, run `experiments/language_conditional/embed.py`.
1. Once complete, run the desired evaluation script in `experiments/language_conditional/eval/`.

To train and evaluate the conditional masks:
1. Run the `experiments/language_masks/train.py`.
1. Once complete, run the desired evaluation script in `experiments/language_masks/eval/`.

# Masters Thesis Summary

A thesis presented to Ryerson University in partial fulfillment of the requirements for the degree of Master of Science in the program of Computer Science.

Domenico Curro
MSc. Computer Science.

## Abstract

Inspired by recent work in human pose metric learning this thesis explores a family of pose-aware embedding networks designed for the purpose of image similarity retrieval. Circumventing the need for direct human joint localization, a series of CNN embedding networks are trained to respect a variety of Euclidean and language-primitive metric spaces. Querying with imagery alone presents certain limitations and thus this thesis proposes a multi-modal image-language embedding space, extending the current model to allow for language-primitive queries. This additional language mode provides the benefit of improving retrieval quality by 3% to 14% under the hit@k metric. Finally, two approaches are constructed to address the issues of conducting partial language-primitive queries, with the former generating maximally likely descriptors and the latter exploiting the network’s tendency to factorize the embedding space into (mostly) linearly separable sub-spaces. These two approaches improve upon recall by 13% and 17% over the provided baselines.


## Similarity Metrics

This work explores the retreival of images across three spatial and one language-primitive similarity metric.

### Spatial Metrics

For any two poses, the similarity is considered.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/poses.png" width="300">
</p>

2D similarity is calculated by first aligning the two poses, and then taking their mean per-joint distance in image space.

3D similarity is calculated by first aligning the two poses, about their volume centers, and then taking their mean per-joint distance in metric space.

Procrustes similarity is calculated by first aligning, and rotating the two poses such that they maximally align, and then taking their mean per-joint distance in metric space.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/2d_similarity.png" width="200"> <img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/3d_similarity.png" width="200"> <img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/procrustes_similarity.png" width="200">
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

The first approach assumes that the relationship between each posebit is governed by a Normal Distribution. Conditionin the Normal Distribution on a desired set of conditions, maximum-likelihood estimation can be used to resolve the remaining bits. This entails simply taking the mean of the conditonal Normal Distribution. Consider the toy example of predicting the value of `bit a`, when `bit b` is one:

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/conditional_posebyte.png" width="400">
</p>

With a subset of desired conditions, like "right knee is bent; left knee is bent," (indicated in red) we can then resolve a new posebyte w.
<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/conditional_posebyte_resolution.png" width="300">
</p>

### Origin Queries by Warping

Instead of generating queries, this approach proposed to manipulate the embedding space directly. When searching for a subset of language conditions, said conditions will exist in multiple places on the embedding space in a variety of modes. By collpasing the subspace of interest which contains the desired subset of language conditions, these various modes can be brought to a common point (the origin), simplifying the search process.

Computing the embeddings offline, this new model simply learns a `mask` which performs a point-wise multiplication warping of the original embedding space.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/MaskedEmbeddings2.png" width="600">
</p>

With the warped embedding space, a simple nearest neighbours search starting from the origin yields the set of embeddings ordered by relevance:

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/zero_query.png" width="400">
</p>

One consequence of this approach is that a new mask must be learned per langauge condition subset.

## Dataset

The dataset, available [here]() is composed of images, with corresponding 2D, 3D, and language-primitive descriptors. For example:

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/dataset_annotations.png" width="600">
</p>

## Evaluation

### Language Queries

Querying with Artifical Posebytes (top rows) and Masks (bottoms):

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/query_examples_pb_and_mask.png" width="600">
</p>

Observing that Masks order images by visual saliency (how obvious the language condition is). For example, in row 5 we can see that as you move further away from the origin in large uniform steps, the left wrist starts as far above the neck as possible and then slowly decends until it is below.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/query_escape_zero.png" width="600">
</p>

Observing the learned Masks, we can see the negative condition ("is not") and the positive condition ("is") reside on what appear to be mutually exclusive subspaces.

<p align="center">
<img src="https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/blob/master/github_images/masks.png" width="600">
</p>


## Refrences

[1] S. Kwak, M. Cho, and I. Laptev. Thin-slicing for pose: Learning to understand pose without explicit pose estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4938–4947. IEEE, 2016.

[2] G. Pons-Moll, D. J. Fleet, and B. Rosenhahn. Posebits for monocular human pose estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2337–2344, 2014.

[3] K. Chatfield, K. Simonyan, A. Vedaldi, and A. Zisserman. Return of the devil in the details: Delving deep into convolutional nets. In Proceedings of the British Machine Vision Conference, 2014.

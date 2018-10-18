
# Pose-Aware Embedding Networks and Multi-Modal Image-Language Retrieval
* [Pose-Aware Embedding Networks and Multi-Modal Image-Language Retrieval](#Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval)
  * [Setting up](#setting-up)
  * [Training from Scratch](#training-from-scratch)
  * [Evaluation Only](#Evaluation-Only)


The provided code is intended to support the exploration of embedding spaces, and multi-modal embedding spaces, especially for language. A summary may be found [here](https://github.com/DCurro/Pose-Aware-Embedding-Networks-and-Multi-Modal-Image-Language-Retrieval/tree/master/thesis_summary), and the full document may be found [here](http://dcurro.com/Masters_Thesis.pdf).

The dataset and pretrained models are available for [download]().

This code is written in python 3.7.0, and uses PyTorch 0.4.1.

## Setting up

1. Install the python dependencies `pip install -r requirements.txt`.
1. Download the [dataset](), and unzip.
1. Unzip images_single.zip.
1. Set the `path_dataset` variable in the `path_manager.py.example` file.
1. Raname `path_manager.py.example` to `path_manager.py`.

## Training from Scratch

To train and evalute these pose-aware embeddings:
1. Run the desired model `experiments/*/train.py` script.
1. Once complete, run `experiments/*/embed.py`.
1. Once complete, run the desired evaluation script in `experiments/*/eval/`.

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

## Evaluation Only

Evaluating the pose-aware embeddings:
1. Copy the desired `precomputed_embeddings/image/*/embedding_0.py` to the corresponding `experiments/image/*/embeddings` folder.
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


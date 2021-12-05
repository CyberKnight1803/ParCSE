from typing import List

import torch

from transformers import AutoTokenizer


from paraphrasegen.constants import PATH_BASE_MODELS
from paraphrasegen.model import Encoder
from paraphrasegen.loss import Similarity


device = (
    "cpu"  # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def tokenize_text(model_name, sentences: List[str]):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, cache_dir=PATH_BASE_MODELS
    )

    tokenized = tokenizer(
        sentences,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokenized


def eval(encoder):
    anchor = "A Washington County man may have the countys first human case of West Nile virus , the health department said Friday ."
    target = "A Hyderabadi man may have the citys first human case of West Nile virus , the health ministry said Friday ."
    # target = "The countys first and only human case of West Nile this year was confirmed by health officials on Sept . 8 ."
    negative = "What the fuck is the County Virus"

    print("Tokenizing Text... ", sep="")
    tokenized = tokenize_text(
        encoder.hparams.model_name_or_path, [anchor, target, negative]
    )

    print("Tokenized!")

    print("Generating Embeddings... ", sep="")
    embeddings = encoder(
        tokenized["input_ids"],
        tokenized["attention_mask"],
        do_mlm=False,
    )

    anchor_embedddings = embeddings[0, ...]
    target_embedddings = embeddings[1, ...]
    negative_embeddings = embeddings[2, ...]

    print("Generated!")

    # print(f"|Anchor|: {torch.norm(anchor_embedddings)}")
    diff = target_embedddings - anchor_embedddings
    print(
        f"|target_embedddings - anchor_embedddings|: {torch.norm(diff)}, %age: {100 * torch.mean(diff / anchor_embedddings)}"
    )

    diff = negative_embeddings - anchor_embedddings
    print(
        f"|negative_embeddings - anchor_embedddings|: {torch.norm(diff)}, %age: {100 * torch.mean(diff / anchor_embedddings)}"
    )

    sim = Similarity(temp=1)
    print(
        f"Similarity between anchor and target: {sim(anchor_embedddings, target_embedddings)}"
    )

    print(
        f"Similarity between anchor and negative: {sim(anchor_embedddings, negative_embeddings)}"
    )


if __name__ == "__main__":
    path_to_checkpoint = "runs/default/version_7/checkpoints/last.ckpt"  # input(">>> Enter Model Checkpoint Path: ")
    print("Loading Model... ", sep="")
    encoder = Encoder.load_from_checkpoint(path_to_checkpoint)
    print("Finished")

    eval(encoder)

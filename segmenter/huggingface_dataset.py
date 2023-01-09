from typing import List, Dict
from datasets import Dataset, load_dataset, ClassLabel, interleave_datasets, concatenate_datasets


def features_to_np(sample) -> Dict[str, List[List[float]]]:
    feas: List[str] = sample["text"].split(";")
    return {"audio_features": [list(map(float, fea.split())) for fea in feas]}


def get_datasets(train_text_file: str, dev_text_file: str, temperature: int,
                 train_audio_features_file: str = None, dev_audio_features_file: str = None) -> Dataset:
    hf_dataset = load_dataset("text", data_files={"train": train_text_file, "dev": dev_text_file})
    hf_dataset = hf_dataset.map(lambda sample: {"words": " ".join(sample["text"].split()[1:])})
    hf_dataset = hf_dataset.map(lambda sample: {"label": int(sample["text"].split()[0])}, remove_columns="text")
    train_uniq_labels = hf_dataset["train"].unique("label")
    dev_uniq_labels = hf_dataset["dev"].unique("label")

    assert len(train_uniq_labels) == len(dev_uniq_labels)
    assert [x == y for x, y in zip(sorted(train_uniq_labels), range(len(train_uniq_labels)))]

    hf_dataset.cast_column("label", ClassLabel(num_classes=len(train_uniq_labels)))

    if train_audio_features_file is not None:
        if dev_audio_features_file is None:
            raise Exception
        else:
            audio_dataset = load_dataset("text", data_files={"train": train_audio_features_file,
                                                             "dev": dev_audio_features_file})

            audio_dataset = audio_dataset.map(features_to_np, remove_columns="text")

            hf_dataset["train"] = concatenate_datasets([hf_dataset["train"], audio_dataset["train"]], axis=1)
            hf_dataset["dev"] = concatenate_datasets([hf_dataset["dev"], audio_dataset["dev"]], axis=1)

    per_class_train_datasets = [hf_dataset["train"].filter(lambda sample: sample["label"] == i) for i in
                                range(len(train_uniq_labels))]

    # Compute class prob and apply temperature
    per_class_pseudo_probs = [(len(ds) / len(hf_dataset["train"])) ** (1 / temperature) for ds in
                              per_class_train_datasets]
    # Re-normalize
    per_class_probs = [prob / sum(per_class_pseudo_probs) for prob in per_class_pseudo_probs]

    # Draw samples (with replacement) until both datasets have been exhausted once
    new_train_dataset = interleave_datasets(datasets=per_class_train_datasets, probabilities=per_class_probs,
                                            stopping_strategy="all_exhausted"
                                            )

    hf_dataset["train"] = new_train_dataset

    return hf_dataset

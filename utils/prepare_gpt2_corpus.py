import datasets
from transformers import GPT2TokenizerFast
from itertools import chain

def main():
    # dataset=datasets.load_dataset("openwebtext")
    # dataset.save_to_disk("openwebtext")
    dataset=datasets.load_from_disk("openwebtext")
    print("loaded")
    dataset=dataset["train"].train_test_split(test_size=0.05, shuffle=False)
    print("split")
    dataset=datasets.DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]   
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=96,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size=1024
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_datasets=tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=96,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    lm_datasets.save_to_disk("openwebtext_prepared_1024")

if __name__=="__main__":
    main()
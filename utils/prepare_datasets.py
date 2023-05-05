import random
import datasets
from transformers import BertTokenizerFast

def main():
    raw_datasets=datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
    raw_datasets=datasets.DatasetDict({"train":raw_datasets["train"], "validation":datasets.concatenate_datasets([raw_datasets["validation"], raw_datasets["test"]])})
    def clean(examples):
        l=[]
        i=0
        while i<len(examples["text"]):
            sent=examples["text"][i]
            if sent=="":
                i+=3
            else:
                l.append(sent)
                i+=1
        return {"text":l}
    raw_datasets["train"]=raw_datasets["train"].map(clean, batched=True, num_proc=12)
    raw_datasets["validation"]=raw_datasets["validation"].map(clean, batched=True, num_proc=12)
    def break_sents(examples):
        l=[]
        for sent in examples["text"]:
            i=0
            words=sent.split(" ")
            while i<len(words):
                s=" ".join(words[i:i+64])
                if s!="":
                    l.append(s)
                i+=64
        return {"text":l}
    raw_datasets["train"]=raw_datasets["train"].map(break_sents, batched=True, num_proc=12)
    raw_datasets["validation"]=raw_datasets["validation"].map(break_sents, batched=True, num_proc=12)
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
    column_names=raw_datasets["train"].column_names
    text_column_name="text" if "text" in column_names else column_names[0]
    max_seq_length=128
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
    
    tokenized_datasets=raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=12,
        remove_columns=column_names,
        desc="Running tokenizer on every text in dataset",
    )

    def group_texts(examples, idx, split):
        # Account for [CLS], [SEP], [SEP]
        max_num_tokens=max_seq_length-3
        # We *usually* want to fill up the entire sequence since we are padding
        # to `max_seq_length` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `max_seq_length` is a hard limit.
        target_seq_length=max_num_tokens
        if random.random()<0.1:
            target_seq_length=random.randint(2, max_num_tokens)
        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        result={k: [] for k, v in tokenizer("", return_special_tokens_mask=True).items()}
        result['next_sentence_label']=[]
        current_chunk=[]
        current_length=0
        i=0 
        while i<len(idx):
            segment={k: examples[k][i][1:-1] for k in examples.keys()}
            current_chunk.append(segment)
            current_length += len(segment['input_ids'])
            if i==len(idx)-1 or current_length>=target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end=1
                    if len(current_chunk)>=2:
                        a_end=random.randint(1, len(current_chunk)-1)
                    tokens_a={k: [] for k, t in tokenizer("", return_special_tokens_mask=True).items()}
                    for j in range(a_end):
                        for k, v in current_chunk[j].items():
                            tokens_a[k].extend(v)

                    tokens_b={k: [] for k, t in tokenizer("", return_special_tokens_mask=True).items()}
                    # Random next
                    is_random_next=False
                    if len(current_chunk)==1 or random.random()<0.5:
                        is_random_next=True
                        target_b_length=target_seq_length-len(tokens_a["input_ids"])
                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_segment_index=random.randint(0, len(tokenized_datasets[split])-len(idx)-1)
                            if (random_segment_index-len(idx) not in idx) and (random_segment_index+len(idx) not in idx):
                                break

                        random_start=random.randint(0, len(idx)-1)
                        for j in range(random_start, len(idx)):
                            for k, v in {k: tokenized_datasets[split][random_segment_index+j][k][1:-1] for k in examples.keys()}.items():
                                tokens_b[k].extend(v)
                            if len(tokens_b['input_ids'])>=target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments=len(current_chunk)-a_end
                        i-=num_unused_segments
                    # Actual next
                    else:
                        is_random_next=False
                        for j in range(a_end, len(current_chunk)):
                            for k, v in current_chunk[j].items():
                                tokens_b[k].extend(v)

                    while True:
                        total_length=len(tokens_a['input_ids'])+len(tokens_b['input_ids'])
                        if total_length<=max_num_tokens:
                            break
                        trunc_tokens= tokens_a if len(tokens_a['input_ids'])>len(tokens_b['input_ids']) else tokens_b
                        # We want to sometimes truncate from the front and sometimes from the
                        # back to add more randomness and avoid biases.
                        if random.random()<0.5:
                            for k in trunc_tokens.keys():
                                del trunc_tokens[k][0]
                        else:
                            for k in trunc_tokens.keys():
                                trunc_tokens[k].pop()
                    inp={k: v[:-1] for k, v in tokenizer("", return_special_tokens_mask=True).items()}
                    for k, v in tokens_a.items():
                        inp[k].extend(v)
                    SEP={k: v[1:] for k, v in tokenizer("", return_special_tokens_mask=True).items()}
                    for k, v in SEP.items():
                        inp[k].extend(v)
                    tokens_b['token_type_ids']=list(map(lambda x: 1, tokens_b['token_type_ids']))
                    for k, v in SEP.items():
                        tokens_b[k].extend(v)
                    tokens_b['token_type_ids'][-1]=1
                    for k, v in tokens_b.items():
                        inp[k].extend(v)
                    inp['next_sentence_label']=int(is_random_next)
                    for k, v in inp.items():
                        result[k].append(v)
                current_chunk=[]
                current_length=0
            i+=1
        return result

    train_dataset=tokenized_datasets["train"]
    eval_dataset=tokenized_datasets["validation"]
    train_dataset=train_dataset.map(
        group_texts,
        fn_kwargs={"split":"train"},
        batched=True,
        batch_size=1000,
        num_proc=12,
        with_indices=True,
        desc=f"Grouping Train texts in chunks of {max_seq_length}",
    )
    eval_dataset=eval_dataset.map(
        group_texts,
        fn_kwargs={"split":"validation"},
        batched=True,
        batch_size=1000,
        num_proc=12,
        with_indices=True,
        desc=f"Grouping Validation texts in chunks of {max_seq_length}",
    )

    wikitext_103_prepared=datasets.DatasetDict({"train":train_dataset, "validation":eval_dataset})
    wikitext_103_prepared.save_to_disk("wikitext-103-prepared")

if __name__=="__main__":
    main()
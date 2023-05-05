import argparse
import datetime
import logging
import math
import datasets
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import(
    BertConfig,
    BertTokenizerFast,
    BertForPreTraining,
    DataCollatorForLanguageModeling,
)
from accelerate import InitProcessGroupKwargs
import os
from tqdm.auto import tqdm

logger=get_logger(__name__)

def parse_args():
    parser=argparse.ArgumentParser(description="Calculate loss on a model checkpoint")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="model path"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=512,
        help="per device eval batch size"
    )
    args=parser.parse_args()
    return args


def main():
    args=parse_args()
    init_process_group=InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=75000))
    accelerator=Accelerator(kwargs_handlers=[init_process_group])
    logging.basicConfig(
        filename="./logs.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    logger.info(f"Loading the model configuration")
    config=BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
    )
    logger.info(f"Loading the tokenizer.")
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
    logger.info(f"Initializing Model.")
    model=BertForPreTraining.from_pretrained(
        args.model_path,
        from_tf=bool(".ckpt" in args.model_path),
        config=config
    )
    model.resize_token_embeddings(len(tokenizer))

    dataset=load_from_disk("bert_dataset_prepared")
    eval_dataset=dataset["validation"]

    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    eval_dataloader=DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    model, eval_dataloader=accelerator.prepare(
        model, eval_dataloader
    )

    model.eval()
    losses=[]
    pbar=tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs=model(**batch)
        pbar.update(1)
        loss=outputs.loss
        losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
    losses=torch.cat(losses)
    losses=losses[:len(eval_dataset)]
    try:
        perplexity=math.exp(torch.mean(losses))
    except OverflowError:
        perplexity=float("inf")
    if accelerator.is_main_process:
        with open(os.path.join(args.model_path, "loss.txt"), "w") as f:
            f.write(f"{torch.mean(losses)}")

if __name__=="__main__":
    main()
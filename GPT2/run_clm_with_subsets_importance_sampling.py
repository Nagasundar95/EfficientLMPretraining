import argparse
import datetime
import time
import logging
import math
import os
import sys
import random
from itertools import chain
import datasets
import torch
from torch.optim import AdamW
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.utils import broadcast_object_list
from transformers import (
    GPT2Config, 
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler
)
from transformers.utils.versions import require_version
from cords.selectionstrategies.SL import SubmodStrategy
import pickle
from accelerate import InitProcessGroupKwargs
from helper_fns import taylor_softmax_v1
import faiss
import numpy as np
from scipy import sparse

logger=get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def parse_args():
    parser=parser=argparse.ArgumentParser(description="Train a language model on Causal Language Modeling Task")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="The directory to which training logs should be written"
    )
    parser.add_argument(
        "--subset_dir",
        type=str,
        required=True,
        help="The directory to which information regarding selected subsets should be stored"
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="If passed, already preprocessed data needs to be given and training will start right away"
    )
    parser.add_argument(
        "--load_data_from_disk",
        action="store_true",
        help="If passed, the dataset is loaded from the disk instead of downloading from the hub"
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        default=None,
        help="The path to the directory containing the dataset"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=float,
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_max_steps",
        type=int,
        default=1000000,
        help="Max training steps for learning rate. (Can tune the rate of decay of the learning rate with this parameter)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=250000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=10000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--subset_fraction", type=float, default=0.25, help="Fraction of the dataset that we want to use for training"
    )
    parser.add_argument(
        "--selection_strategy", type=str, default="fl", help="Subset selection strategy"
    )
    parser.add_argument(
        "--select_every", type=int, default=25000, help="Select a new subset for training every select_every training steps"
    )
    parser.add_argument(
        "--partition_strategy", type=str, default="random", help="Partition strategy for subset selection"
    )
    parser.add_argument(
        "--layer_for_similarity_computation", type=int, default=9, help="The hidden layer to use while calculating the similarities in submodular functions"
    )
    parser.add_argument(
        "--num_partitions", type=int, default=5000, help="Number of partitions in subset selection"
    )
    parser.add_argument(
        "--parallel_processes", type=int, default=96, help="Number of parallel processes for subset selection"
    )
    parser.add_argument(
        "--num_warmstart_epochs", type=int, default=0, help="Number of epochs to run in the warmstart stage"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature while calculating taylor softmax"
    )
    args = parser.parse_args()
    return args

    
def main():
    args=parse_args()
    init_process_group=InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=75000))
    # torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=75000))
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator=Accelerator(kwargs_handlers=[init_process_group])
    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        filename=args.log_dir+"/train_logs.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    # logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if not args.preprocessed:
        logger.info(f"Loading the data.")
        if args.load_data_from_disk is not None:
            if args.data_directory is not None:
                raw_datasets=load_from_disk(args.data_directory)
                if "validation" not in raw_datasets.keys():
                    raise Exception("ERROR: load_data_from_disk expects validation and train splits to be already present")
        elif args.dataset_name is not None:
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
        else:
            data_files = {}
            dataset_args = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
            extension = args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
            raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{args.validation_split_percentage}%]",
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{args.validation_split_percentage}%:]",
                    **dataset_args,
                )
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    logger.info(f"Loading the model configuration.")
    if args.config_name:
        config=GPT2Config.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config=GPT2Config.from_pretrained(args.model_name_or_path)
    else:
        config=GPT2Config()
    
    logger.info(f"Loading the tokenizer.")
    if args.tokenizer_name:
        tokenizer=GPT2TokenizerFast.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer=GPT2TokenizerFast.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    logger.info(f"Initializing Model.")
    if args.model_name_or_path:
        model=GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config
        )
    else:
        logger.info("Training new model from scratch")
        model=GPT2LMHeadModel(config)
    
    model.resize_token_embeddings(len(tokenizer))

    #Preprocessing the datasets
    #First we tokenize all the texts
    if not args.preprocessed:
        column_names=raw_datasets['train'].column_names
        text_column_name="text" if "text" in column_names else column_names[0]
    else:
        column_names=["text"]
        text_column_name="text"
    
    if not args.preprocessed:
        logger.info(f"Beginning Tokenization.")
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)
    
    if not args.preprocessed:
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
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
        
        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        
        train_dataset=lm_datasets["train"]
        eval_dataset=lm_datasets["validation"]
    
    if args.preprocessed:
        dataset=load_from_disk(args.data_directory)
        train_dataset=dataset["train"]
        eval_dataset=dataset["validation"]
    
    if accelerator.is_main_process:
        num_samples = int(round(len(train_dataset) * args.subset_fraction, 0))
        init_subset_indices = [random.sample(list(range(len(train_dataset))), num_samples)]
    else:
        init_subset_indices = [[]]
    accelerator.wait_for_everyone()
    broadcast_object_list(init_subset_indices)
    full_dataset=train_dataset
    subset_dataset=full_dataset.select(init_subset_indices[0])
    logger.info(f"Full data has {len(full_dataset)} datapoints, subset data has {len(subset_dataset)} datapoints.")

    if len(train_dataset)>3:
        # Log a few random samples from the training data
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    tokenizer.pad_token=tokenizer.eos_token
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # DataLoaders creation:
    warmstart_dataloader=DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    full_dataloader=DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    lr_scheduler=get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.lr_max_steps
    )

    logger.info(f"Prepare model, optimizer, train_dataloader, eval_dataloader ")
    # Prepare everything with our `accelerator`.
    model, optimizer, warmstart_dataloader, full_dataloader, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, warmstart_dataloader, full_dataloader, train_dataloader, eval_dataloader
    )

    if args.selection_strategy in ["fl", "logdet", "gc"]:
        subset_strategy=SubmodStrategy(logger, args.selection_strategy,
                                        num_partitions=args.num_partitions, partition_strategy=args.partition_strategy,
                                        optimizer="LazierThanLazyGreedy", similarity_criterion="feature",
                                        metric="cosine", eta=1, stopIfZeroGain=False,
                                        stopIfNegativeGain=False, verbose=False, lambdaVal=1)

     # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps=args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps=int(args.checkpointing_steps)
    else:
        checkpointing_steps=None

    # Train!
    total_batch_size=args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num warm-start epochs = {args.num_warmstart_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps=0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)

    logger.info(f"Begin the training.")
    timing = []
    for epoch in range(args.num_warmstart_epochs):
        if epoch==0:
            logger.info("Begin the warm-start")
        model.train()
        for step, batch in enumerate(warmstart_dataloader):
            start_time=time.time()
            outputs=model(**batch)
            loss=outputs.loss
            logger.info(f"Completed Steps: {1+completed_steps}; Loss: {loss.detach().float()}; lr: {lr_scheduler.get_last_lr()};")
            loss=loss/args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step%args.gradient_accumulation_steps==0 or step==len(warmstart_dataloader)-1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps+=1
            if isinstance(checkpointing_steps, int):
                if completed_steps%checkpointing_steps==0:
                    output_dir=f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir=os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps>=args.max_train_steps:
                break
            timing.append([(time.time() - start_time), 0])
        
        model.eval()
        losses=[]
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs=model(**batch)
            
            loss=outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
        losses=torch.cat(losses)
        losses=losses[:len(eval_dataset)]
        try:
            perplexity=math.exp(torch.mean(losses))
        except OverflowError:
            perplexity=float("inf")
        logger.info(f"Steps {completed_steps}: perplexity: {perplexity}")

    probs=[]
    greedyList=[]
    gains=[]
    partition_indices=[]
    if (args.num_warmstart_epochs>0) or (args.resume_from_checkpoint):
        start_time = time.time()
        if args.selection_strategy == 'Random-Online':
            if accelerator.is_main_process:
                init_subset_indices = [random.sample(list(range(len(full_dataset))), num_samples)]
            else:
                init_subset_indices = [[]]
        elif args.selection_strategy in ["fl", "logdet", "gc"]:
            pbar=tqdm(range(len(full_dataloader)), disable=not accelerator.is_local_main_process)
            model.eval()
            representations=[]
            total_cnt=0
            total_storage=0
            for step, batch in enumerate(full_dataloader):
                with torch.no_grad():
                    output=model(**batch, output_hidden_states=True)
                embeddings=output["hidden_states"][args.layer_for_similarity_computation]
                attention_mask=batch['attention_mask']
                embeddings=accelerator.gather(embeddings)
                attention_mask=accelerator.gather(attention_mask)
                total_cnt+=embeddings.size(0)
                if accelerator.is_main_process:
                    embeddings=embeddings.cpu()
                    attention_mask=attention_mask.cpu()
                    last_token_indices=torch.cat((attention_mask, torch.zeros((embeddings.shape[0],1))), dim=1).argmin(axis=1)-1
                    embeddings=torch.cat([embeddings[i][last_token_indices[i]].reshape((1,-1)) for i in range(embeddings.shape[0])], dim=0)
                    total_storage+=sys.getsizeof(embeddings.storage())
                    representations.append(embeddings)
                pbar.update(1)
            if accelerator.is_main_process:
                representations=torch.cat(representations, dim=0)
                representations = representations[:len(full_dataset)]
                total_storage += sys.getsizeof(representations.storage())
                representations = representations.numpy()
                logger.info('Representations Size: {}, Total number of samples: {}'.format(total_storage/(1024 * 1024), total_cnt))
                batch_indices=list(range(len(full_dataset)))
                logger.info('Length of indices: {}'.format(len(batch_indices)))
                logger.info('Representations gathered. Shape of representations: {}. Length of indices: {}'.format(representations.shape, len(batch_indices)))
            if accelerator.is_main_process:
                partition_indices, greedyIdx, gains = subset_strategy.select(len(batch_indices)-1, batch_indices, representations, parallel_processes=args.parallel_processes, return_gains=True)
                init_subset_indices=[[]]
                i=0
                for p in gains:
                    greedyList.append(greedyIdx[i:i+len(p)])
                    i+=len(p)
                probs=[taylor_softmax_v1(torch.from_numpy(np.array([partition_gains])/args.temperature)).numpy()[0] for partition_gains in gains]
                for i, partition_prob in enumerate(probs):
                    rng=np.random.default_rng(int(time.time()))
                    partition_budget=min(math.ceil((len(partition_prob)/len(batch_indices)) * num_samples), len(partition_prob)-1)
                    init_subset_indices[0].extend(rng.choice(greedyList[i], size=partition_budget, replace=False, p=partition_prob).tolist())
            else:
                init_subset_indices = [[]]
    accelerator.wait_for_everyone()
    broadcast_object_list(init_subset_indices)
    timing.append([0, (time.time() - start_time)])
    if accelerator.is_main_process:
        output_file=f"subset_indices_after_step_{completed_steps}.pt"
        output_file=os.path.join(args.subset_dir, output_file)
        torch.save(torch.tensor(init_subset_indices[0]), output_file)
        output_file=f"gains_after_step_{completed_steps}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(gains, f)
        output_file=f"partitions_after_step_{completed_steps}.pkl"
        output_file=os.path.join(args.subset_dir, output_file)
        with open(output_file, "wb") as f:
            pickle.dump(partition_indices, f)
    accelerator.wait_for_everyone()
    subset_dataset = full_dataset.select(init_subset_indices[0])
    subset_dataloader=DataLoader(
        subset_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    subset_dataloader = accelerator.prepare(subset_dataloader)

    logger.info("Begin training along with subset selection")
    while completed_steps<args.max_train_steps:
        model.train()
        for step, batch in enumerate(train_dataloader):
            train_time = 0
            subset_time = 0
            start_time = time.time()
            outputs=model(**batch)
            loss=outputs.loss
            logger.info(f"Completed Steps: {1+completed_steps}; Loss: {loss.detach().float()}; lr: {lr_scheduler.get_last_lr()};")
            loss=loss/args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step%args.gradient_accumulation_steps==0 or step==len(train_dataloader)-1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps+=1
            train_time += (time.time() - start_time)

            if isinstance(checkpointing_steps, int):
                if completed_steps%checkpointing_steps==0:
                    output_dir=f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir=os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            
            if completed_steps>=args.max_train_steps:
                break

            if completed_steps%args.select_every==0:
                accelerator.wait_for_everyone()
                start_time = time.time()
                num_samples = int(round(len(full_dataset) * args.subset_fraction, 0)) 
                if args.selection_strategy == 'Random-Online':
                    if accelerator.is_main_process:
                        init_subset_indices = [random.sample(list(range(len(full_dataset))), num_samples)]
                    else:
                        init_subset_indices = [[]]
                elif args.selection_strategy in ["fl", "logdet", "gc"]:
                    if accelerator.is_main_process:
                        init_subset_indices = [[]]
                        for i, partition_prob in enumerate(probs):
                            rng=np.random.default_rng(int(time.time()))
                            partition_budget=min(math.ceil((len(partition_prob)/len(batch_indices)) * num_samples), len(partition_prob)-1)
                            init_subset_indices[0].extend(rng.choice(greedyList[i], size=partition_budget, replace=False, p=partition_prob).tolist())
                    else:
                        init_subset_indices = [[]]
                accelerator.wait_for_everyone()
                broadcast_object_list(init_subset_indices)
                if accelerator.is_main_process:
                    output_file=f"subset_indices_after_step_{completed_steps}.pt"
                    output_file=os.path.join(args.subset_dir, output_file)
                    torch.save(torch.tensor(init_subset_indices[0]), output_file)
                    output_file=f"partitions_after_step_{completed_steps}.pkl"
                    output_file=os.path.join(args.subset_dir, output_file)
                    with open(output_file, "wb") as f:
                        pickle.dump(partition_indices, f)
                accelerator.wait_for_everyone()
                subset_dataset = full_dataset.select(init_subset_indices[0])
                subset_dataloader=DataLoader(
                    subset_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
                subset_dataloader = accelerator.prepare(subset_dataloader)
                subset_time = (time.time() - start_time)
                timing.append([train_time, subset_time])
                break
            timing.append([train_time, subset_time])

        model.eval()
        losses=[]
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs=model(**batch)
            
            loss=outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
        losses=torch.cat(losses)
        losses=losses[:len(eval_dataset)]
        try:
            perplexity=math.exp(torch.mean(losses))
        except OverflowError:
            perplexity=float("inf")
        logger.info(f"Steps {completed_steps}: perplexity: {perplexity}")
    
    logger.info(f"Saving the final model after {completed_steps} steps.")
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model=accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

if __name__=="__main__":
    main()
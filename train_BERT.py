import os
import subprocess
from datetime import datetime
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="train BERT")
    parser.add_argument(
        "--visible_gpus",
        type=str,
        help="visible_gpus"
    )
    parser.add_argument(
        "--main_process_port",
        type=int,
        default=55555,
        help="main process port for accelerate launch"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    now=datetime.now()
    timestamp=now.strftime("%d_%m_%Y_%H:%M:%S")
    log_dir="./logs/fl_bert_"+timestamp+"/"
    model_dir="./models/fl_bert_"+timestamp +"/"
    subset_dir="./subsets/fl_bert_"+timestamp+"/"
    # partitions_dir="./partitions/fl_bert_"+timestamp+"/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(subset_dir, exist_ok=True)
    # os.makedirs(partitions_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_gpus
    l=[
        "accelerate", "launch", "--main_process_port", f"{args.main_process_port}", "BERT/run_mlm_with_subsets_importance_sampling.py",
        "--preprocessed",
        "--log_dir", log_dir,
        "--subset_dir", subset_dir,
        # "--partitions_dir", partitions_dir,
        "--load_data_from_disk",
        "--data_directory", "./bert_dataset_prepared",
        "--tokenizer_name", "bert-base-uncased",
        "--vocab_size", "30522",
        "--preprocess_batch_size", "2000",
        "--per_device_train_batch_size", "128",
        "--per_device_eval_batch_size", "256",
        "--learning_rate", "1e-4",
        "--lr_max_steps", "1000000",
        "--weight_decay" ,"0.01",
        "--max_train_steps", "500000",
        "--gradient_accumulation_steps", "1",
        "--num_warmup_steps", "10000",
        "--output_dir", model_dir,
        "--max_seq_length", "128",
        "--preprocessing_num_workers", "96",
        "--mlm_probability" ,"0.15",
        "--short_seq_prob", "0.1",
        "--nsp_probability", "0.5",
        "--subset_fraction", "0.25",
        # "--select_every", "250000000",
        "--partition_strategy", "random",
        "--layer_for_similarity_computation", "9",
        "--num_partitions", "1500",
        "--selection_strategy", "fl",
        "--parallel_processes", "96",
        "--num_warmstart_epochs", "0",
        "--checkpointing_steps", "25000",
    ]
    with open(log_dir+"parameters.txt", "w") as f:
        for item in l:
            f.write(item)
            f.write("\n")
    subprocess.run(l)
if __name__=="__main__":
    main()
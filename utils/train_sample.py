import os
import subprocess
from datetime import datetime

def run_layer(layer_num):
    now=datetime.now()
    timestamp=now.strftime("%d_%m_%Y_%H:%M:%S")
    log_dir=f"./logs/testing_bert_logs_"+timestamp+"/"
    model_dir=f"./models/testing_bert_"+timestamp +"/"
    subset_dir=f"./subsets/testing_bert_"+timestamp+"/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(subset_dir, exist_ok=True)
    l=[
        "accelerate", "launch", "run_lm_with_subsets.py",
        "--preprocessed",
        "--log_dir", log_dir,
        "--subset_dir", subset_dir,
        "--load_data_from_disk",
        "--data_directory", "./wikitext-103-prepared",
        "--tokenizer_name", "bert-base-uncased",
        "--vocab_size", "30522",
        "--preprocess_batch_size", "2000",
        "--per_device_train_batch_size", "128",
        "--per_device_eval_batch_size", "128",
        "--learning_rate", "1e-4",
        "--weight_decay" ,"0.01",
        "--max_train_steps", "1000",
        "--gradient_accumulation_steps", "1",
        "--num_warmup_steps", "200",
        "--output_dir", model_dir,
        "--seed", "23",
        "--max_seq_length", "128",
        "--preprocessing_num_workers", "96",
        "--mlm_probability" ,"0.15",
        "--short_seq_prob", "0.1",
        "--nsp_probability", "0.5",
        "--subset_fraction", "0.25",
        "--select_every", "900",
        "--partition_strategy", "random",
        "--layer_for_similarity_computation", layer_num,
        "--num_partitions", "125",
        "--selection_strategy", "fl",
        "--parallel_processes", "96",
        "--checkpointing_steps", "900",
    ]
    subprocess.run(l)

def main():
    for i in range(12, 0, -1):
        run_layer(str(i))

if __name__=="__main__":
    main()
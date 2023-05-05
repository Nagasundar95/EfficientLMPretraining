import os
import subprocess
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="superglue")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to pytorch saved model"
    )
    parser.add_argument(
        "--main_process_port",
        type=str,
        default=55555,
        help="main process port for huggingface accelerate"
    )
    parser.add_argument(
        "--visible_gpus",
        type=str,
        required=True,
        help="gpu number to use"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    model_dir=args.model_dir
    log_dir=model_dir
    for i in range(1, 21):
        model_name_or_path=model_dir#+"step_{}/".format(100)
        if i>5:
            tasks=["wsc", "cb", "copa"]
        else:
            tasks=["boolq", "multirc", "wsc", "wic", "cb", "copa"] #can also add "mnli", "qnli", "qqp", "sst2" 
        # tasks=["cb", "copa"]
        task2batchsize={"boolq": 32, "multirc":32, "wic": 32, "wsc": 8, "cb": 8, "copa": 8}
        task2epochs={"boolq": 3, "multirc":3, "wic": 3, "wsc": 10, "cb": 10, "copa": 10}
        # task2batchsize={"boolq": 32, "multirc":32, "wic": 32, "wsc": 16, "cb": 16, "copa": 16}
        # task2epochs={"boolq": 3, "multirc":3, "wic": 3, "wsc": 5, "cb": 5, "copa": 5}
        glue_log_dir=model_name_or_path+f"superglue_run_{i}/"
        os.makedirs(glue_log_dir, exist_ok=True)
        for task in tasks:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_gpus
            l=[
                "accelerate", "launch", "--main_process_port", f"{args.main_process_port}", f"superglue/run_{task}_gpt2.py",
                "--log_file", glue_log_dir+task+".log",
                "--model_name_or_path", model_name_or_path,
                "--per_device_train_batch_size", f"{task2batchsize[task]}",
                "--per_device_eval_batch_size", f"{task2batchsize[task]}",
                "--learning_rate", f"1e-5",
                "--num_train_epochs", f"{task2epochs[task]}",
            ]
            subprocess.run(l)
    
if __name__=="__main__":
    main()
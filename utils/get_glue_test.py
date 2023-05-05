import os
import subprocess
import argparse

def main():
    all_models=os.listdir("test_models")
    for model_dir in all_models:
        for i in range(1, 3):
            model_name_or_path=f"./test_models/{model_dir}"#+"step_{}/".format(100)
            tasks=["cola", "mrpc"]#, "rte", "stsb", "sst2", "qnli", "mnli", "qqp"] #can also add "mnli", "qnli", "qqp", "sst2" 
            glue_log_dir=f"{model_name_or_path}/glue_run_{i}/"
            os.makedirs(glue_log_dir, exist_ok=True)
            for task in tasks:
                l=[
                    "accelerate", "launch", "run_glue.py",
                    "--log_file", glue_log_dir+task+".log",
                    "--task_name", task,
                    "--max_length", "128",
                    "--model_name_or_path", model_name_or_path,
                    "--per_device_train_batch_size", "4",
                    "--per_device_eval_batch_size", "8",
                    "--learning_rate", "5e-5",
                    "--weight_decay" ,"0.0",
                    "--num_train_epochs", "3",
                    "--output_dir", f"{glue_log_dir}{task}/"
                ]
                subprocess.run(l)
    
if __name__=="__main__":
    main()
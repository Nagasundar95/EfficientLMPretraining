import os
import subprocess

def main():
    model_paths=[
        "/home/sumbhati/ingenious/LM-pretraining/models/regular_bert_13_06_2022_18:22:00/",
    ]
    for model_path in model_paths:
        steps=sorted([int(x[5:]) for x in list(filter(lambda x: x.startswith("step_"), os.listdir(model_path)))])
        for step in steps:
            path=model_path+f"step_{step}/"
            extra_files=["config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]
            for extra_file in extra_files:
                subprocess.run(["cp", model_path+extra_file, path+extra_file])
            subprocess.run(["accelerate", "launch", "get_eval_loss.py", "--model_path", path])
            for extra_file in extra_files:
                subprocess.run(["rm", path+extra_file]) 
   
if __name__=="__main__":
    main()
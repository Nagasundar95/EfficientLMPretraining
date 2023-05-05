import subprocess

def main():
    main_model_dirs=[
        "/home/hrenduchinta/LM-pretraining/models/fl_bert_22_09_2022_20:35:55",
        "/home/hrenduchinta/LM-pretraining/models/fl_bert_02_09_2022_04:19:07",
    ]
    steps=[
        [100000, 200000, 300000, 400000, 500000],
        [50000, 100000, 150000, 200000, 250000]
    ]
    config_files=["config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]
    model_dirs=[]
    for i, main_model_dir in enumerate(main_model_dirs):
        for step in steps[i]:
            model_dirs.append(f"{main_model_dir}/step_{step}/")
            for f in config_files:
                subprocess.run(["cp", f"{main_model_dir}/{f}", f"{main_model_dir}/step_{step}/{f}"])
    for i, model_dir in enumerate(model_dirs):
        subprocess.Popen(
            f"nohup python3 get_glue_metrics.py --model_dir {model_dir} --main_process_port {52355+i} --visible_gpus {i%8} > ./gluelogs_{i}.txt",
            shell=True
        )
    
if __name__=="__main__":
    main()
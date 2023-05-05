import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os
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
        "--visible_gpus",
        type=str,
        required=True,
        help="gpu number to use"
    )
    parser.add_argument(
        "--do_save",
        action="store_true",
        help="save the finetuned models"
    )
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    tasks=[
        # "cola", "mnli", "mrpc", "qnli", "qqp", "sst", "stsb", "wnli", "rte",
        "boolq", "cb", "copa", "multirc", "wic", "wsc"#, "record",
    ]
    model_dir=args.model_dir
    for task in tasks:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_gpus
        num_gpus=1
        if task in ["boolq", "multirc", "wic"]:
            train_batch_size=32
            eval_batch_size=32
            epochs=3
            runs=5
        elif task in ["cb", "copa", "wsc"]:
            train_batch_size=16
            eval_batch_size=16
            epochs=5
            runs=20
        elif task=="record":
            train_batch_size=64
            eval_batch_size=64
            epochs=3
            runs=5
            num_gpus=8
        for i in range(1, runs+1):
            jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
                task_config_base_path="./tasks/configs",
                task_cache_base_path="./cache",
                train_task_name_list=[task],
                val_task_name_list=[task],
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                epochs=epochs,
                num_gpus=num_gpus,
            ).create_config()
            os.makedirs(f"{model_dir}/superglue_run_{i}/run_configs/", exist_ok=True)
            py_io.write_json(jiant_run_config, f"{model_dir}/superglue_run_{i}/run_configs/{task}_run_config.json")
            display.show_json(jiant_run_config)

            run_args = main_runscript.RunConfiguration(
                jiant_task_container_config_path=f"{model_dir}/superglue_run_{i}/run_configs/{task}_run_config.json",
                output_dir=f"{model_dir}/superglue_run_{i}/runs/{task}",
                hf_pretrained_model_name_or_path=model_dir,
                model_path=f"{model_dir}/pytorch_model.bin",
                model_config_path=f"{model_dir}config.json",
                learning_rate=5e-5,
                adam_epsilon=1e-6,
                eval_every_steps=500,
                do_train=True,
                do_val=True,
                do_save=args.do_save,
                force_overwrite=True,
            )
            main_runscript.run_loop(run_args)

if __name__=="__main__":
    main()
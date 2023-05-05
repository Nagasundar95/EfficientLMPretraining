import argparse
import datasets
from datasets import load_dataset

def parse_args():
    parser=argparse.ArgumentParser(description="Download wikitext corpus")
    parser.add_argument(
        "--validation_split_percentage",
        type=float,
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )

    args=parser.parse_args()

    return args

def main():
    args=parse_args()
    
    dataset=load_dataset("wikitext", "wikitext-103-raw-v1", split='train')
    dataset=dataset.train_test_split(test_size=(args.validation_split_percentage/100), shuffle=False)
    dataset=datasets.DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
    dataset.save_to_disk("./wikitext-103-raw-v1")

if __name__=="__main__":
    main()
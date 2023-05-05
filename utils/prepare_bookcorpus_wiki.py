import subprocess
import argparse
import datasets

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
    subprocess.run(["wget", "https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2"])
    subprocess.run(["tar", "-xf", "bookcorpus.tar.bz2"])
    bookcorpus1=datasets.Dataset.from_text("books_large_p1.txt")
    bookcorpus2=datasets.Dataset.from_text("books_large_p2.txt")
    bookcorpus=datasets.concatenate_datasets([bookcorpus1, bookcorpus2])
    wiki=datasets.load_dataset("wikipedia", "20220301.en", split="train")
    wiki=wiki.remove_columns(['id', 'url', 'title'])
    def break_sents(examples):
        l=[]
        for sent in examples["text"]:
            i=0
            words=sent.split(" ")
            while i<len(words):
                s=" ".join(words[i:i+64])
                if s!="":
                    l.append(s)
                i+=64
        return {"text":l}
    wiki=wiki.map(break_sents, batched=True, num_proc=96)
    bert_dataset=datasets.concatenate_datasets([wiki, bookcorpus])
    bert_dataset=bert_dataset.train_test_split(test_size=args.validation_split_percentage/100, shuffle=False)
    bert_dataset=datasets.DatasetDict({"train":bert_dataset["train"], "validation": bert_dataset["test"]})
    bert_dataset.save_to_disk("./bert_dataset/")

if __name__=="__main__":
    main()
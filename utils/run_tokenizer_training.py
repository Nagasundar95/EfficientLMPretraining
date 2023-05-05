import argparse
from datasets import load_dataset, load_from_disk

from transformers import BertTokenizerFast
from tokenizers import(
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

def parse_args():
    parser=argparse.ArgumentParser(description="Train a BertTokenizerFast using WordPiece algorithm, on a dataset")
    parser.add_argument(
        "--load_data_from_disk",
        action="store_true",
        help="If passed, the dataset is loaded from the disk."
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        default=None,
        help="The path to the directory in which dataset is present in the disk."
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
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size in tokenizer training"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=25000,
        help="The vocabulary size of the tokenizer"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final tokenizer.")

    args=parser.parse_args()
    return args


def main():
    args=parse_args()

    if args.load_data_from_disk:
        if args.data_directory is not None:
            dataset=load_from_disk(args.data_directory)
    else:
        dataset=load_dataset(args.dataset_name, args.dataset_config_name)
        
    dataset=dataset["train"]
    column_names=dataset.column_names
    text_column_name="text" if "text" in column_names else column_names[0]

    batch_size=args.batch_size
    def batch_iterator():
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i+batch_size][text_column_name]

    tokenizer=Tokenizer(models.WordPiece(unl_token="[UNK]"))
    tokenizer.normalizer=normalizers.Sequence([
        normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer=pre_tokenizers.BertPreTokenizer()
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer=trainers.WordPieceTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    cls_token_id=tokenizer.token_to_id("[CLS]")
    sep_token_id=tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor=processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id)        
        ]
    )
    tokenizer.decoder=decoders.WordPiece(prefix="##")
    fast_bert_tokenizer=BertTokenizerFast(tokenizer_object=tokenizer)

    fast_bert_tokenizer.save_pretrained(args.output_dir)

if __name__=='__main__':
    main()
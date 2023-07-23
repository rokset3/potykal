from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

if __name__ == "__main__":
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=["<unk>",
                                         "<s>",
                                         "<pad>",
                                         "<bos>",
                                         ], vocab_size=5000) #i took 5k just randomly
    tokenizer.pre_tokenizer = Whitespace()

    files = ["data/input.txt"]
    tokenizer.train(files, trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<pad>", tokenizer.token_to_id("<pad>"))
        ],
    )
    tokenizer.enable_padding(pad_id=2, pad_token="<pad>")
    tokenizer.save("data/tokenizer.json")

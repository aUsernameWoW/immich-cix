"""Build an HF tokenizers file equivalent to cn_clip.tokenize (BERT WordPiece, [CLS] ... [SEP])."""

import sys

from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

vocab_path, out_path = sys.argv[1], sys.argv[2]
vocab = {}
with open(vocab_path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        vocab[line.rstrip("\n")] = i

tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=200))
tokenizer.normalizer = normalizers.BertNormalizer(
    clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True
)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
)
tokenizer.save(out_path)
print(f"saved {out_path} with {len(vocab)} tokens")

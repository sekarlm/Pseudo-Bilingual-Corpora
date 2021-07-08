from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Create BPE model
tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase()
])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

def train_tokenizer(model, text_file, vocab_size, lang):
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True, initial_alphabet=ByteLevel.alphabet())
    model.train(files=[text_file], trainer=trainer)

    print("Trained {} vocab size: {}".format(lang, model.get_vocab_size()))
    model.model.save('./' + lang)

# Train model
ROOT_DATA = "../../data/"
train_tokenizer(model=tokenizer, text_file=ROOT_DATA+"jvwiki.txt", vocab_size=10000, lang="java")
train_tokenizer(model=tokenizer, text_file=ROOT_DATA+"suwiki.txt", vocab_size=10000, lang="sunda")


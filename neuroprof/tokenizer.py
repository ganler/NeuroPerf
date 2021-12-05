

if __name__ == '__main__':  # Train a tokenizer
    from pathlib import Path

    paths = [str(x) for x in Path("./DATASET/text/").glob("**/*.txt")]

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.enable_truncation(max_length=512)

    trainer = trainers.BpeTrainer(
        vocab_size=2048,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[
            "$START",
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
            ",",
            *r"""%rip %rax %eax %ax %al
%rcx %ecx %cx %cl
%rdx %edx %dx %dl
%rbx %ebx %bx %bl
%rsi %esi %sil %si
%rdi %edi %dil %di
%rsp %esp %spl %sp
%rbp %ebp %bpl %bp
%r8d %r8w %r8b %r8
%r9d %r9w %r9b %r9
%r10d %r10w %r10b %r10
%r11d %r11w %r11b %r11
%r12d %r12w %r12b %r12
%r13d %r13w %r13b %r13
%r14d %r14w %r14b %r14
%r15d %r15w %r15b %r15
%xmm0 %xmm1 %xmm2 %xmm3 %xmm4 %xmm5 %xmm6 %xmm7""".split()
        ],
    )

    tokenizer.train(files=paths, trainer=trainer)

    tokenizer.save("byte-level-bpe.tokenizer.json", pretty=True)


# if __name__ == '__main__':
#     from tokenizers import ByteLevelBPETokenizer
#     from pathlib import Path

#     paths = [str(x) for x in Path("./DATASET/text/").glob("**/*.txt")]

#     tokenizer = ByteLevelBPETokenizer(trim_offsets=True)

#     tokenizer.train(files=paths, vocab_size=2048, min_frequency=2,
#                     show_progress=True,
#                     special_tokens=[
#                         "<s>",
#                         "<pad>",
#                         "</s>",
#                         "<unk>",
#                         "<mask>",
#                     ])

#     tokenizer.save_model('tokenizer_model')


# if __name__ == '__main__':
#     from tokenizers import Tokenizer
#     from tokenizers.models import WordPiece
#     from tokenizers.trainers import WordPieceTrainer

#     # a pretokenizer to segment the text into words
#     from tokenizers.pre_tokenizers import Whitespace
#     from pathlib import Path

#     paths = [str(x) for x in Path("./DATASET/text/").glob("**/*.txt")]

#     tokenizer = Tokenizer(WordPiece(un))
#     trainer = WordPieceTrainer(special_tokens=spl_tokens)

#     tokenizer.train(files=paths, vocab_size=2048, min_frequency=2,
#                     show_progress=True,
#                     special_tokens=[
#                         "<s>",
#                         "<pad>",
#                         "</s>",
#                         "<unk>",
#                         "<mask>",
#                     ])

#     tokenizer.save_model('tokenizer_model')

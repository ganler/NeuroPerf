from transformers import pipeline
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast(
    tokenizer_file='./byte-level-bpe.tokenizer.json')

# Create a Fill mask pipeline
fill_mask = pipeline(
    "fill-mask",
    model='./pretrained_model/checkpoint-1888',
    tokenizer=tokenizer,
)
# Test some examples
# knit midi dress with vneckline
# =>
res = fill_mask(r"""mov	%ecx,0x278(%rsp)
cmp	$0x6,%dl
mov	$0x6,%ecx
cmovb	%edx,%<mask>
mov	%ecx,0x468(%rsp)""")
for r in res:
    print(r['token_str'])
    print(r['sequence'])

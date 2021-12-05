from transformers import pipeline
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast(
    tokenizer_file='./byte-level-bpe.tokenizer.json')

# Create a Fill mask pipeline
fill_mask = pipeline(
    "fill-mask",
    model='./pretrained_model/checkpoint-7552',
    tokenizer=tokenizer,
)
# Test some examples
# knit midi dress with vneckline
# =>
task = r"""mov	%ecx,0x278(%rsp)
cmp	$0x6,%dl
mov	$0x6,%ecx
cmovb	%edx,<mask>
mov	%ecx,0x468(%rsp)"""
res = fill_mask(task)
for r in res:
    print(f"--- score = {r['score']:.3f} : token = {r['token_str']} ---")
    print(task.replace('<mask>', r['token_str']))

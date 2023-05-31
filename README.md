# Vietnamese Sentence Paraphrasing Project
This project aims to develop a system that can paraphrase Vietnamese sentences. Paraphrasing is the process of rewriting a sentence or passage of text while maintaining its original meaning. 
This can be useful for a variety of purposes. <br />
We use paraphase dataset [Link](https://huggingface.co/datasets/humarin/chatgpt-paraphrases) in English and translate it to Vietnamese.  <br />
We also provide translated data and script for fine-tune T5 model.
# Requirements
The system will be implemented in Python and will require the following libraries:
* transformers
* sentencepiece

# Usage
```Python
# File run_gg_colab.ipynb guided how to run in google colab

# Download my fine-tuned model using vi-T5-base.
CKPT = 'chieunq/vietnamese-sentence-paraphase'
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
tokenizer = MT5Tokenizer.from_pretrained(CKPT)
model = MT5ForConditionalGeneration.from_pretrained(CKPT)

# run
def paraphase(text):
    inputs = tokenizer(text, padding='longest', max_length=64, return_tensors='pt')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=64)
    return tokenizer.decode(output[0], skip_special_tokens=True)

texts = [
        "Thật tự hào khi là sinh viên trường Đại học Công nghệ",
        "Cách kiếm nhiều tiền ?",
        "Những nguyên lí cơ bản của vật lý ?",
        "Làm thế nào để học ngôn ngữ Java",
        "Ngoài ra, nắng nóng còn có thể gây tình trạng mất nước, kiệt sức, đột qụy do sốc nhiệt đối với cơ thể người khi tiếp xúc lâu với nền nhiệt độ cao."
        ]
for text in texts:
    print(f'Input: {text}')
    print(f'Output: {paraphase(text)}')
    print('-'*100)
```

# Ouput
```
Input: Thật tự hào khi là sinh viên trường Đại học Công nghệ
Output: Là sinh viên Đại học Công nghệ, tôi rất tự hào về điều đó.
----------------------------------------------------------------------------------------------------
Input: Cách kiếm nhiều tiền ?
Output: Một số cách để kiếm được nhiều tiền là gì?
----------------------------------------------------------------------------------------------------
Input: Những nguyên lí cơ bản của vật lý ?
Output: Các nguyên tắc cơ bản của vật lý là gì?
----------------------------------------------------------------------------------------------------
Input: Làm thế nào để học ngôn ngữ Java
Output: Các bước để thành thạo ngôn ngữ Java là gì?
----------------------------------------------------------------------------------------------------
Input: Ngoài ra, nắng nóng còn có thể gây tình trạng mất nước, kiệt sức, đột qụy do sốc nhiệt đối với cơ thể người khi tiếp xúc lâu với nền nhiệt độ cao.
Output: Hơn nữa, nắng nóng có thể dẫn đến mất nước, kiệt sức, đột quỵ do sốc nhiệt đối với cơ thể người khi tiếp xúc lâu với nền nhiệt độ cao.
----------------------------------------------------------------------------------------------------
```

# Reference
@inproceedings{chatgpt_paraphrases_dataset,
  author={Vladimir Vorobev, Maxim Kuznetsov},
  title={ChatGPT paraphrases dataset},
  year={2023}
}
  

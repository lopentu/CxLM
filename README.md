# CxLM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seantyh/conart/blob/main/CxLM-samples.ipynb)

CxLM is a masked language model fine-tuned on constructions in traditional Chinese. The model generates construction-informed and context-awared candidates at "variable" sites in the constructions. The generated samples can be further applied in corpus linguistic, psycholinguistics, or behavioral studies.


```python
!pip -q install transformers
!git clone https://github.com/seantyh/CxLM
```


```python
import sys
sys.path.append("CxLM/src")
import re
import numpy as np
import torch
from transformers import BertTokenizerFast, BertForMaskedLM
from conart.sample import sample_site
```


```python
device = torch.device("cuda") \
         if torch.cuda.is_available() else torch.device("cpu")
```


```python
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('seantyh/CxLM')
model = model.to(device)
```


```python
def generate_samples(input_text):
    tokens = re.findall("\[MASK\]|.", input_text)
    mindex = [i for i, x in enumerate(tokens) if x=="[MASK]"]
    batch = {
        "masked": [tokens],
        "mindex": np.array([mindex])
    }
    
    samples = sample_site(batch, model, tokenizer, n_sample=10)[0]
    prob_sort = samples["probs"].sum(0).argsort()
    print("CxLM (separated): ")
    for site_x in tokenizer.batch_decode(samples["ids"]):
        print("  ", site_x)
    samples = sample_site(batch, model, tokenizer, merge_pair2=True)[0]
    prob_sort = samples["probs"].sum(0).argsort()
    print("CxLM (merged): ")
    print("  ", " ".join(tokenizer.batch_decode(samples["ids"])))
```


```python
generate_samples("[MASK]一[MASK]")
```

    CxLM (separated): 
       想 算 看 洗 笑 舔 聊 摸 走 動
       想 算 笑 洗 看 舔 聊 忍 摸 動
    CxLM (merged): 
       想 算 看 洗 笑 舔 聊 摸 忍 動



```python
generate_samples("買本書[MASK]一[MASK]")
```

    CxLM (separated): 
       讀 看 寫 翻 唸 買 聽 逛 走 想
       讀 看 寫 翻 想 買 唸 逛 聽 走
    CxLM (merged): 
       讀 看 寫 翻 唸 想 買 逛 聽 走



```python
generate_samples("[MASK]一[MASK]也好")
```

    CxLM (separated): 
       忍 哭 笑 算 罵 洗 死 吵 收 想
       忍 哭 算 笑 死 罵 洗 收 喊 吵
    CxLM (merged): 
       忍 哭 算 笑 死 罵 洗 收 吵 喊

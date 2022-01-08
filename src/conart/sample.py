import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda") \
         if torch.cuda.is_available() else torch.device("cpu")

def sample_site(batch, model, tokenizer, merge_pair2=False, n_sample=10, query_idxs=None, max_len=200):
    masked_tokens = tokenizer(batch["masked"], return_tensors="pt", 
                          is_split_into_words=True, padding=True, truncation=True,
                          max_length=max_len)  
    with torch.no_grad():
        masked_tokens = masked_tokens.to(model.device)
        out = model(**masked_tokens)
        logits = out.logits.cpu().numpy()
        probs = F.log_softmax(out.logits, dim=2).cpu().numpy()        
    
    mindex = batch["mindex"]
    
    samples = []
    for i in range(mindex.shape[0]):
        mindex_x = mindex[i,:]+1
        mindex_x = mindex_x[mindex_x>0]
        probs_x = probs[i, mindex_x, :]
        logits_x = logits[i, mindex_x, :]
        if len(mindex_x) == 2 and merge_pair2:            
            probs_sum = probs[i, mindex_x, :].sum(0)
            arg_x = probs_sum.argsort()[::-1][:n_sample]            
            samples.append({"ids": arg_x, 
                            "probs": probs_sum[arg_x]})
        else:
            if query_idxs is not None:
                arg_x = query_idxs
            else:
                arg_x = probs[i, mindex_x, :].argsort(axis=1)[:, ::-1][:, :n_sample]
            prob_arg = probs[i, mindex_x, :].argsort(axis=1)[:, ::-1][:, :n_sample]            
            samples.append({"ids": arg_x, 
                            "probs": np.take_along_axis(probs_x, arg_x, 1),
                            "logits": np.take_along_axis(logits_x, arg_x, 1)})
    return samples
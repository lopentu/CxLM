from itertools import chain
import numpy as np
from typing import List

def characterize(text: List[str], mask: List[str]):
    def transform_mask(mask, count):
        return ([mask] * count if not mask.startswith("B")
                else [mask] + ["I"+mask[1:]] * (count-1))
    char_mask = (transform_mask(m, len(t))
                 for t, m in zip(text, mask))
    char_mask = list(chain.from_iterable(char_mask))        
    return char_mask

def get_masked(cx_inst):
    char_cx = characterize(cx_inst["text"], cx_inst["cnstr"])
    cx_mask = [x!="O" for x in char_cx]
    char_slots = characterize(cx_inst["text"], cx_inst["slot"])
    cx_cslot = [x.endswith("C") for x in char_slots]
    cx_vslot = [x.endswith("V") for x in char_slots]
    return {"cx": cx_mask, "cslot": cx_cslot, "vslot": cx_vslot}

def get_masked_indices(masked_text):
    return [i for i, x 
            in enumerate(masked_text) 
            if x=="[MASK]"]

def get_masked_text(cx_inst, mask_field):
    text = list(chain.from_iterable(cx_inst["text"]))
    mask_dict = get_masked(cx_inst)
    MASK_FIELDS = ("cx", "cslot", "vslot", "none", "random-cx", "random-cslot", "random-vslot")
    if mask_field not in MASK_FIELDS:
        raise ValueError("Unsupported mask fields")
    if mask_field == "none":
        masked_text = text
    elif mask_field.startswith("random"):        
        tgt_field = mask_field.split("-")[1]
        n_mask = sum(mask_dict[tgt_field])
        target_mask = np.full(len(text), False)
        mask_idxs = np.random.choice(len(text), min(len(text), n_mask), replace=False)
        target_mask[mask_idxs] = True
        masked_text = [("[MASK]" if m else t) for t, m in zip(text, target_mask)]                
    else:        
        target_mask = mask_dict[mask_field]
        masked_text = [("[MASK]" if m else t) for t, m in zip(text, target_mask)]            
    masked_indices = get_masked_indices(masked_text)
    
    return {"masked": masked_text, 
            "text": text, "mindex": masked_indices}

def batched_text(data, idxs, mask_field):
    M = len(idxs)    
    
    m_data = [get_masked_text(data[i], mask_field) 
              for i in idxs]
    b_masked = [x["masked"] for x in m_data]
    b_text = [x["text"] for x in m_data]
    max_nidx = max(len(x["mindex"]) for x in m_data)
                       
    mindex_batched = np.zeros((M, max_nidx), dtype=np.int64)-1
    mindex_bool = np.zeros((M, max_nidx), dtype=bool)
    for i in range(M):
        idx_data = m_data[i]["mindex"]
        n_idx = len(idx_data)
        mindex_batched[i, :n_idx] = idx_data
        mindex_bool[i, :n_idx] = 1    
    return {"masked": b_masked, "text": b_text, 
            "mindex": mindex_batched, "mindex_bool": mindex_bool}
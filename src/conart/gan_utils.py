import torch

def make_gendcr_labels(batch, adv_ids):
    # create adv_labels, where cell value is a 1 if it is a variable (BV/IV)
    # otherwise, it's a 0.
    slot_tags = batch["slot_tags"]
    slot_mask = slot_tags != 0    
    adv_labels = (slot_tags == adv_ids[0]).clone()
    for adv_id_x in adv_ids[1:]:
        adv_labels = torch.logical_or(adv_labels, slot_tags==adv_id_x, out=adv_labels)
    
    # generate GAN real/fake labels
    gen_labels = torch.full_like(slot_tags, -100)
    gen_labels.masked_fill_(adv_labels, 1)
    # dcr_real_mask = torch.logical_and(slot_mask, adv_labels.logical_not())
    # dcr_fake_mask = torch.logical_and(slot_mask, adv_labels)
    # dcr_labels = torch.full_like(slot_tags, -100)
    # dcr_labels.masked_fill_(dcr_real_mask, 1)
    # dcr_labels.masked_fill_(dcr_fake_mask, 0)
    return {"gen_labels": gen_labels}

    
def generate_adversarials(batch, lm_probs):
    adv_ids = batch["masked_text"].input_ids.clone()    
    gen_labels = batch["gen_labels"]
    real_ids = batch["real_text"].input_ids
    adv_ids[gen_labels==1] = torch.multinomial(lm_probs[gen_labels==1], 1).squeeze()    
    dcr_labels = (adv_ids == real_ids).long()
    return {"adv_ids": adv_ids, "real_ids": real_ids, 
            "dcr_labels": dcr_labels}

def visualize_gen(batch, masked_ids, tokenizer, maxlen=20):
    real_ids = batch["real_text"].input_ids
    for i in range(real_ids.size(0)):
        real_seq = real_ids[i].tolist()[:maxlen]
        mask_seq = (masked_ids[i]>0).tolist()[:maxlen]
        tr = tokenizer.decode
        for r, m in zip(real_seq, mask_seq):
            ch = tr(r) if not m else \
                 f"\x1b[31m[MASK](\x1b[4m{tr(r)}\x1b[0;31m)\x1b[0m"
            print(ch, end='')
        print("")
        
def visualize_adv(adv_data, tokenizer, maxlen=20):
    adv_ids = adv_data["adv_ids"]
    real_ids = adv_data["real_ids"]
    dcr_masks = adv_data["dcr_labels"]
    for i in range(adv_ids.size(0)):
        adv_seq = adv_ids[i].tolist()[:maxlen]
        real_seq = real_ids[i].tolist()[:maxlen]
        dcr_seq = dcr_masks[i].tolist()[:maxlen]
        tr = tokenizer.decode
        for a, r, d in zip(adv_seq, real_seq, dcr_seq):
            ach = tr(a); rch = tr(r)
            ch = rch if d else \
                 f"\x1b[31m{ach}(\x1b[4m{rch}\x1b[0;31m)\x1b[0m"
            print(ch, end='')
        print("")

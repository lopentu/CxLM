def list_equal(x, y):    
    if len(x) != len(y):
        return False
    return all(xx==yy for xx, yy in zip(x,y))


def get_form_offset(cxinst):
    text = cxinst["text"]
    example = cxinst["cnstr_example"]
    form = cxinst["cnstr_form"]
    elem_map = []
    form_len = len(form)
    for i in range(len(text)-len(form)+1):    
        if not list_equal(text[i:i+form_len], example):
            continue            
        for offset, elem in enumerate(form):
            elem_map.append(i + offset)
    return elem_map

def get_form_groups(cxinst):
    form = cxinst["cnstr_form"]
    form_groups = {}
    form_offset = get_form_offset(cxinst)    
    if not form_offset:
        raise ValueError("Cannot determine form_offset, possibly because variable phrase is involved")
    for idx, elem in enumerate(form):        
        form_groups.setdefault(elem, [])\
                   .append(form_offset[idx])
    return form_groups

def get_tok_char_map(cxinst):    
    toks = cxinst["text"]
    tok_char_map = {}
    counter = 0
    for tok_i, tok in enumerate(toks):        
        tok_char_map[tok_i] = [char_i+counter for char_i in range(len(tok))]
        counter += len(tok)
    return tok_char_map

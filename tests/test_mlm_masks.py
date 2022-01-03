from import_conart import conart

import json
from import_conart import conart
from conart.mlm_masks import *
with open("test_mlm_data.json", "r") as fin:
    TEST_DATA = json.load(fin)
test_data = {
    "text": ["中文", "測試", "構式", "猜想", "來", "猜想", "去"],
    "cnstr": ["O", "O", "O", "BX", "IX", "IX", "IX"],
    "slot": ["O", "O", "O", "BV", "BC", "BV", "BC"]
}

def test_characterize():
    cmask = characterize(["測試","中文字","構式","吧"], "O,BX,IX,O".split(","))        
    assert cmask == "O,O,BX,IX,IX,IX,IX,O".split(",")
    smask = characterize(["測試","中文字","構式","吧"], "BV,IV,BC,IC".split(","))        
    assert smask == "BV,IV,IV,IV,IV,BC,IC,IC".split(",")
    
def test_get_masked():
    mask_dict = get_masked(test_data)
    cx_ans    = [0,0,0,0,0,0,1,1,1,1,1,1]
    cslot_ans = [0,0,0,0,0,0,0,0,1,0,0,1]
    vslot_ans = [0,0,0,0,0,0,1,1,0,1,1,0]
    assert mask_dict["cx"] == [bool(x) for x in cx_ans]
    assert mask_dict["cslot"] == [bool(x) for x in cslot_ans]
    assert mask_dict["vslot"] == [bool(x) for x in vslot_ans]
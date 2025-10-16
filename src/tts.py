# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
import re
import random
import matplotlib.colors as mcolors
import logging
from utils_tts import *
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""## Find steps with high and low TTS, which are then used to extract TrueThinking directions  

"""

d_no_perturb=read_jsonl('') #p(y|C,s)
d_perturb_s=read_jsonl('') #p(y|C,s')
d_perturb_s_c=read_jsonl('') #p(y|C',s')
d_perturb_c=read_jsonl('') #p(y|C',s)

ret_uf=[]
ret_ff=[]
for id in range(len(d_perturb_s)):
  if is_consistent(d_perturb_s[id]):
    continue
  uf,ff,tts,_,_=analyze_ret(id,analyze_mode=False,t_uf=0,t_ff=0.9,a=0.5,b=0.5, \
                              d_perturb_early_exit=d_perturb_s,\
                              d_initial_early_exit=d_no_perturb,\
                              d_reverse_perturb_before_early_exit=d_perturb_c,\
                              d_reverse_perturb_all_early_exit=d_perturb_s_c,\
                              use_abs=True,perturb_mode='replace',score='add',apply_constraint=False)

  chunks=d_perturb_s[id]['ori_output'].split('.')
  key='problem'
  for i in uf:
    ckpt=d_perturb_s[id]['checkpoint_analysis'][f'checkpoint_{i+1}']
    if ckpt['perturbed']:
      cur={'case_id':id,'problem':d_perturb_s[id][key],'answer':d_perturb_s[id]['answer'],'perturb_step':ckpt['text_up_to_checkpoint_perturbed'],\
            'initial_step':ckpt['text_up_to_checkpoint'],'context_text':'.'.join(chunks[:i-1])}
      ret_uf.append(cur)
  for i in ff:
    ckpt=d_perturb_s[id]['checkpoint_analysis'][f'checkpoint_{i+1}']
    if ckpt['perturbed']:
      cur={'case_id':id,'problem':d_perturb_s[id][key],'answer':d_perturb_s[id]['answer'],'perturb_step':ckpt['text_up_to_checkpoint_perturbed'],\
            'initial_step':ckpt['text_up_to_checkpoint'],'context_text':'.'.join(chunks[:i-1])}
      ret_ff.append(cur)

with open('low_tts_steps.jsonl', 'w') as f:
    random.seed(100)
    download=random.sample(ret_uf,min(200,len(ret_uf)))
    for item in download:
        f.write(json.dumps(item) + '\n')

with open('high_tts_steps.jsonl', 'w') as f:
    random.seed(100)
    download=random.sample(ret_ff,min(200,len(ret_ff)))
    for item in download:
        f.write(json.dumps(item) + '\n')
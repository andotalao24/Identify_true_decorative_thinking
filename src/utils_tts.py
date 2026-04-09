def load_checkpoint_data(file_path):
    """Robustly load multi-line JSON objects from a file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        buffer = []
        brace_count = 0
        for line in f:
            if '{\n' ==line:
                brace_count += line.count('{')
            if '}\n' == line:
                brace_count -= line.count('}')
            buffer.append(line)

            if brace_count == 0 and buffer:
                block = ''.join(buffer).strip()
                try:
                    #print(f"Loading block: {block}")
                    obj = json.loads(block)
                    data.append(obj)
                except Exception as e:
                    print(f"Skipping malformed block: {e}")
                buffer = []
    print(f"Successfully loaded {len(data)} samples")
    return data
    
def read_jsonl(file):
    ret=[]
    try:
        with open(file,'r', encoding="UTF-8") as f:
            for row in f.readlines():
                d=json.loads(row)
                ret.append(d)
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        with open(file,'r') as f:
            ret=json.load(f)
    return ret


def compare_answers(extracted_ans, correct_ans):
    extracted_ans = str(extracted_ans).split("$")[0].split("\n")[0]
    for sign in [',',':','%']:
        extracted_ans = extracted_ans.replace(sign, '')
    extracted_ans = str(extracted_ans).strip()
    correct_ans = str(correct_ans).strip()
    extracted_ans_no_space= re.sub(r"\s+", "", extracted_ans)
    correct_ans_no_space= re.sub(r"\s+", "", correct_ans)
    if extracted_ans_no_space == correct_ans_no_space:
        return True
    try:
        extracted_float = float(extracted_ans)
        correct_float = float(correct_ans)

        if abs(extracted_float - correct_float) < 1e-6:
            return True
    except Exception as e:
        pass
    try:
        if '/' in extracted_ans:
            num, denom = extracted_ans.split('/')
            extracted_float = float(num) / float(denom)
            correct_float = float(correct_ans)
            if abs(extracted_float - correct_float) < 1e-6:
                return True
        elif '/' in correct_ans:
            num, denom = correct_ans.split('/')
            correct_float = float(num) / float(denom)
            extracted_float = float(extracted_ans)
            if abs(extracted_float - correct_float) < 1e-6:
                return True
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return False
    
def parse_ans(ans):
  if isinstance(ans,float):
    return ans
  try:
    ans=ans.split('boxed{')[-1].split('}')[0]
  except:
    pass
  return ans

def if_error_in_ckpt(d):
  ret=set()
  for i,dt in enumerate(d):
    for ckpt in dt['checkpoint_analysis'].keys():
      if 'error' in dt['checkpoint_analysis'][ckpt]:
        ret.add(i)
        break
  return ret

def find_first_ckpt_show_correct_ans(d):
  ans=d['ori_output'].split('boxed{')[-1].split('}')[0]
  if not compare_answers(ans,parse_ans(d['answer'])):
    print(ans,d['answer'])
  for i,ckpt in enumerate(d['checkpoint_analysis'].values()):
    if i==0:
      continue
    text=ckpt['text_up_to_checkpoint'] if 'text_up_to_checkpoint_initial' not in ckpt else ckpt['text_up_to_checkpoint_initial']
    ans_in_text=False
    for pattern in [f'is {ans}', f'={ans}',f'= {ans}',f'{ans}}}']:
      if pattern in text:
        ans_in_text=True
        break
    if ans in text and 'so' in text.lower():
      ans_in_text=True
    if ans_in_text:
      return i
      break

  return len(d['checkpoint_analysis'])

def get_first_ckpt_ee_right(d):
  for i in range(len(d['checkpoint_analysis'])):
    if compare_answers(d['checkpoint_analysis'][f'checkpoint_{i+1}']['extracted_generated_answer'],parse_ans(d['answer'])):
      return i
  return len(d['checkpoint_analysis'])-1

def is_consistent(d,upper=9999):
  first_ckpt_ee_right=get_first_ckpt_ee_right(d)
  count=0
  for i in range(first_ckpt_ee_right+1,len(d['checkpoint_analysis'])):
    if not compare_answers(d['checkpoint_analysis'][f'checkpoint_{i+1}']['extracted_generated_answer'],parse_ans(d['answer'])):
      count+=1
  if count <= upper and count>0:
    return False
  return True
  
def perturb_ee_score(case_id,d_perturb,d_initial,d_perturb_refer,use_abs=False,perturb_mode='replace',eval_mode='positive'):
  #d is one example
  """
  if reverse score:
    d_perturb is reverse_before_ckpt
    d_initial is reverse all, including ckpt
  """
  ret=[]
  assert len(d_perturb['checkpoint_analysis'])==len(d_initial['checkpoint_analysis'])
  total_ckpt=len(d_perturb['checkpoint_analysis'])
  first_correct_ans_found=False

  for i in range(1,total_ckpt):
    if 'perturbed' not in d_perturb_refer['checkpoint_analysis'][f'checkpoint_{i+1}']:
      print(case_id,i)
      continue
    if not d_perturb_refer['checkpoint_analysis'][f'checkpoint_{i+1}']['perturbed']:
      perturb_mode_local='drop'
    else:
      perturb_mode_local=perturb_mode
    #p_perturb,p_initial=0,0
    if perturb_mode_local=='replace':
      if compare_answers(d_perturb['checkpoint_analysis'][f'checkpoint_{i+1}']['extracted_generated_answer'],\
                                                       parse_ans(d_perturb['answer'])):
         p_perturb=d_perturb['checkpoint_analysis'][f'checkpoint_{i+1}']['extracted_generated_answer_probability']
      else:
        p_perturb=d_perturb['checkpoint_analysis'][f'checkpoint_{i+1}']['ans_probability_multi']

      if compare_answers(d_initial['checkpoint_analysis'][f'checkpoint_{i+1}']['extracted_generated_answer'],\
                                                       parse_ans(d_initial['answer'])):

        p_initial=d_initial['checkpoint_analysis'][f'checkpoint_{i+1}']['extracted_generated_answer_probability']
      else:
        p_initial=d_initial['checkpoint_analysis'][f'checkpoint_{i+1}']['ans_probability_multi']
      if use_abs:
        ret.append(abs(p_initial-p_perturb))
      else:
        ret.append(p_initial-p_perturb)

    elif perturb_mode_local=='drop':
      if  eval_mode=='positive':
        if compare_answers(d_initial['checkpoint_analysis'][f'checkpoint_{i}']['extracted_generated_answer'],\
                                                        parse_ans(d_initial['answer'])):
          p_perturb=d_initial['checkpoint_analysis'][f'checkpoint_{i}']['extracted_generated_answer_probability']
        else:
          p_perturb=d_initial['checkpoint_analysis'][f'checkpoint_{i}']['ans_probability_multi']
      elif eval_mode=='negative':
        if compare_answers(d_perturb['checkpoint_analysis'][f'checkpoint_{i}']['extracted_generated_answer'],\
                                                        parse_ans(d_initial['answer'])):
          p_perturb=d_perturb['checkpoint_analysis'][f'checkpoint_{i}']['extracted_generated_answer_probability']
        else:
          p_perturb=d_perturb['checkpoint_analysis'][f'checkpoint_{i}']['ans_probability_multi']

      if compare_answers(d_initial['checkpoint_analysis'][f'checkpoint_{i+1}']['extracted_generated_answer'],\
                                                       parse_ans(d_initial['answer'])):
        p_initial=d_initial['checkpoint_analysis'][f'checkpoint_{i+1}']['extracted_generated_answer_probability']
      else:
        p_initial=d_initial['checkpoint_analysis'][f'checkpoint_{i+1}']['ans_probability_multi']

      if use_abs:
        ret.append(abs(p_initial-p_perturb))
      else:
        ret.append(p_initial-p_perturb)

  return ret

def final_uf_score_add(ret_case1_pos,ret_case1_neg,a=0.5,b=0.5):
  return a*np.array(ret_case1_pos)+b*np.array(ret_case1_neg)

def analyze_ret(case_idx,analyze_mode=True,t_uf=0,t_ff=0.5,a=1,b=1, \
                d_perturb_early_exit=d_perturb_early_exit,d_initial_early_exit=d_initial_early_exit,\
                d_reverse_perturb_before_early_exit=d_reverse_perturb_before_early_exit,\
                d_reverse_perturb_all_early_exit=d_reverse_perturb_all_early_exit,\
                use_abs=True,apply_constraint=False):
  uf=[]
  ff=[]
  ret_case1_pos=perturb_ee_score(case_idx,d_perturb_early_exit[case_idx],d_initial_early_exit[case_idx],d_perturb_refer=d_perturb_early_exit[case_idx],use_abs=use_abs,perturb_mode=perturb_mode,eval_mode='positive')
  ret_case1_neg=perturb_ee_score(case_idx,d_reverse_perturb_before_early_exit[case_idx],d_reverse_perturb_all_early_exit[case_idx],d_perturb_refer=d_perturb_early_exit[case_idx],use_abs=use_abs,perturb_mode=perturb_mode,eval_mode='negative')
  x,y=[],[]
  s=final_uf_score_add(ret_case1_pos,ret_case1_neg,a=a,b=b)

  first_ckpt_show_ans=find_first_ckpt_show_correct_ans(d_perturb_early_exit[case_idx])
  first_ckpt_ee_right=get_first_ckpt_ee_right(d_perturb_early_exit[case_idx])

  for i in range(1,len(d_initial_early_exit[case_idx]['checkpoint_analysis'])):

    if apply_constraint and i<first_ckpt_ee_right:
      continue
    if apply_constraint and i==first_ckpt_show_ans:
      break
    x.append(i)
    y.append(s[i-1])


    if not use_abs:
      if s[i-1]<=t_uf: #lower
        uf.append(i)
      elif s[i-1]>np.mean(np.array(s))+2*np.std(np.array(s)): #higher
        ff.append(i)
    else: 
      if s[i-1]<=t_uf:
        uf.append(i)
      elif s[i-1]>t_ff:
        ff.append(i)


  #print(np.array(s).mean,np.array(s).std)
  return uf,ff,s,x,y
  
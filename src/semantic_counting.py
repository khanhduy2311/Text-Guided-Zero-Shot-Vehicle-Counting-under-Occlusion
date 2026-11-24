"""Annotation-free query validation, conservative association, and ID-based counts."""
import math

def validate_queries(config):
    if config.get('object_detector','A3') not in ('A3','official') or config.get('subject_detector','official') not in ('A3','official'):raise ValueError('Supported frozen detector variants: A3, official')
    queries=config.get('queries',[])
    if not queries:raise ValueError('Provide at least one query')
    seen=set()
    for q in queries:
        if not isinstance(q.get('id'),str) or not q['id'] or q['id'] in seen:raise ValueError('Query IDs must be unique nonempty strings')
        seen.add(q['id'])
        if q.get('kind') not in ('class','attribute','relation'):raise ValueError('Unknown query kind')
        prompts=[q.get('object_prompt','')]
        if q['kind']=='relation':
            if q.get('relation') not in ('riding','near'):raise ValueError('Supported geometric relation candidates: riding, near')
            if q.get('subject_region','full') not in ('full','upper_body'):raise ValueError('Invalid subject crop region')
            prompts.append(q.get('subject_prompt',''))
        for prompt in prompts:
            if not isinstance(prompt,str) or not prompt.strip() or any(x in prompt for x in '.?'):raise ValueError('Detection prompts must be one positive category')
        if q['kind']!='class':
            for key in ('positive','negative'):
                if not isinstance(q.get(key),list) or not q[key] or any(not isinstance(s,str) or not s.strip() for s in q[key]):raise ValueError('Explicit positive AND contrastive negative descriptions required')
        if ('subject_positive' in q)!=('subject_negative' in q):raise ValueError('Subject attributes require positive and negative descriptions')
        if 'subject_positive' in q:
            if q['kind']!='relation':raise ValueError('Subject attribute only applies to relation queries')
            for key in ('subject_positive','subject_negative'):
                if not q[key] or not isinstance(q[key],list) or any(not isinstance(s,str) or not s.strip() for s in q[key]):raise ValueError('Invalid subject attribute descriptions')
    policy=config.get('policy',{})
    overlap={q['object_prompt'] for q in queries}&{q['subject_prompt'] for q in queries if q['kind']=='relation'}
    if overlap and config.get('object_detector','A3')!=config.get('subject_detector','official'):raise ValueError('A prompt used for both roles must use the same detector variant')
    for key,default in [('cosine_margin',.02),('association_margin',.1),('context_expansion',.15),('minimum_side_px',12)]:
        value=policy.get(key,default)
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:raise ValueError('Invalid policy '+key)
    return queries

def decision(positive,negative,margin):
    if not all(math.isfinite(x) for x in [positive,negative,margin]):raise ValueError('Nonfinite semantic score')
    delta=positive-negative
    if delta>0 and delta>=margin:return 'matched'
    if delta<0 and -delta>=margin:return 'rejected'
    return 'unknown'

def box_valid(box,size):
    w,h=size
    return len(box)==4 and all(math.isfinite(v) for v in box) and 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h

def union_box(a,b):return [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]

def association_score(subject,obj,relation):
    """Geometry proposes a candidate, never asserts the semantic relation is true."""
    sx1,sy1,sx2,sy2=subject;ox1,oy1,ox2,oy2=obj
    sw,sh=sx2-sx1,sy2-sy1;ow,oh=ox2-ox1,oy2-oy1
    if min(sw,sh,ow,oh)<=0:return None
    scx,scy=(sx1+sx2)/2,(sy1+sy2)/2;ocx,ocy=(ox1+ox2)/2,(oy1+oy2)/2
    distance=math.hypot(scx-ocx,scy-ocy)/max(math.hypot(ow,oh),1.)
    if relation=='near':return 1/(1+distance) if distance<=2 else None
    overlap=max(0,min(sx2,ox2)-max(sx1,ox1))/min(sw,ow)
    feet=(sy2-oy1)/oh
    if overlap<.2 or not -.35<=feet<=1.5 or scy>oy2 or sy1>ocy:return None
    return .6*overlap+.4/(1+abs(scx-ocx)/ow+abs(feet-.5))

def associate(subjects,objects,relation,ambiguity_margin=.1):
    """Assign each subject to at most one unambiguous object; passengers may share it."""
    assigned=[];ambiguous=set();unassigned=[]
    for subject in subjects:
        ranked=[]
        for obj in objects:
            score=association_score(subject['box'],obj['box'],relation)
            if score is not None:ranked.append((score,obj['id'],obj))
        ranked.sort(key=lambda x:(-x[0],x[1]))
        if not ranked:unassigned.append(subject['id']);continue
        if len(ranked)>1 and ranked[0][0]-ranked[1][0]<ambiguity_margin:
            ambiguous.update(item[1] for item in ranked if ranked[0][0]-item[0]<ambiguity_margin);continue
        score,_,obj=ranked[0]
        assigned.append({'subject_id':subject['id'],'object_id':obj['id'],'box':union_box(subject['box'],obj['box']),'subject_box':subject['box'],'geometry_score':score})
    return assigned,ambiguous,unassigned

def reduce_pairs(objects,pairs,ambiguous):
    result=[]
    for obj in objects:
        relevant=[p for p in pairs if p['object_id']==obj['id']]
        states=[p['status'] for p in relevant]
        if 'matched' in states:state='matched';reason='at_least_one_matching_subject'
        elif not states or obj['id'] in ambiguous or 'unknown' in states:state='unknown';reason='missing_ambiguous_or_uncertain_subject'
        else:state='rejected';reason='candidate_pairs_rejected'
        result.append({**obj,'status':state,'reason':reason,'pair_ids':[p['pair_id'] for p in relevant]})
    return result

def summarize_objects(objects):
    ids=[r['id'] for r in objects]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate object IDs would double-count')
    return {'count':sum(r['status']=='matched' for r in objects),'unknown_count':sum(r['status']=='unknown' for r in objects),'rejected_count':sum(r['status']=='rejected' for r in objects),'detected_objects':len(objects),'count_unit':'unique detected object IDs','interpretation':'zero-shot estimate; unknowns excluded, no accuracy claim'}

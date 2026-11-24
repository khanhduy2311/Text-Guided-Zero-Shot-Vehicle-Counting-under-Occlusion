"""Open-text routing. Color pairs share a serving policy; research stays frozen.

This module selects an inference strategy. It is not a whitelist of valid queries.
Every additional condition is retained in semantic text or an explicit spatial rule.
"""
import re,unicodedata
from web_demo.query_legacy import parse_query as legacy_parse,normalize,COLORS,NOUNS,ALIASES,COLOR_ALIASES

OPEN_COLORS={**COLORS,'brown':'nâu','pink':'hồng','purple':'tím','beige':'be'}
COLOR_WORDS={**COLOR_ALIASES,'brown':['brown','nau'],'pink':['pink','hong'],'purple':['purple','violet','tim'],'beige':['beige','tan','be']}
VEHICLES={**NOUNS,'pickup truck':'xe bán tải','SUV':'xe SUV','taxi':'taxi','ambulance':'xe cứu thương','fire truck':'xe cứu hỏa','police car':'xe cảnh sát','tractor':'máy kéo','forklift':'xe nâng','train':'tàu hỏa','tram':'xe điện','trailer':'rơ moóc','scooter':'xe tay ga'}
WORDS={**ALIASES,
 'pickup truck':['pickup truck','pickup trucks','pickup','pickups','xe ban tai'],
 'SUV':['suv','suvs','xe suv'], 'taxi':['taxi','taxis','cab','cabs'],
 'ambulance':['ambulance','ambulances','xe cuu thuong'],
 'fire truck':['fire truck','fire trucks','fire engine','fire engines','xe cuu hoa'],
 'police car':['police car','police cars','xe canh sat'],
 'tractor':['tractor','tractors','may keo','xe dau keo'],
 'forklift':['forklift','forklifts','xe nang'],
 'train':['train','trains','tau hoa'], 'tram':['tram','trams','xe dien'],
 'trailer':['trailer','trailers','ro mooc'],
 'scooter':['scooter','scooters','xe tay ga']}
PREFIXES=['hay dem so luong','dem so luong','hay dem','dem cac','dem so','dem','count the number of','count all the','count all','count the','count','all the','all']
POSITION={
 'left':[r'(?:on|in|at) the left(?: side)?(?: of (?:the |this )?(?:image|picture|photo))?',r'(?:o |nam o )?ben trai(?: (?:cua |trong )?(?:anh|buc anh))?'],
 'right':[r'(?:on|in|at) the right(?: side)?(?: of (?:the |this )?(?:image|picture|photo))?',r'(?:o |nam o )?ben phai(?: (?:cua |trong )?(?:anh|buc anh))?'],
 'top':[r'(?:at|in) the top(?: half)?(?: of (?:the |this )?(?:image|picture|photo))?',r'o nua tren(?: (?:cua )?anh)?'],
 'bottom':[r'(?:at|in) the bottom(?: half)?(?: of (?:the |this )?(?:image|picture|photo))?',r'o nua duoi(?: (?:cua )?anh)?']}

def strip_instruction(text):
 text=unicodedata.normalize('NFC',text).strip().rstrip('.!?').strip()
 s=normalize(text)
 for prefix in PREFIXES:
  if s.startswith(prefix+' '):
   text=' '.join(text.split()[len(prefix.split()):]);break
 return text

def target_noun(text):
 s=normalize(text);hits=[]
 for noun,aliases in WORDS.items():
  for a in aliases:
   # "xe" is only a generic fallback; do not override xe bus/xe tải.
   if a=='xe':continue
   for m in re.finditer(r'(?<!\w)'+re.escape(a)+r'(?!\w)',s):hits.append((m.start(),-len(a),noun))
 return min(hits)[2] if hits else 'vehicle'

def spatial_query(text):
 s=normalize(text)
 for side,patterns in POSITION.items():
  for pattern in patterns:
   m=re.search(r'\s+('+pattern+r')$',s)
   if m:
    # Normalization preserves word count, including Vietnamese diacritics.
    keep=len(s[:m.start()].split())
    if keep:return ' '.join(text.split()[:keep]),[side]
 return text,[]

def needs_category_verification(query):
 """The motorcycle-only request must not bypass crop type verification."""
 return query['kind']=='count' and query['noun']=='motorcycle'

def category_route(query):
 if needs_category_verification(query):
  query.update(pipeline='A3 + SigLIP2 · xác minh loại xe',route='verified_category',
               detail='Tìm ứng viên xe máy, rồi xác minh loại xe trên từng vùng ảnh trước khi đếm.')
 return query

def parse_query(text):
 if not isinstance(text,str) or not text.strip():raise ValueError('Nhập mô tả phương tiện bạn muốn đếm.')
 if len(text)>160:raise ValueError('Mô tả tối đa 160 ký tự; hãy viết gọn các đặc điểm cần đếm.')
 if any(ord(c)<32 and c not in '\t\n\r' for c in text):raise ValueError('Mô tả chứa ký tự điều khiển không hợp lệ.')
 if not any(c.isalpha() for c in text):raise ValueError('Hãy nhập một mô tả có chữ, ví dụ “green bus”.')
 original=text.strip();phrase=strip_instruction(original);core,spatial=spatial_query(phrase)
 try:
  q=legacy_parse(core)
  q.update(original=original,semantic_prompt=q['canonical'],spatial=spatial,route='validated_recipe',warnings=[])
  if q['kind']=='color':
   # Serve every color/type pair through the same open-color policy. The old
   # car-only CV recipe stays frozen for reproducibility in research scripts.
   q.update(kind='open_color',color_negated=False,route='zero_shot_color',
            pipeline='A3 + SigLIP2 · màu mở',
            detail='Tìm xe bằng loại xe và mô tả màu, rồi xác minh từng vùng ảnh.',
            warnings=['Nhánh zero-shot mở rộng; chưa có benchmark cho mọi cặp màu–loại xe.'])
  if spatial:
   q['canonical']+=' on the '+spatial[0];q['label']=original;q['detail']+=' Vị trí được lọc theo tâm box trong nửa ảnh tương ứng.'
  return category_route(q)
 except ValueError:pass
 s=normalize(core)
 s=re.sub(r'^(?:cac|nhung|the|a|an)\s+','',s)
 s=re.sub(r'\s+(?:trong anh|trong buc anh|in this image|in the image|in the picture)$','',s)
 for noun,aliases in WORDS.items():
  if s in aliases:
   q={'kind':'count','noun':noun,'color':None,'canonical':noun,'label':VEHICLES.get(noun,noun),'pipeline':'A3 · filtered_pseudo3','detail':'Đếm theo tên phương tiện mở; chất lượng tùy loại xe.','original':original,'semantic_prompt':noun,'spatial':spatial,'route':'open_category','warnings':[]}
   if spatial:q['canonical']+=' on the '+spatial[0];q['label']=original
   return category_route(q)
 for color,colors in COLOR_WORDS.items():
  for c in colors:
   for noun,aliases in WORDS.items():
    for n in aliases:
     positive=[f'{c} {n}',f'{n} {c}',f'{n} mau {c}',f'{n} co mau {c}']
     negative=[f'not {c} {n}',f'non {c} {n}',f'{n} that are not {c}',f'{n} khong co mau {c}',f'{n} khong mau {c}']
     if s in positive+negative:
      negate=s in negative;canonical=('not ' if negate else '')+color+' '+noun
      return {'kind':'open_color','noun':noun,'color':color,'color_negated':negate,'canonical':canonical+(' on the '+spatial[0] if spatial else ''),'label':original,'pipeline':'A3 + SigLIP2 · màu mở','detail':'Tìm đúng loại xe rồi so màu với các mô tả đối chứng của loại xe đó.','original':original,'semantic_prompt':canonical,'spatial':spatial,'route':'zero_shot_color','warnings':['Nhánh zero-shot mở rộng; chưa có benchmark cho mọi cặp màu–loại xe.']}
 noun=target_noun(core)
 warnings=['Mô tả tự do được đưa vào mô hình; các quan hệ phức tạp vẫn có thể bị hiểu sai.']
 if re.search(r'\bxanh\b',s) and not any(c in s for c in ['xanh duong','xanh la','xanh luc','xanh nuoc bien']):warnings.append('“Xanh” có thể chỉ xanh dương hoặc xanh lá; viết rõ màu sẽ dễ kiểm tra kết quả hơn.')
 return {'kind':'open_text','noun':noun,'color':None,'canonical':phrase,'label':original,'pipeline':'A3 + SigLIP2 · mô tả tự do','detail':'A3 dùng mô tả đầy đủ và loại xe để tìm ứng viên; SigLIP2 đối chiếu nguyên mô tả trên crop và ngữ cảnh.','original':original,'semantic_prompt':core,'spatial':spatial,'route':'open_text_v1','warnings':warnings}

def capabilities():
 return {'open_text':True,'colors':OPEN_COLORS,'vehicles':VEHICLES,'relation':'motorcycle with rider','examples':['green bus','black truck','a blue delivery van','xe máy màu đỏ','red car on the left'],'max_prompt_length':160,'scope':'Free-text descriptions are accepted; correctness of arbitrary descriptions is not guaranteed.'}

def spatial_match(box,size,constraints):
 x=(box[0]+box[2])/2/size[0];y=(box[1]+box[3])/2/size[1]
 return all({'left':x<.5,'right':x>=.5,'top':y<.5,'bottom':y>=.5}[side] for side in constraints)

"""Explicit bilingual query grammar; never silently drop unsupported conditions."""
import re,unicodedata

COLORS={'black':'đen','blue':'xanh dương','gray':'xám','green':'xanh lá','orange':'cam','red':'đỏ','white':'trắng','yellow':'vàng'}
NOUNS={'car':'xe ô tô','vehicle':'phương tiện','motorcycle':'xe máy','bicycle':'xe đạp','bus':'xe buýt','truck':'xe tải','van':'xe van'}
ALIASES={
 'car':['car','cars','automobile','automobiles','xe','xe hoi','o to','xe o to','oto','xe oto'],
 'vehicle':['vehicle','vehicles','phuong tien','cac phuong tien'],
 'motorcycle':['motorcycle','motorcycles','motorbike','motorbikes','scooter','scooters','xe may','xe mo to'],
 'bicycle':['bicycle','bicycles','bike','bikes','xe dap'],
 'bus':['bus','buses','xe buyt','xe bus'],
 'truck':['truck','trucks','xe tai'],
 'van':['van','vans','xe van'],
}
COLOR_ALIASES={'black':['black','den'],'blue':['blue','xanh duong','xanh nuoc bien'],'gray':['gray','grey','silver','xam','bac'],'green':['green','xanh la','xanh luc'],'orange':['orange','cam'],'red':['red','do'],'white':['white','trang'],'yellow':['yellow','vang']}
RELATIONS=['motorcycle with rider','motorcycles with riders','motorcycle with a rider','motorcycles with a rider','person riding a motorcycle','people riding motorcycles','person riding motorcycle','motorcycle being ridden','xe may co nguoi lai','xe may co nguoi','xe may co nguoi ngoi','xe may dang duoc lai','nguoi dang di xe may','nguoi lai xe may','nguoi di xe may']
def normalize(value):
 value=unicodedata.normalize('NFKD',value.lower().replace('đ','d'))
 value=''.join(c for c in value if not unicodedata.combining(c))
 return ' '.join(value.strip().strip('.!?').split())

def parse_query(text):
 if not isinstance(text,str) or not text.strip():raise ValueError('Nhập mô tả vật cần đếm, ví dụ “red car” hoặc “xe ô tô màu đỏ”.')
 if len(text)>160:raise ValueError('Mô tả tối đa 160 ký tự.')
 s=normalize(text)
 for prefix in ['hay dem so luong ','dem so luong ','hay dem ','dem cac ','dem so ','dem ','count the number of ','count all the ','count all ','count the ','count ','all the ','all ']:
  if s.startswith(prefix):s=s[len(prefix):];break
 s=re.sub(r'^(cac|nhung|the|a|an)\s+','',s)
 s=re.sub(r'\s+(trong anh|trong buc anh|in this image|in the image|in the picture)$','',s)
 if s in RELATIONS:
  return {'kind':'relation','noun':'motorcycle','color':None,'canonical':'motorcycle with rider','label':'Xe máy có người lái','pipeline':'A3 + SigLIP2 · quan hệ','detail':'Đếm phương tiện có người lái, không đếm số người.','original':text.strip()}
 for noun,aliases in ALIASES.items():
  if s in aliases:return {'kind':'count','noun':noun,'color':None,'canonical':noun,'label':NOUNS[noun].capitalize(),'pipeline':'A3 · filtered_pseudo3','detail':'Đếm và định vị theo tên phương tiện.','original':text.strip()}
 for color,aliases in COLOR_ALIASES.items():
  for c in aliases:
   for n in ALIASES['car']:
    if s in [f'{c} {n}',f'{n} {c}',f'{n} mau {c}',f'{n} co mau {c}']:
     return {'kind':'color','noun':'car','color':color,'canonical':color+' car','label':'Xe ô tô màu '+COLORS[color],'pipeline':'A3 + SigLIP2 · màu sắc','detail':'Tìm xe trước, sau đó kiểm tra màu trên từng vùng ảnh.','original':text.strip()}
 if re.search(r'\bxanh\b',s) and not any(c in s for c in ['xanh duong','xanh nuoc bien','xanh la','xanh luc']):
  raise ValueError('Bạn muốn xanh dương hay xanh lá? Hãy ghi rõ, ví dụ “xe màu xanh dương”.')
 raise ValueError('Mô tả này chưa được hỗ trợ đầy đủ. Thử “red car”, “xe ô tô màu trắng”, “xe máy có người lái” hoặc tên phương tiện. Hiện chưa hỗ trợ kết hợp màu với vị trí, phủ định hay nhiều điều kiện.')

def capabilities():
 return {'colors':COLORS,'vehicles':NOUNS,'relation':'motorcycle with rider','examples':['red car','xe ô tô màu trắng','car','xe máy có người lái'],'max_prompt_length':160}

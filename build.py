"""Build the Ming Crisis Atlas dataset from REACHES (NOAA WDS Paleo) + climate context series."""
import pandas as pd, numpy as np, re, json, os

RAW = 'raw'
OUT = 'out/data'
os.makedirs(OUT, exist_ok=True)


def tidy_codebooks(src=f'{RAW}/codebooks.txt', dst=f'{RAW}/codebook_tidy.tsv'):
    """Flatten the three wide REACHES coding guides into one row per (category, subcategory, term)."""
    import csv
    txt = open(src, encoding='utf-8').read()
    out = []
    for part in txt.split('@@FILE ')[1:]:
        _, body = part.split('\n', 1)
        rows = [l.rstrip('\r').split('\t') for l in body.split('\n') if not l.startswith('#')]
        hi = [i for i, r in enumerate(rows) if r and r[0] == 'Sub category'][0]
        mains = rows[hi - 1]
        cats = [(mains[k], mains[k + 1]) for k in range(0, len(mains) - 1, 6) if mains[k]]
        for ci, (mname, mcode) in enumerate(cats):
            sub = subname = None
            for r in rows[hi + 1:]:
                c = (r + [''] * 400)[ci * 6:ci * 6 + 6]
                if c[0].strip() and c[1].strip():
                    sub, subname = c[1].strip(), c[0].strip()
                if c[3].strip() and sub is not None:
                    out.append((mcode, mname, sub, subname, c[3].strip(), c[2].strip()))
    with open(dst, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['main', 'main_name', 'sub', 'sub_name', 'vocab', 'vocab_label'])
        w.writerows(out)


tidy_codebooks()

# ---------------------------------------------------------------- records
txt = open(f'{RAW}/reaches_ming_compact.txt').read().split('\n')
li, ri = txt.index('L'), txt.index('R')
locs = [tuple(map(float, x.split(','))) for x in txt[li + 1:ri]]
rows = [r.split('\t') for r in txt[ri + 1:] if r]
rec = pd.DataFrame(rows, columns=['record_id', 'year', 'month', 'year_end', 'loc', 'codes'])
for c in ['record_id', 'year', 'loc']:
    rec[c] = rec[c].astype(int)
rec['month'] = pd.to_numeric(rec['month'], errors='coerce').astype('Int64')
rec['year_end'] = pd.to_numeric(rec['year_end'], errors='coerce').astype('Int64')
rec['longitude'] = [locs[i][0] for i in rec['loc']]
rec['latitude'] = [locs[i][1] for i in rec['loc']]
nm = pd.read_csv(f'{RAW}/reaches_ming_names.tsv', sep='\t', header=None, dtype=str,
                 names=['record_id', 'province_zh', 'county_zh', 'source_zh', 'vol'])
nm['record_id'] = nm.record_id.astype(int)
rec = rec.merge(nm.drop(columns='vol'), on='record_id', how='left')
rec.loc[rec.county_zh.isin(['-9999', '']), 'county_zh'] = None
rec.loc[rec.province_zh.isin(['-9999', '']), 'province_zh'] = None

PROV = {'北京市': 'Beijing', '天津市': 'Tianjin', '河北省': 'Hebei', '山西省': 'Shanxi', '內蒙古自治區': 'Inner Mongolia',
        '内蒙古自治区': 'Inner Mongolia', '遼寧省': 'Liaoning', '吉林省': 'Jilin', '黑龍江省': 'Heilongjiang',
        '上海市': 'Shanghai', '江蘇省': 'Jiangsu', '浙江省': 'Zhejiang', '安徽省': 'Anhui', '福建省': 'Fujian',
        '江西省': 'Jiangxi', '山東省': 'Shandong', '河南省': 'Henan', '湖北省': 'Hubei', '湖南省': 'Hunan',
        '廣東省': 'Guangdong', '廣西壯族自治區': 'Guangxi', '海南省': 'Hainan', '四川省': 'Sichuan', '重慶市': 'Chongqing',
        '貴州省': 'Guizhou', '雲南省': 'Yunnan', '西藏自治區': 'Tibet', '陝西省': 'Shaanxi', '甘肅省': 'Gansu',
        '青海省': 'Qinghai', '寧夏回族自治區': 'Ningxia', '新疆維吾爾自治區': 'Xinjiang', '臺灣省': 'Taiwan', '台灣省': 'Taiwan',
        '香港': 'Hong Kong', '澳門': 'Macau'}
rec['province'] = rec.province_zh.map(PROV)
unmapped = rec.loc[rec.province.isna() & rec.province_zh.notna(), 'province_zh'].value_counts()
if len(unmapped):
    print('UNMAPPED provinces:', unmapped.to_dict())

# Ming reign eras (Gregorian approximation: a year belongs to the era in force for most of it)
ERAS = [(1368, 1398, 'Hongwu', '洪武'), (1399, 1402, 'Jianwen', '建文'), (1403, 1424, 'Yongle', '永樂'),
        (1425, 1425, 'Hongxi', '洪熙'), (1426, 1435, 'Xuande', '宣德'), (1436, 1449, 'Zhengtong', '正統'),
        (1450, 1456, 'Jingtai', '景泰'), (1457, 1464, 'Tianshun', '天順'), (1465, 1487, 'Chenghua', '成化'),
        (1488, 1505, 'Hongzhi', '弘治'), (1506, 1521, 'Zhengde', '正德'), (1522, 1566, 'Jiajing', '嘉靖'),
        (1567, 1572, 'Longqing', '隆慶'), (1573, 1620, 'Wanli', '萬曆'), (1621, 1627, 'Tianqi', '天啓'),
        (1628, 1644, 'Chongzhen', '崇禎')]
def era(y):
    for a, b, en, zh in ERAS:
        if a <= y <= b:
            return en, zh
    return None, None
rec['reign'], rec['reign_zh'] = zip(*rec.year.map(era))

# place ids: stable, sorted by province then lon/lat
places = rec.groupby('loc').agg(longitude=('longitude', 'first'), latitude=('latitude', 'first'),
                                province=('province', lambda s: s.mode().iat[0] if s.notna().any() else None),
                                province_zh=('province_zh', lambda s: s.mode().iat[0] if s.notna().any() else None),
                                county_zh=('county_zh', lambda s: s.dropna().mode().iat[0] if s.notna().any() else None),
                                n_records=('record_id', 'size'), first_year=('year', 'min'), last_year=('year', 'max')).reset_index()
places = places.sort_values(['province', 'county_zh', 'longitude', 'latitude'], na_position='last').reset_index(drop=True)
places['place_id'] = ['P%04d' % (i + 1) for i in range(len(places))]
pmap = dict(zip(places['loc'], places['place_id']))
rec['place_id'] = rec['loc'].map(pmap)

# ---------------------------------------------------------------- codebook
cb = pd.read_csv(f'{RAW}/codebook_tidy.tsv', sep='\t', dtype=str)
cb = cb.drop_duplicates(['main', 'sub', 'vocab'])
cb_vocab = {(r.main, r.sub, r.vocab): r.vocab_label for r in cb.itertuples()}
cb_sub = cb.drop_duplicates(['main', 'sub']).set_index(['main', 'sub']).sub_name.to_dict()
cb_main = cb.drop_duplicates('main').set_index('main').main_name.to_dict()

ev = rec.assign(event_code=rec.codes.str.split(';')).explode('event_code')
ev = ev[ev.event_code.str.fullmatch(r'\d{9}', na=False)].copy()
ev['main'] = ev.event_code.str[:2]; ev['subc'] = ev.event_code.str[2:4]
ev['vocab'] = ev.event_code.str[4:7]; ev['mag'] = ev.event_code.str[7].astype(int); ev['dur'] = ev.event_code.str[8].astype(int)
ev['domain'] = ev.main.map(lambda m: 'meteorology' if m < '30' else 'hazard' if m < '50' else 'other')
ev['main_category'] = ev.main.map(cb_main)
ev['subcategory'] = [cb_sub.get((a, b)) for a, b in zip(ev.main, ev.subc)]
ev['term_zh'] = [cb_vocab.get((a, b, c)) for a, b, c in zip(ev.main, ev.subc, ev.vocab)]
ev['term_zh'] = ev.term_zh.str.replace(r'\*+', '', regex=True).str.replace(r'（[^）]*）', '', regex=True).str.strip()
MAG = {0: 'uncertain', 1: 'light', 2: 'heavy', 3: 'uncertain, repeated', 4: 'light, repeated', 5: 'heavy, repeated',
       8: 'no occurrence', 9: 'not described'}
ev['magnitude'] = ev.mag.map(MAG)
# Severity: vocabulary 300-399 is the coding guide's 'degree-heavy' band (大旱, 大饑, 人相食...), or a heavy magnitude flag.
v = ev.vocab.astype(int)
ev['severe'] = ((v >= 300) & (v < 400)) | ev.mag.isin([2, 5])
ev['occurred'] = ev.mag != 8

def theme(m, s, vocab):
    if m == '30': return 'drought'
    if m == '31': return 'flood'
    if m == '32': return 'locusts_pests'
    if m == '33':
        return 'harvest_failure' if 100 <= int(vocab) < 500 else 'good_harvest' if 500 <= int(vocab) < 800 else 'crops_other'
    if m == '34': return 'epidemic'
    if m == '35': return 'cannibalism' if s == '02' else 'famine'
    if m == '11' and s in ('11', '12', '13', '14', '21', '22', '31', '32'): return 'cold_frost'
    if m == '71':
        return {'01': 'relief_tax_relief', '03': 'migration_refugees', '05': 'war_banditry', '09': 'impoverishment',
                '10': 'deaths', '11': 'price_spike', '12': 'sale_of_people', '16': 'abandoned_settlements'}.get(s, 'damage_other')
    return 'other'
ev['theme'] = [theme(a, b, c) for a, b, c in zip(ev.main, ev.subc, ev.vocab)]

events = ev[['record_id', 'event_code', 'year', 'month', 'year_end', 'reign', 'reign_zh', 'place_id', 'province', 'province_zh',
             'county_zh', 'longitude', 'latitude', 'domain', 'main_category', 'subcategory', 'theme', 'term_zh',
             'magnitude', 'severe', 'occurred', 'source_zh']].rename(columns={'source_zh': 'source_document_zh'})
events = events.sort_values(['year', 'record_id', 'event_code']).reset_index(drop=True)
events['month'] = events.month.astype('Int64'); events['year_end'] = events.year_end.astype('Int64')
events.to_parquet(f'{OUT}/events.parquet', index=False)

# ---------------------------------------------------------------- annual panel
THEMES = ['drought', 'flood', 'locusts_pests', 'cold_frost', 'harvest_failure', 'epidemic', 'famine', 'cannibalism',
          'war_banditry', 'price_spike', 'migration_refugees', 'sale_of_people', 'deaths', 'relief_tax_relief']
occ = events[events.occurred]
years = pd.DataFrame({'year': range(1368, 1645)})
years['reign'], years['reign_zh'] = zip(*years.year.map(era))
years['n_records'] = years.year.map(rec.groupby('year').size()).fillna(0).astype(int)
years['n_places_reporting'] = years.year.map(rec.groupby('year').place_id.nunique()).fillna(0).astype(int)
g = occ.groupby(['year', 'theme']).place_id.nunique().unstack(fill_value=0)
for t in THEMES:
    years[f'places_{t}'] = years.year.map(g[t] if t in g else pd.Series(dtype=int)).fillna(0).astype(int)
gs = occ[occ.severe].groupby(['year', 'theme']).place_id.nunique().unstack(fill_value=0)
for t in ['drought', 'flood', 'famine']:
    years[f'places_severe_{t}'] = years.year.map(gs[t] if t in gs else pd.Series(dtype=int)).fillna(0).astype(int)
den = years.n_places_reporting.replace(0, np.nan)
for t in ['drought', 'flood', 'famine', 'cannibalism', 'war_banditry', 'epidemic']:
    years[f'share_{t}'] = (years[f'places_{t}'] / den).round(4)

# --- ENSO events, Gergis & Fowler (2009)
clim = open(f'{RAW}/climate.txt', encoding='utf-8', errors='replace').read().split('@@FILE ')
gtxt = [p for p in clim if p.startswith('gergis2009')][0]
sec = re.split(r'\n\s*2\.\s+Gergis and Fowler 2009 La Ni', gtxt)
def parse_events(block):
    out = {}
    for m in re.finditer(r'^(\d{4})\s+\d+/\d+,\s+[\d.]+\s+[\d.]+\s+([A-Z]+)\s*$', block, re.M):
        out[int(m.group(1))] = m.group(2)
    return out
nino, nina = parse_events(sec[0]), parse_events(sec[1]) if len(sec) > 1 else {}
CLS = {'E': 'extreme', 'VS': 'very strong', 'S': 'strong', 'M': 'moderate', 'W': 'weak'}
def enso(y):
    if y < 1525: return None
    if y in nino and y in nina: return 'both phases listed'
    if y in nino: return 'El Nino (' + CLS.get(nino[y], nino[y]) + ')'
    if y in nina: return 'La Nina (' + CLS.get(nina[y], nina[y]) + ')'
    return 'neutral or not identified'
years['enso_event'] = years.year.map(enso)

# --- China temperature anomaly, Ge et al. (2013), decadal; each year gets its decade's value
ge = [p for p in clim if p.startswith('ge2013')][0]
gl = [l.split('\t') for l in ge.splitlines() if re.match(r'^\d+\t', l)]
ged = {int(r[0]): float(r[1]) for r in gl}
def ge_temp(y):
    k = (y // 10) * 10 + 5
    return ged.get(k)
years['china_temp_anomaly_c'] = years.year.map(ge_temp)
years.to_parquet(f'{OUT}/annual.parquet', index=False)

# ---------------------------------------------------------------- places
pl = places.drop(columns='loc')
pl = pl[['place_id', 'province', 'province_zh', 'county_zh', 'longitude', 'latitude', 'n_records', 'first_year', 'last_year']]
pl.to_parquet(f'{OUT}/places.parquet', index=False)

# small CSV mirrors for convenience
years.to_csv(f'{OUT}/annual.csv', index=False)

stats = dict(records=len(rec), events=len(events), events_occurred=int(events.occurred.sum()), places=len(pl),
             provinces=int(events.province.nunique()), sources=int(rec.source_zh.nunique()),
             cannibalism_places_1368_1644=int(occ[occ.theme == 'cannibalism'].place_id.nunique()),
             nino_years_1525_1644=sum(1 for y in nino if y <= 1644 and y not in nina), nina_years_1525_1644=sum(1 for y in nina if y <= 1644 and y not in nino), both_1525_1644=sum(1 for y in nino if y<=1644 and y in nina))
json.dump(stats, open('out/stats.json', 'w'), indent=1)
print(stats)
print(events.theme.value_counts().to_string())

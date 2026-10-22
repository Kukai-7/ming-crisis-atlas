import json, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Polygon as MPoly, Rectangle
from matplotlib.collections import PatchCollection
from shapely.geometry import shape, box
import os

os.makedirs('out/assets', exist_ok=True)
import glob
for f in glob.glob('fonts/*.otf') + ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                                      '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
    if os.path.exists(f): fm.fontManager.addfont(f)
_names = {f.name for f in fm.fontManager.ttflist}
FONT = next((n for n in ('Noto Sans CJK SC', 'Noto Sans CJK JP') if n in _names), 'DejaVu Sans')
plt.rcParams.update({'font.family': FONT, 'font.size': 10, 'axes.edgecolor': '#BBBBBB',
                     'axes.linewidth': 0.6, 'xtick.color': '#666666', 'ytick.color': '#666666',
                     'axes.labelcolor': '#444444', 'text.color': '#1F1F1F', 'savefig.dpi': 200})

INK, MID, LIGHT, LAND, PAPER = '#1F1F1F', '#8A8A8A', '#C9C6BF', '#ECE9E3', '#FFFFFF'
RED, RED_SOFT = '#B33A2B', '#D9A097'
RIVER = '#A9B4BC'

ev = pd.read_parquet('out/data/events.parquet')
an = pd.read_parquet('out/data/annual.parquet')
pl = pd.read_parquet('out/data/places.parquet')
occ = ev[ev.occurred]

# ---------------------------------------------------------------- basemap
EXT = (96.5, 125.5, 17.5, 44.5)
clip = box(EXT[0] - 2, EXT[2] - 2, EXT[1] + 2, EXT[3] + 2)
adm0 = json.load(open('raw/ne_admin0.geojson'))
adm1 = json.load(open('raw/ne_admin1.geojson'))
riv = json.load(open('raw/ne_rivers.geojson'))

def polys(geom):
    g = geom.intersection(clip)
    if g.is_empty: return []
    gs = getattr(g, 'geoms', [g])
    return [np.array(p.exterior.coords) for p in gs if p.geom_type == 'Polygon']

china_iso = {'CHN', 'TWN'}
neighbours = [shape(f['geometry']) for f in adm0['features'] if f['properties'].get('ADM0_A3') not in china_iso]
china = [shape(f['geometry']) for f in adm0['features'] if f['properties'].get('ADM0_A3') in china_iso]
provs = [shape(f['geometry']) for f in adm1['features'] if f['properties'].get('adm0_a3') in china_iso]
rivers = [shape(f['geometry']) for f in riv['features']
          if f['properties'].get('name') in ('Yangtze', 'Huang', 'Yellow', 'Chang Jiang', 'Huang He', 'Xi', 'Huai')]

def basemap(ax, labels=True):
    ax.set_facecolor(PAPER)
    for g in neighbours:
        for a in polys(g): ax.add_patch(MPoly(a, closed=True, fc='#F6F5F2', ec='none'))
    for g in provs:
        for a in polys(g): ax.add_patch(MPoly(a, closed=True, fc=LAND, ec='#FFFFFF', lw=0.5))
    for g in china:
        for a in polys(g): ax.add_patch(MPoly(a, closed=True, fc='none', ec=LIGHT, lw=0.6))
    for r in rivers:
        g = r.intersection(clip)
        for l in getattr(g, 'geoms', [g]):
            if l.geom_type == 'LineString':
                x, y = l.xy; ax.plot(x, y, color=RIVER, lw=0.7, zorder=2)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_aspect(1.15)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

# ---------------------------------------------------------------- 1. hero map, 1637-1644
win = occ[occ.year.between(1637, 1644)]
fam = win[win.theme == 'famine'].place_id.unique()
can = win[win.theme == 'cannibalism'].place_id.unique()
rep = win.place_id.unique()
P = pl.set_index('place_id')

fig, ax = plt.subplots(figsize=(8.2, 8.2))
basemap(ax)
p0 = P.loc[np.setdiff1d(rep, np.union1d(fam, can))]
ax.scatter(p0.longitude, p0.latitude, s=4, c=MID, alpha=.45, lw=0, zorder=3, label='Any other record')
p1 = P.loc[np.setdiff1d(fam, can)]
ax.scatter(p1.longitude, p1.latitude, s=11, c=RED_SOFT, lw=0, zorder=4, label='Famine')
p2 = P.loc[can]
ax.scatter(p2.longitude, p2.latitude, s=15, c=RED, lw=0, zorder=5, label='"People ate one another" (人相食)')
bj = (116.40, 39.90)
ax.scatter(*bj, s=46, marker='s', facecolor='none', edgecolor=INK, lw=1, zorder=6)
ax.annotate('Beijing\nfalls, April 1644', bj, xytext=(119.6, 42.2), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle='-', color=INK, lw=.6))
fig.text(.06, .955, 'The last famine of the Ming, 1637 to 1644', fontsize=17, weight='bold', color=INK)
fig.text(.06, .918, f'Places whose local records report famine or cannibalism. {len(can)} of {len(rep)} reporting places record '
         'people eating one another.', fontsize=9.5, color='#555555')
leg = ax.legend(loc='lower right', frameon=False, fontsize=8.5, markerscale=1.6, handletextpad=.4)
fig.text(.06, .03, 'Data: REACHES Chinese Historical Climate Database (Wang et al., NOAA WDS Paleo), Ming subset. '
         'Modern boundaries for orientation only.', fontsize=7.2, color=MID)
plt.subplots_adjust(left=.04, right=.98, top=.9, bottom=.06)
fig.savefig('out/assets/map_1637_1644.png', facecolor=PAPER)
plt.close(fig)

# ---------------------------------------------------------------- 2. timeline 1368-1644
fig, ax = plt.subplots(figsize=(12, 4.6))
ERAS = an.groupby('reign', sort=False).year.agg(['min', 'max']).reset_index().sort_values('min')
for i, r in enumerate(ERAS.itertuples()):
    if i % 2 == 0:
        ax.axvspan(r.min - .5, r.max + .5, color='#F3F2EF', lw=0, zorder=0)
    if r.max - r.min >= 13 or r.reign in ('Chongzhen',):
        ax.text((r.min + r.max) / 2, -0.10, r.reign, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=7.6, color=MID)
ax.bar(an.year, an.places_famine, width=1, color=RED_SOFT, lw=0, label='Places reporting famine', zorder=2)
ax.bar(an.year, an.places_cannibalism, width=1, color=RED, lw=0, label='Places reporting cannibalism', zorder=3)
ax.step(an.year, an.n_places_reporting, where='mid', color=MID, lw=.7, alpha=.8,
        label='Places with any record (how much was written down)', zorder=1)
ax.set_xlim(1367.5, 1644.5); ax.set_ylim(0, 620)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Places (counties) per year')
ax.set_xticks(range(1370, 1650, 30))
ax.tick_params(axis='x', labelsize=8.5)
el = an[an.enso_event.str.startswith('El Nino', na=False)]
ax.scatter(el.year, np.full(len(el), 600), marker='|', s=40, c=INK, lw=.8, zorder=4)
ax.text(1522, 600, 'El Niño years (Gergis and Fowler, from 1525)', fontsize=7.6, color=INK, ha='right', va='center')
for y, lab in [(1586, '1586 to 1589\nWanli famines'), (1640, '1640')]:
    v = an.set_index('year').loc[y, 'places_famine']
    ax.annotate(lab, (y, v), xytext=(y - 34 if y == 1640 else y - 22, v + 120 if y == 1640 else v + 150), fontsize=8,
                color=INK, ha='center', arrowprops=dict(arrowstyle='-', color=INK, lw=.5))
ax.legend(loc='upper left', frameon=False, fontsize=8.2, bbox_to_anchor=(0, .93))
fig.text(.055, .95, 'Two and a half centuries of local disaster records', fontsize=15, weight='bold')
fig.text(.055, .905, 'Counts follow the survival of sources as much as the weather: read peaks against the grey line.',
         fontsize=9, color='#555555')
plt.subplots_adjust(left=.055, right=.985, top=.85, bottom=.15)
fig.savefig('out/assets/timeline_1368_1644.png', facecolor=PAPER)
plt.close(fig)

# ---------------------------------------------------------------- 3. province x decade heatmap of famine
occ2 = occ.assign(decade=(occ.year // 10) * 10)
den = occ2.groupby(['province', 'decade']).apply(lambda d: d[['place_id', 'year']].drop_duplicates().shape[0])
num = occ2[occ2.theme.isin(['famine', 'cannibalism'])].groupby(['province', 'decade']).apply(
    lambda d: d[['place_id', 'year']].drop_duplicates().shape[0])
share = (num / den).unstack().fillna(0)
mask = den.unstack().fillna(0) < 5
share = share.where(~mask)
lat = pl.groupby('province').latitude.median()
keep = pl.province.value_counts()
keep = keep[keep >= 20].index
order = lat.loc[keep].sort_values(ascending=False).index
share = share.loc[order, [c for c in share.columns if 1360 <= c <= 1640]]
fig, ax = plt.subplots(figsize=(12, 5.4))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('ink', ['#F4F2EE', RED_SOFT, RED, '#5A1C14'])
cmap.set_bad('#FFFFFF')
im = ax.imshow(share.values, aspect='auto', cmap=cmap, vmin=0, vmax=.45, interpolation='nearest')
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8.5)
ax.set_xticks(range(len(share.columns))[::3]); ax.set_xticklabels([f'{c}s' for c in share.columns][::3], fontsize=8.2)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=.018, pad=.01)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, length=0)
cb.set_label('Share of place-years with famine', fontsize=8, labelpad=6)
fig.text(.055, .95, 'Where hunger was written down, by decade', fontsize=15, weight='bold')
fig.text(.055, .905, 'Provinces ordered north to south. Blank cells: fewer than 5 place-years recorded.', fontsize=9, color='#555555')
plt.subplots_adjust(left=.1, right=.93, top=.86, bottom=.08)
fig.savefig('out/assets/famine_by_province_decade.png', facecolor=PAPER)
plt.close(fig)
print('ok')

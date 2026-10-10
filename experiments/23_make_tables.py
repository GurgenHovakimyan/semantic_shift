"""
Experiment 23 -- LaTeX tables for the revised manuscript.

Reads the CSV outputs of experiments 16-21 and writes complete ``tabular``
blocks to ``results/tables/*.tex``.  Every table is laid out to fit the
31 pc (372 pt) text width of the Springer ``sn-jnl`` template in portrait
orientation at the class's default 8 bp table font: the power, correlation
and stream tables are transposed (detectors as rows), the calibration table
puts the two statistics side by side, and the delay table uses group rows
instead of wide identifier columns.  The correlation table uses the macro
``\\cifont`` (defined in the manuscript preamble) for the confidence-interval
lines.  ``24_fill_manuscript.py`` pastes the blocks into the single-file
manuscript.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from experiments.common import RESULTS_DIR
from src.statistics import PRETTY

TAB_DIR = RESULTS_DIR / 'tables'
TAB_DIR.mkdir(exist_ok=True)

SHORT_KEY = {'imdb_distilbert': 'IMDb / DistilBERT', 'imdb_roberta': 'IMDb / RoBERTa',
             'amazon_distilbert': 'Amazon / DistilBERT', 'amazon_roberta': 'Amazon / RoBERTa'}
CAL_GROUPS = [
    ('imdb_distilbert', 'IMDb / DistilBERT (disjoint pools of 15,000)'),
    ('imdb_roberta', 'IMDb / RoBERTa (disjoint pools of 15,000)'),
    ('amazon_distilbert', 'Amazon / DistilBERT, shared pool (1,500)'),
    ('amazon_distilbert_sep', 'Amazon / DistilBERT, disjoint pools (750 + 750)'),
    ('amazon_roberta', 'Amazon / RoBERTa, shared pool (1,500)'),
    ('amazon_roberta_sep', 'Amazon / RoBERTa, disjoint pools (750 + 750)'),
]
VARIANT_NAME = {
    'mean_multi_fixed_U (default)': 'mean / 5 scales / fixed / U (default)',
    'median_multi_fixed_U': 'median / 5 scales / fixed / U',
    'mean_single_fixed_U': 'mean / 1 scale / fixed / U',
    'median_single_fixed_U': 'median / 1 scale / fixed / U',
    'mean_multi_adaptive_U': 'mean / 5 scales / per-window / U',
    'mean_multi_fixed_V': 'mean / 5 scales / fixed / V (submitted version)',
}
SC_SHORT = {'natural': 'Natural', 'attenuation': 'Attenuation', 'boundary': 'Boundary',
            'asym_temporal': 'Asym.\\ temporal', 'label_noise': 'Label noise', 'label_shift': 'Label shift'}
SC_HEAD = dict(SC_SHORT, asym_temporal='\\makecell{Asym.\\\\temporal}')
SC_ORDER = ['natural', 'attenuation', 'boundary', 'asym_temporal', 'label_noise', 'label_shift']
LAM_SEL = {'natural': [1.0], 'attenuation': [0.5, 1.0], 'boundary': [0.5, 1.0], 'asym_temporal': [0.5, 1.0],
           'label_noise': [0.2, 0.5], 'label_shift': [0.5, 1.0]}
UNSUP = ['global_mmd', 'energy', 'c2st_linear', 'c2st_mlp', 'bbsd_ks']
LABEL_AWARE = ['cmmd', 'cond_energy', 'cond_c2st', 'centroid_shift', 'joint_mmd', 'error_rate']
DET_ORDER = UNSUP + LABEL_AWARE
SHORT = {'global_mmd': 'G-MMD', 'energy': 'Energy', 'c2st_linear': 'C2ST-lin', 'c2st_mlp': 'C2ST-MLP', 'bbsd_ks': 'BBSD',
         'cmmd': '\\textbf{C-MMD}', 'cond_energy': 'C-Energy', 'cond_c2st': 'C-C2ST', 'centroid_shift': 'Centroid',
         'joint_mmd': 'Joint', 'error_rate': 'Error', 'prior_shift': 'Prior'}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write(name, text):
    (TAB_DIR / name).write_text(text)
    print(f'  wrote {name}')


def fmt(x, nd=2):
    return '--' if pd.isna(x) else f'{x:.{nd}f}'


def sci(x):
    if pd.isna(x):
        return '--'
    return f'{x:.2e}'.replace('e-0', 'e-').replace('e+0', 'e')


def row(cells, end=' \\\\'):
    return ' & '.join(cells) + end


def group(label, ncol, extra=''):
    return f'\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{label}}}}}{extra} \\\\'


def tabular(colspec, header, body, tabcolsep='2.5pt', stretch='1.1'):
    """Complete tabular block; tabcolsep/arraystretch are local to the enclosing table."""
    lines = [f'\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\\renewcommand{{\\arraystretch}}{{{stretch}}}%',
             f'\\begin{{tabular}}{{{colspec}}}', '\\toprule'] + header + ['\\midrule'] + body + ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


def label_row(det):
    return PRETTY[det] if det != 'cmmd' else '\\textbf{C-MMD (ours)}'


# ---------------------------------------------------------------------------
# calibration (Table: thresholds, FPR, permutation type-I)
# ---------------------------------------------------------------------------

def table_calibration():
    thr = pd.read_csv(RESULTS_DIR / 'calibration_thresholds.csv')
    fpr = pd.read_csv(RESULTS_DIR / 'calibration_fpr.csv')
    perm = pd.read_csv(RESULTS_DIR / 'permutation_type1.csv')
    stats = [('cmmd', 'C-MMD'), ('global_mmd', 'Global MMD')]
    sub_head = ['s.d.', '$\\tau_{0.05}$', 'FPR$_{0.05}$', '$k{=}2$', 'perm.']
    header = [row([''] + [f'\\multicolumn{{5}}{{c}}{{{name}}}' for _, name in stats]),
              '\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}',
              row(['$N$'] + sub_head + sub_head)]
    body = []
    for key, label in CAL_GROUPS:
        sub = thr[(thr['key'] == key) & np.isclose(thr['alpha'], 0.05)]
        if sub.empty:
            continue
        if body:
            body.append('\\addlinespace[3pt]')
        body.append(group(label, 11))
        for n in sorted(sub['N'].unique()):
            cells = [str(n)]
            for det, _ in stats:
                t = sub[(sub['N'] == n) & (sub['detector'] == det)].iloc[0]
                f = fpr[(fpr['key'] == key) & (fpr['N'] == n) & (fpr['detector'] == det) & np.isclose(fpr['alpha'], 0.05)].iloc[0]
                p = perm[(perm['key'] == key) & (perm['N'] == n) & (perm['detector'] == det)]
                cells += [f'{t["null_std"]:.1e}', f'{t["tau"]:.1e}',
                          f'{f["fpr_mean"]:.3f}$\\pm${f["fpr_std"]:.3f}', f'{f["fpr_k2_mean"]:.3f}',
                          fmt(p['reject_05'].iloc[0], 2) if not p.empty else '--']
            body.append(row(cells))
    write('tab_calibration.tex', tabular('c' + 'c' * 10, header, body, tabcolsep='2pt'))


def table_fpr_all():
    fpr = pd.read_csv(RESULTS_DIR / 'calibration_fpr.csv')
    cols = [('imdb_distilbert', 200), ('imdb_distilbert', 500), ('imdb_roberta', 200), ('amazon_distilbert', 200)]
    header = [row(['Detector'] + [f'\\makecell{{{SHORT_KEY[k].replace(" / ", "\\\\")}\\\\$N{{=}}{n}$}}' for k, n in cols])]
    body = []
    for det in DET_ORDER + ['prior_shift']:
        cells = [label_row(det)]
        for k, n in cols:
            f = fpr[(fpr['key'] == k) & (fpr['N'] == n) & (fpr['detector'] == det) & np.isclose(fpr['alpha'], 0.05)]
            cells.append(f'{f["fpr_mean"].iloc[0]:.3f}$\\pm${f["fpr_std"].iloc[0]:.3f}' if not f.empty else '--')
        body.append(row(cells))
    write('tab_fpr_all.tex', tabular('lcccc', header, body, tabcolsep='4pt'))


# ---------------------------------------------------------------------------
# power tables (transposed: detectors as rows, scenario x level as columns)
# ---------------------------------------------------------------------------

def _power_columns(s):
    cols = []
    for sc in SC_ORDER:
        g = s[s['scenario'] == sc]
        if g.empty:
            continue
        for lam in LAM_SEL[sc]:
            if (np.isclose(g['lam'], lam)).any():
                cols.append((sc, lam))
    return cols


def _power_block(s, cols, with_groups=True):
    body = []
    best = {}
    for sc, lam in cols:
        gg = s[(s['scenario'] == sc) & np.isclose(s['lam'], lam)].set_index('detector')
        vals = [gg.loc[d, 'power_mean'] for d in DET_ORDER if d != 'error_rate' and d in gg.index]
        best[(sc, lam)] = max(vals) if vals else np.nan
    for grp, dets in [('Unsupervised', UNSUP), ('Label-aware', LABEL_AWARE)]:
        if with_groups:
            body.append(group(grp, len(cols) + 1))
        for d in dets:
            cells = [label_row(d)]
            for sc, lam in cols:
                gg = s[(s['scenario'] == sc) & np.isclose(s['lam'], lam)].set_index('detector')
                v = gg.loc[d, 'power_mean'] if d in gg.index else np.nan
                cell = fmt(v)
                if d != 'error_rate' and sc != 'label_shift' and not pd.isna(v) and v >= best[(sc, lam)] - 1e-9:
                    cell = f'\\textbf{{{cell}}}'
                cells.append(cell)
            body.append(row(cells))
    body.append('\\midrule')
    f1 = ['Downstream macro-F1']
    for sc, lam in cols:
        gg = s[(s['scenario'] == sc) & np.isclose(s['lam'], lam)]
        f1.append(f'{gg["f1_mean"].iloc[0]:.3f}')
    body.append(row(f1))
    return body


def _power_header(cols):
    top, mid, cmid, i = ['Detector'], ['$\\lambda$'], [], 2
    for sc in SC_ORDER:
        k = sum(1 for c in cols if c[0] == sc)
        if k == 0:
            continue
        top.append(f'\\multicolumn{{{k}}}{{c}}{{{SC_HEAD[sc]}}}')
        cmid.append(f'\\cmidrule(lr){{{i}-{i + k - 1}}}')
        for c in cols:
            if c[0] == sc:
                mid.append('--' if sc == 'natural' else f'{c[1]:g}')
        i += k
    return [row(top), ''.join(cmid), row(mid)]


def table_power(keys, n, name):
    s_all = pd.read_csv(RESULTS_DIR / 'benchmark_summary.csv')
    s_all = s_all[(s_all['N'] == n) & (s_all['key'].isin(keys))]
    if s_all.empty:
        return
    cols = _power_columns(s_all[s_all['key'] == keys[0]])
    header = _power_header(cols)
    body = []
    for key in keys:
        s = s_all[s_all['key'] == key]
        if s.empty:
            continue
        if len(keys) > 1:
            if body:
                body += ['\\midrule']
            body.append(group(SHORT_KEY[key], len(cols) + 1))
        body += _power_block(s, cols)
    write(name, tabular('l' + 'c' * len(cols), header, body, tabcolsep='2pt'))


def table_auroc(key, n, name):
    s = pd.read_csv(RESULTS_DIR / 'benchmark_summary.csv')
    s = s[(s['key'] == key) & (s['N'] == n)]
    if s.empty:
        return
    cols = [(sc, (0.5 if sc == 'label_noise' else s[s['scenario'] == sc]['lam'].max())) for sc in SC_ORDER
            if not s[s['scenario'] == sc].empty]
    header = [row(['Detector'] + [SC_SHORT[sc] for sc, _ in cols]), row(['$\\lambda$'] + ['--' if sc == 'natural' else f'{lam:g}' for sc, lam in cols])]
    body = []
    for d in DET_ORDER:
        cells = [label_row(d)]
        for sc, lam in cols:
            gg = s[(s['scenario'] == sc) & np.isclose(s['lam'], lam)].set_index('detector')
            cells.append(fmt(gg.loc[d, 'auroc']) if d in gg.index else '--')
        body.append(row(cells))
    write(name, tabular('l' + 'c' * len(cols), header, body))


# ---------------------------------------------------------------------------
# detection delay
# ---------------------------------------------------------------------------

def table_delay():
    d = pd.read_csv(RESULTS_DIR / 'delay_summary.csv')
    dets = ['global_mmd', 'bbsd_ks', 'c2st_mlp', 'cmmd', 'cond_c2st', 'centroid_shift', 'joint_mmd', 'error_rate']
    head = dict(SHORT, c2st_mlp='\\makecell{C2ST\\\\MLP}')
    header = [row(['Drift', 'Onset'] + [head[x] for x in dets])]
    body = []
    for key, n in [('imdb_distilbert', 200), ('imdb_distilbert', 500), ('imdb_roberta', 200), ('amazon_distilbert', 200)]:
        sub = d[(d['key'] == key) & (d['N'] == n) & (d['rule'] == 'k2')]
        if sub.empty:
            continue
        if body:
            body.append('\\addlinespace[3pt]')
        body.append(group(f'{SHORT_KEY[key]}, $N={n}$', len(dets) + 2))
        for drift in ['asym_temporal', 'boundary', 'attenuation', 'label_noise']:
            for onset in ['abrupt', 'gradual']:
                g = sub[(sub['drift'] == drift) & (sub['onset_type'] == onset)]
                if g.empty:
                    continue
                cells = [SC_SHORT[drift] if onset == 'abrupt' else '', onset]
                for det in dets:
                    v = g[g['detector'] == det]['delay'].values
                    rate = np.mean(np.isfinite(v))
                    med = np.median(v[np.isfinite(v)]) if rate > 0 else np.nan
                    cells.append(f'{med:.0f} ({rate:.1f})' if rate > 0 else '--')
                body.append(row(cells))
    write('tab_delay.tex', tabular('ll' + 'c' * len(dets), header, body, tabcolsep='2pt'))


# ---------------------------------------------------------------------------
# chronological streams
# ---------------------------------------------------------------------------

def table_streams():
    s = pd.read_csv(RESULTS_DIR / 'stream_summary.csv')
    dets = ['global_mmd', 'bbsd_ks', 'c2st_mlp', 'cmmd', 'cond_c2st', 'centroid_shift', 'joint_mmd', 'error_rate']
    keys = [k for k in SHORT_KEY if not s[s['key'] == k].empty]
    subs = {k: s[s['key'] == k].set_index('detector') for k in keys}
    header = [row(['Detector'] + [f'\\makecell{{{SHORT_KEY[k].replace(" / ", "\\\\")}\\\\$N{{=}}{int(subs[k]["N"].iloc[0])}$'
                                  f'\\\\{int(subs[k]["n_windows"].iloc[0])} windows}}' for k in keys])]
    body = [group('Fraction of windows exceeding $\\tau_{0.05}$', len(keys) + 1)]
    for det in dets:
        body.append(row([label_row(det)] + [fmt(subs[k].loc[det, 'frac_exceed']) for k in keys]))
    body += ['\\midrule', group('Fraction flagged after Benjamini--Hochberg at FDR 5\\%', len(keys) + 1)]
    for det in ['global_mmd', 'cmmd']:
        body.append(row([label_row(det)] + [fmt(subs[k].loc[det, 'frac_bh']) for k in keys]))
    write('tab_streams.tex', tabular('l' + 'c' * len(keys), header, body, tabcolsep='5pt'))


# ---------------------------------------------------------------------------
# downstream correlations (transposed, two-line cells)
# ---------------------------------------------------------------------------

def table_correlations():
    c = pd.read_csv(RESULTS_DIR / 'downstream_correlations.csv')
    dets = ['global_mmd', 'bbsd_ks', 'c2st_mlp', 'cmmd', 'cond_c2st', 'centroid_shift', 'joint_mmd', 'error_rate']
    keys = ['imdb_distilbert', 'imdb_roberta', 'amazon_distilbert']
    settings = [('natural_stream', 'stream'), ('benchmark_pooled', 'pooled')]
    subs = {(k, st): c[(c['key'] == k) & (c['setting'] == st)].set_index('detector') for k in keys for st, _ in settings}
    header = [row([''] + [f'\\multicolumn{{2}}{{c}}{{{SHORT_KEY[k]}}}' for k in keys]),
              ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(keys))),
              row(['Detector'] + [lbl for _ in keys for _, lbl in settings])]
    body = []
    for d in dets:
        cells = [label_row(d)]
        for k in keys:
            for st, _ in settings:
                sub = subs[(k, st)]
                if sub.empty or d not in sub.index:
                    cells.append('--')
                    continue
                r = sub.loc[d]
                cells.append(f'\\makecell{{{r["rho"]:+.2f}\\\\{{\\cifont[{r["ci_lo"]:+.2f}, {r["ci_hi"]:+.2f}]}}}}')
        body.append(row(cells))
    body.append('\\midrule')
    body.append(row(['windows $n$'] + [str(int(subs[(k, st)]['n'].iloc[0])) if not subs[(k, st)].empty else '--' for k in keys for st, _ in settings]))
    body.append(row(['F1 range'] + [f'[{subs[(k, st)]["f1_min"].iloc[0]:.2f}, {subs[(k, st)]["f1_max"].iloc[0]:.2f}]' if not subs[(k, st)].empty else '--'
                                    for k in keys for st, _ in settings]))
    write('tab_correlations.tex', tabular('l' + 'c' * (2 * len(keys)), header, body, tabcolsep='2.5pt', stretch='1.0'))


# ---------------------------------------------------------------------------
# encoder scale check
# ---------------------------------------------------------------------------

def table_encoder():
    e = pd.read_csv(RESULTS_DIR / 'encoder_scale_check.csv')
    rows = {
        'beta': 'Bandwidth $\\beta$ (mean sq.\\ pairwise distance)',
        'anisotropy_cos': 'Mean cosine similarity of random pairs',
        'cmmd_null_std_N200': 'C-MMD null s.d.\\ ($N{=}200$)',
        'cmmd_tau05_N200': 'C-MMD $\\tau_{0.05}$ ($N{=}200$)',
        'cmmd_swap50_mean_N200': 'C-MMD at 50\\% label noise ($N{=}200$)',
        'cmmd_swap50_effect_N200': 'Label-noise effect size (null s.d.\\ units)',
        'cmmd_natural_mean_N200': 'C-MMD, natural stream windows ($N{=}200$)',
        'global_natural_mean_N200': 'Global MMD, natural stream windows ($N{=}200$)',
    }
    header = [row(['', '\\multicolumn{3}{c|}{IMDb}', '\\multicolumn{3}{c}{Amazon}']),
              row(['Quantity', 'DistilBERT', 'RoBERTa', 'ratio', 'DistilBERT', 'RoBERTa', 'ratio'])]
    body = []
    for q, lbl in rows.items():
        cells = [lbl]
        for ds in ['imdb', 'amazon']:
            r = e[(e['dataset'] == ds) & (e['quantity'] == q)]
            if r.empty:
                cells += ['--', '--', '--']
            else:
                r = r.iloc[0]
                cells += [sci(r['distilbert']) if abs(r['distilbert']) < 0.01 else fmt(r['distilbert'], 3),
                          sci(r['roberta']) if abs(r['roberta']) < 0.01 else fmt(r['roberta'], 3), fmt(r['ratio_distil_over_roberta'], 2)]
        body.append(row(cells))
    write('tab_encoder.tex', tabular('>{\\raggedright\\arraybackslash}p{118pt}ccc|ccc', header, body, tabcolsep='3pt'))


# ---------------------------------------------------------------------------
# bandwidth ablation
# ---------------------------------------------------------------------------

def table_bandwidth():
    a = pd.read_csv(RESULTS_DIR / 'bandwidth_ablation.csv')
    scs = ['label_noise', 'attenuation', 'asym_temporal', 'boundary', 'label_shift']
    names = ['noise', 'atten.', 'asym.', 'bound.', 'shift']
    header = [row(['', '\\multicolumn{5}{c|}{$N=200$}', '\\multicolumn{5}{c}{$N=500$}']),
              row(['Variant'] + names + names)]
    body = []
    for variant in a['variant'].unique():
        cells = [VARIANT_NAME.get(variant, variant.replace('_', '\\_'))]
        for n in [200, 500]:
            for sc in scs:
                r = a[(a['variant'] == variant) & (a['N'] == n) & (a['scenario'] == sc)]
                cells.append(fmt(r['power'].iloc[0]) if not r.empty else '--')
        body.append(row(cells))
    write('tab_bandwidth.tex', tabular('>{\\raggedright\\arraybackslash}p{108pt}|ccccc|ccccc', header, body, tabcolsep='2pt'))


# ---------------------------------------------------------------------------
# speed
# ---------------------------------------------------------------------------

def table_speed():
    s = pd.read_csv(RESULTS_DIR / 'speed_benchmark_v2.csv')
    ns = [100, 200, 500, 1000, 2000]
    header = [row(['Detector'] + [f'$N{{=}}{n}$' for n in ns] + ['GPU, $N{=}500$'])]
    body = []
    for det in ['cmmd', 'global_mmd', 'centroid_shift', 'joint_mmd', 'energy', 'cond_energy', 'bbsd_ks', 'c2st_linear', 'cond_c2st', 'c2st_mlp', 'error_rate']:
        cells = [label_row(det)]
        for n in ns:
            r = s[(s['detector'] == det) & (s['N'] == n) & (s['device'] == 'cpu')]
            cells.append(f'{r["ms_median"].iloc[0]:.1f}' if not r.empty else '--')
        r = s[(s['detector'] == det) & (s['device'] == 'cuda') & (s['N'] == 500)]
        cells.append(f'{r["ms_median"].iloc[0]:.1f}' if not r.empty else '--')
        body.append(row(cells))
    body.append('\\midrule')
    for det, lbl in [('cmmd_permutation200', 'C-MMD, 200 permutations'), ('distilbert_embedding', 'DistilBERT embedding')]:
        cells = [lbl]
        for n in ns:
            r = s[(s['detector'] == det) & (s['N'] == n) & (s['device'] == 'cpu')]
            cells.append(f'{r["ms_median"].iloc[0]:.0f}' if not r.empty else '--')
        r = s[(s['detector'] == det) & (s['device'] == 'cuda') & (s['N'] == 500)]
        cells.append(f'{r["ms_median"].iloc[0]:.0f}' if not r.empty else '--')
        body.append(row(cells))
    write('tab_speed.tex', tabular('lccccc|c', header, body, tabcolsep='3pt'))


if __name__ == '__main__':
    for fn in [table_calibration, table_fpr_all,
               lambda: table_power(['imdb_distilbert'], 500, 'tab_power_imdb_N500.tex'),
               lambda: table_power(['imdb_distilbert'], 200, 'tab_power_imdb_N200.tex'),
               lambda: table_power(['imdb_distilbert'], 1000, 'tab_power_imdb_N1000.tex'),
               lambda: table_power(['imdb_roberta'], 200, 'tab_power_imdb_roberta_N200.tex'),
               lambda: table_power(['imdb_roberta'], 500, 'tab_power_imdb_roberta_N500.tex'),
               lambda: table_power(['amazon_distilbert', 'amazon_roberta'], 200, 'tab_power_amazon_N200.tex'),
               lambda: table_auroc('imdb_distilbert', 500, 'tab_auroc_imdb_N500.tex'),
               table_delay, table_streams, table_correlations, table_encoder, table_bandwidth, table_speed]:
        try:
            fn()
        except Exception as exc:
            print(f'  skipped: {type(exc).__name__}: {exc}')

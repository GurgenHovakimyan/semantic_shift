"""
Experiment 24 -- Inject the generated LaTeX tables into the manuscript.

The Springer template requires a single .tex file (no \\input), so the
complete ``tabular`` blocks written by ``23_make_tables.py`` are pasted
between marker comments inside the corresponding ``table`` environment:

    %%<TAB_NAME
    \\begin{tabular}...\\end{tabular}
    %%TAB_NAME>

A bare placeholder ``%%TAB_NAME%%`` is expanded on the first run; subsequent
runs replace the block between the markers, so the operation is idempotent.
Any hand-written ``\\begin{tabular} ... \\midrule`` shell left immediately
before a marker (and ``\\bottomrule \\end{tabular}`` after it) is removed, so
that older manuscripts whose markers enclosed only the body rows migrate
automatically.  All tables are kept in portrait orientation with placement
``[!htbp]``; ``sidewaystable`` environments from earlier versions are
converted back.

Usage:  python experiments/24_fill_manuscript.py "<path to sn-article.tex>"
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS_DIR

TAB_DIR = RESULTS_DIR / 'tables'

MAPPING = {
    'TAB_CALIBRATION': 'tab_calibration.tex',
    'TAB_FPR_ALL': 'tab_fpr_all.tex',
    'TAB_POWER_IMDB_N500': 'tab_power_imdb_N500.tex',
    'TAB_POWER_IMDB_N200': 'tab_power_imdb_N200.tex',
    'TAB_POWER_IMDB_N1000': 'tab_power_imdb_N1000.tex',
    'TAB_POWER_IMDB_ROBERTA_N200': 'tab_power_imdb_roberta_N200.tex',
    'TAB_POWER_IMDB_ROBERTA_N500': 'tab_power_imdb_roberta_N500.tex',
    'TAB_POWER_AMAZON_N200': 'tab_power_amazon_N200.tex',
    'TAB_DELAY': 'tab_delay.tex',
    'TAB_STREAMS': 'tab_streams.tex',
    'TAB_CORR': 'tab_correlations.tex',
    'TAB_ENCODER': 'tab_encoder.tex',
    'TAB_BANDWIDTH': 'tab_bandwidth.tex',
    'TAB_SPEED': 'tab_speed.tex',
}


def _strip_shell(tex: str, name: str) -> str:
    """Remove a hand-written tabular shell wrapped around the marker block of ``name``."""
    start = tex.find(f'%%<{name}\n')
    end_marker = f'%%{name}>'
    end = tex.find(end_marker, start)
    if start < 0 or end < 0:
        return tex
    end += len(end_marker)
    # trailing "\bottomrule \end{tabular}" (possibly with a stray closing brace from resizebox)
    tail = re.match(r'\s*(\\bottomrule\s*)?\\end\{tabular\}\}?', tex[end:])
    if tail:
        tex = tex[:end] + '\n' + tex[end + tail.end():]
    # leading "\begin{tabular}{..} \toprule ... \midrule" with no \end{tabular} in between
    b = tex.rfind('\\begin{tabular}', 0, start)
    if b >= 0 and '\\end{tabular}' not in tex[b:start]:
        lead = tex[b:start]
        if re.match(r'\\begin\{tabular\}\{[^}]*\}', lead):
            # keep the indentation that preceded \begin{tabular}
            tex = tex[:b] + tex[start:]
    return tex


def fill(tex: str) -> tuple[str, list[str]]:
    report = []
    for name, fname in MAPPING.items():
        path = TAB_DIR / fname
        if not path.exists():
            report.append(f'{name}: table file missing, left untouched')
            continue
        body = path.read_text().strip('\n')
        block = f'%%<{name}\n{body}\n%%{name}>'
        placeholder = f'%%{name}%%'
        pattern = re.compile(rf'%%<{name}\n.*?\n%%{name}>', re.S)
        if placeholder in tex:
            tex = tex.replace(placeholder, block)
            report.append(f'{name}: inserted')
        elif pattern.search(tex):
            tex = pattern.sub(lambda _m: block, tex)
            if body.lstrip().startswith(('\\setlength', '\\begin{tabular}')):
                tex = _strip_shell(tex, name)
            report.append(f'{name}: replaced')
        else:
            report.append(f'{name}: no placeholder found')
    return tex, report


WIDE_TABLES: set = set()   # every table now fits the 31 pc text width in portrait


def normalise_floats(tex: str) -> tuple[str, int]:
    """Portrait ``table[!htbp]`` for every table not listed in WIDE_TABLES (those become
    ``sidewaystable``); figures get ``[!htbp]`` too.  Tables use the class's default
    8 bp body font (``\\scriptsize`` is 9 pt in sn-jnl, so it is removed), and tables
    holding a generated block drop their hand-written tabcolsep/arraystretch lines
    because the block carries its own.  Idempotent."""
    count = 0

    def repl(m):
        nonlocal count
        block = m.group(0)
        label = re.search(r'\\label\{(tab:[a-zA-Z0-9_]+)\}', block)
        block = block.replace('\\resizebox{\\textwidth}{!}{%\n', '').replace('\\end{tabular}}', '\\end{tabular}')
        block = re.sub(r'\n[ \t]*\\scriptsize[ \t]*(?=\n)', '', block)
        if '%%<TAB_' in block:
            block = re.sub(r'\n[ \t]*\\(?:renewcommand\{\\arraystretch\}|setlength\{\\tabcolsep\})\{[^}]*\}[ \t]*(?=\n)', '', block)
        wide = bool(label) and label.group(1) in WIDE_TABLES
        env_open = '\\begin{sidewaystable}' if wide else '\\begin{table}[!htbp]'
        env_close = '\\end{sidewaystable}' if wide else '\\end{table}'
        new = re.sub(r'\\begin\{(?:table|sidewaystable)\}(\[[a-zA-Z!]*\])?', lambda _m: env_open, block, count=1)
        new = re.sub(r'\\end\{(?:table|sidewaystable)\}$', lambda _m: env_close, new)
        if new != block:
            count += 1
        return new

    tex = re.sub(r'\\begin\{(table|sidewaystable)\}.*?\\end\{(table|sidewaystable)\}', repl, tex, flags=re.S)
    tex = re.sub(r'\\begin\{figure\}\[[a-zA-Z!]*\]', '\\\\begin{figure}[!htbp]', tex)
    return tex, count


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    tex_path = Path(sys.argv[1])
    tex = tex_path.read_text(encoding='utf-8')
    new, report = fill(tex)
    new, changed = normalise_floats(new)
    tex_path.write_text(new, encoding='utf-8')
    print('\n'.join(report))
    print(f'normalised {changed} float environments')
    leftovers = re.findall(r'%%[A-Z_]+%%', new)
    if leftovers:
        print('remaining placeholders:', sorted(set(leftovers)))


if __name__ == '__main__':
    main()

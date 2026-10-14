from pathlib import Path
import csv

f = Path('artifacts/results/benchmark_results_v5.csv')
if f.exists():
    with open(f, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    print('Rows written: %d/300' % len(rows))
    if rows:
        correct = sum(1 for r in rows if r.get('consensus_correct', '0') == '1')
        pct = 100 * correct // len(rows) if rows else 0
        print('Correct so far: %d/%d (%d%%)' % (correct, len(rows), pct))
        last_q = rows[-1]
        print('Last row_id:', last_q.get('row_id', '?'))
else:
    print('File not yet created')

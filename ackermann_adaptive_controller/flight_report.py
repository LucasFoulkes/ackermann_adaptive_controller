# Copyright 2026 Lucas Foulkes
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""Score a drive from the flight recorder -- ROS-free.

    python3 -m ackermann_adaptive_controller.flight_report            # last session
    python3 -m ackermann_adaptive_controller.flight_report --all      # every session
    ros2 run ackermann_adaptive_controller ackermann_flight_report ~/.ros/ackermann_flight.csv

A session is a run of the node: rows separated by more than SESSION_GAP
seconds of silence. For each, the numbers the operator feels in the room
-- launches and their overshoot, tracking error, surge-stall cycles,
stalls -- plus the model's trajectory minute by minute, so "did it
learn?" and "how well does it drive?" are answered by the log, not by a
theory. Reads the NUL-holed files a power cut leaves behind.
"""

import argparse
import math
import os
import statistics
import sys
import time

DEFAULT_LOG = os.path.expanduser('~/.ros/ackermann_flight.csv')
SESSION_GAP = 60.0      # s without rows = a new run of the node
# Same event definitions as core.DriveScore, so the offline score and
# the live one agree.
SURGE, STALL_FRAC, REACHED = 1.3, 0.3, 0.8


def read_log(path):
    """Rows as dicts of floats (strings for phase), NUL-tolerant."""
    rows = []
    with open(path, errors='replace') as fh:
        header = fh.readline().strip().split(',')
        for line in fh:
            if '\x00' in line or '�' in line:
                continue
            parts = line.rstrip('\n').split(',')
            if len(parts) != len(header):
                continue
            row = {}
            try:
                for k, v in zip(header, parts):
                    row[k] = v if k == 'phase' else float(v)
            except ValueError:
                continue
            rows.append(row)
    return header, rows


def sessions(rows):
    out, cur = [], []
    for r in rows:
        if cur and (r['stamp'] - cur[-1]['stamp'] > SESSION_GAP
                    or r['stamp'] < cur[-1]['stamp']):
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def score_session(s, cmd_min=0.05, gate=0.05):
    """The DriveScore of one session, computed offline."""
    t0 = s[0]['stamp']
    cmd = [r['cmd_v'] for r in s]
    v = [r['v'] for r in s]
    commanded = [i for i in range(len(s)) if abs(cmd[i]) > cmd_min]
    errs = [(cmd[i] - v[i]) ** 2 for i in commanded]
    err_rms = math.sqrt(sum(errs) / len(errs)) if errs else None
    mean_cmd = (sum(abs(cmd[i]) for i in commanded) / len(commanded)
                if commanded else 0.0)
    # segments of one command direction
    launches, cycles, seg = [], 0, None
    surged = False
    stalls = sum(1 for i in range(1, len(s))
                 if s[i]['stalled'] and not s[i - 1]['stalled'])
    for i in range(len(s)):
        d = (1 if cmd[i] > cmd_min else -1 if cmd[i] < -cmd_min else 0)
        if seg is None or d != seg['dir']:
            if seg and seg.get('reached') is not None:
                launches.append(seg)
            seg = {'dir': d, 'i0': i, 'peak': 0.0, 'reached': None,
                   'from_rest': abs(v[i]) < gate}
            surged = False
        if not d:
            continue
        a, c = abs(v[i]), abs(cmd[i])
        seg['peak'] = max(seg['peak'], a)
        seg['cmd'] = max(seg.get('cmd', 0.0), c)   # largest command seen
        if seg['from_rest'] and seg['reached'] is None and a >= REACHED * c:
            seg['reached'] = s[i]['stamp'] - s[seg['i0']]['stamp']
        if a >= SURGE * c:
            surged = True
        elif surged and a < STALL_FRAC * c:
            surged = False
            cycles += 1
    if seg and seg.get('reached') is not None:
        launches.append(seg)
    overs = [L['peak'] / L['cmd'] for L in launches if L['cmd'] > 0]
    reach = [L['reached'] for L in launches]
    # model trajectory per minute
    minutes = []
    for m in range(int((s[-1]['stamp'] - t0) / 60.0) + 1):
        seg_rows = [r for r in s if m * 60 <= r['stamp'] - t0 < (m + 1) * 60]
        if not seg_rows:
            continue
        e = seg_rows[-1]
        c2 = [r for r in seg_rows if abs(r['cmd_v']) > 0.2]
        rms = (math.sqrt(sum((r['cmd_v'] - r['v']) ** 2 for r in c2) / len(c2))
               if c2 else None)
        minutes.append({
            'min': m, 'rows': len(seg_rows), 'commanded': len(c2),
            'err_rms': rms,
            'b0': e['b0'], 'b1': e['b1'], 'b3': e['b3'],
            'breakaway': e['breakaway'],
            'probe_b0': e.get('probe_b0'), 'probe_eq': e.get('probe_eq'),
            'a0l': e['a0l'], 'a0r': e['a0r'],
            'a0lr': e['a0lr'], 'a0rr': e['a0rr'],
            'ready_lon': int(e['ready_lon']), 'ready_lat': int(e['ready_lat']),
        })
    # steering, per (travel direction x steering side) cell: curvature
    # achieved over commanded, how often the servo sat at lock, and the
    # empirical gain (kappa over the steering command one delay earlier,
    # steady for half a second) against the fitted cell
    cells = {}
    for i, r in enumerate(s):
        vv = r['v']
        if abs(vv) < 0.2 or abs(r['cmd_v']) < 0.1:
            continue
        j = i
        while j > 0 and s[i]['stamp'] - s[j]['stamp'] < 0.45:
            j -= 1
        k = j
        while k > 0 and s[i]['stamp'] - s[k]['stamp'] < 0.95:
            k -= 1
        qs = s[j]['qs']
        key = ('fwd' if vv > 0 else 'rev',
               'left' if qs > 0.05 else 'right' if qs < -0.05 else None)
        if key[1] is None:
            continue
        c = cells.setdefault(key, {'ratio': [], 'gain': [], 'sat': 0, 'n': 0})
        c['n'] += 1
        c['sat'] += abs(r['qs']) >= 0.9
        k_cmd = r['cmd_w'] / r['cmd_v']
        if abs(k_cmd) > 0.3:
            c['ratio'].append((r['psidot'] / vv) / k_cmd)
        if abs(qs) > 0.2 and abs(s[k]['qs'] - qs) <= 0.1:
            c['gain'].append((r['psidot'] / vv) / qs)
    last = s[-1]
    fitted = {('fwd', 'left'): last['a0l'], ('fwd', 'right'): last['a0r'],
              ('rev', 'left'): last['a0lr'], ('rev', 'right'): last['a0rr']}
    steering = []
    for key in sorted(cells):
        c = cells[key]
        steering.append({
            'cell': f'{key[0]} {key[1]}', 'n': c['n'],
            'ratio': statistics.median(c['ratio']) if c['ratio'] else None,
            'sat': c['sat'] / c['n'],
            'gain': statistics.median(c['gain']) if c['gain'] else None,
            'fitted': fitted[key],
        })
    return {
        'steering': steering,
        'start': t0, 'end': s[-1]['stamp'], 'rows': len(s),
        'commanded': len(commanded), 'mean_cmd': mean_cmd,
        'err_rms': err_rms,
        'err_rel': (err_rms / mean_cmd if err_rms is not None and mean_cmd > 0
                    else None),
        'launches': len(launches),
        'launch_over_median': statistics.median(overs) if overs else None,
        'launch_over_max': max(overs) if overs else None,
        'launch_reach_median': statistics.median(reach) if reach else None,
        'cycles': cycles, 'stalls': stalls,
        'minutes': minutes,
    }


def verdict(sc):
    """One line, in the operator's terms."""
    if sc['commanded'] < 50:
        return 'too little driving to judge'
    dur = max(sc['end'] - sc['start'], 1.0) / 60.0
    cpm = sc['cycles'] / dur
    if cpm >= 2.0:
        return f'LUNGING: {cpm:.1f} surge-stall cycles/min'
    parts = []
    if sc['launch_over_median'] and sc['launch_over_median'] > 1.6:
        parts.append(f'launches overshoot {sc["launch_over_median"]:.1f}x')
    if sc['err_rel'] is not None and sc['err_rel'] > 0.5:
        parts.append(f'tracking error {sc["err_rel"] * 100:.0f}% of command')
    if cpm >= 0.5:
        parts.append(f'{cpm:.1f} cycles/min')
    return 'rough: ' + ', '.join(parts) if parts else 'smooth'


def fmt(x, spec='.2f'):
    return '  -  ' if x is None else format(x, spec)


def render(sc):
    lt = time.localtime
    out = [
        f"session {time.strftime('%m-%d %H:%M:%S', lt(sc['start']))} - "
        f"{time.strftime('%H:%M:%S', lt(sc['end']))}  "
        f"({(sc['end'] - sc['start']) / 60:.1f} min, {sc['rows']} ticks, "
        f"{sc['commanded']} commanded)",
        f"  verdict: {verdict(sc)}",
        f"  tracking: err_rms {fmt(sc['err_rms'], '.3f')} m/s "
        f"({fmt(None if sc['err_rel'] is None else sc['err_rel'] * 100, '.0f')}% "
        f"of the mean command {sc['mean_cmd']:.2f})",
        f"  launches: {sc['launches']}  overshoot median "
        f"{fmt(sc['launch_over_median'])}x max {fmt(sc['launch_over_max'])}x  "
        f"time-to-command median {fmt(sc['launch_reach_median'], '.1f')} s",
        f"  surge-stall cycles: {sc['cycles']}   stalls: {sc['stalls']}",
        "  steering   cell      n  achieved/cmd  at-lock  gain measured / fitted",
    ]
    for st_ in sc['steering']:
        out.append(
            f"             {st_['cell']:9s} {st_['n']:5d}     {fmt(st_['ratio'])}"
            f"      {st_['sat'] * 100:3.0f}%     {fmt(st_['gain'])} / "
            f"{st_['fitted']:.2f}")
    out += [
        "  min  cmd  err_rms |   b0    b1    b3  brk | probe b0  eq | "
        "a0l  a0r  a0lr a0rr | rdy",
    ]
    for m in sc['minutes']:
        out.append(
            f"  {m['min']:3d} {m['commanded']:4d}  {fmt(m['err_rms'], '.3f')} | "
            f"{m['b0']:5.2f} {m['b1']:+5.2f} {m['b3']:+5.2f} {m['breakaway']:.2f} | "
            f"{fmt(m['probe_b0'])}  {fmt(m['probe_eq'])} | "
            f"{m['a0l']:4.2f} {m['a0r']:4.2f} {m['a0lr']:4.2f} {m['a0rr']:4.2f} | "
            f"{m['ready_lon']}{m['ready_lat']}")
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('log', nargs='?', default=DEFAULT_LOG)
    ap.add_argument('--all', action='store_true', help='every session')
    ap.add_argument('--last', type=int, default=1,
                    help='how many of the latest sessions (default 1)')
    args = ap.parse_args(argv)
    if not os.path.exists(args.log):
        print(f'no flight log at {args.log}', file=sys.stderr)
        return 1
    _, rows = read_log(args.log)
    ss = sessions(rows)
    if not ss:
        print('flight log is empty', file=sys.stderr)
        return 1
    chosen = ss if args.all else ss[-args.last:]
    for s in chosen:
        print(render(score_session(s)))
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())

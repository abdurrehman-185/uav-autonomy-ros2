#!/usr/bin/env python3
"""Builds a 1280x720 H.264 video from recorded camera frames and a run log.

Layout: chase camera (top left), overhead camera (bottom left), live top-view map with
paths and error traces (right). Usage: python3 compose_video.py video_config.json
"""
import bisect
import csv
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

W, H = 1280, 720
BG = (24, 24, 28)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_frames(folder):
    if not folder or not os.path.isdir(folder):
        return [], []
    names = sorted(f for f in os.listdir(folder) if f.endswith('.jpg'))
    return [float(n[:-4]) for n in names], [os.path.join(folder, n) for n in names]


def frame_at(times, paths, t, size, cache):
    if not times:
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.putText(img, 'no camera frames', (20, size[1] // 2), FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        return img
    i = max(0, bisect.bisect_right(times, t) - 1)
    if cache.get('i') != i:
        img = cv2.imread(paths[i])
        cache['i'], cache['img'] = i, cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cache['img']


def label(img, text, org, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def main():
    cfg = json.load(open(sys.argv[1]))
    base = os.path.dirname(os.path.abspath(sys.argv[1]))
    path = lambda p: p if os.path.isabs(p) else os.path.join(base, p)

    with open(path(cfg['run_csv'])) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r['t']) for r in rows])
    col = lambda k: np.array([float(r[k]) for r in rows])
    tx, ty = col(cfg['truth'][0]), col(cfg['truth'][1])
    series = [dict(s, X=col(s['x']), Y=col(s['y'])) for s in cfg['series']]
    for s in series:
        s['E'] = np.hypot(s['X'] - tx, s['Y'] - ty)
    states = [r.get(cfg.get('state_col', 'state'), '') for r in rows]

    ct, cp = load_frames(path(cfg['frames']['chase'])) if cfg['frames'].get('chase') else ([], [])
    ot, op = load_frames(path(cfg['frames']['overhead'])) if cfg['frames'].get('overhead') else ([], [])
    cache_c, cache_o = {}, {}

    # Map panel geometry.
    mx0, my0, mw, mh = 650, 78, 620, 400
    xmin, xmax, ymin, ymax = cfg['bounds']
    scale = min(mw / (xmax - xmin), mh / (ymax - ymin))
    ox = mx0 + (mw - scale * (xmax - xmin)) / 2
    oy = my0 + (mh + scale * (ymax - ymin)) / 2
    to_px = lambda x, y: (int(ox + (x - xmin) * scale), int(oy - (y - ymin) * scale))

    base_map = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(base_map, (mx0, my0), (mx0 + mw, my0 + mh), (36, 36, 42), -1)
    for x0, y0, w, h in cfg.get('walls', []):
        cv2.rectangle(base_map, to_px(x0, y0 + h), to_px(x0 + w, y0), (110, 110, 118), -1)
    for cx, cy, r in cfg.get('circles', []):
        cv2.circle(base_map, to_px(cx, cy), max(2, int(r * scale)), (90, 130, 90), -1)
    if cfg.get('map_npy') and os.path.exists(path(cfg['map_npy'])):
        for x, y in np.load(path(cfg['map_npy']))[::2]:
            cv2.circle(base_map, to_px(x, y), 1, (40, 110, 170), -1)
    for wx, wy in cfg.get('waypoints', []):
        cv2.drawMarker(base_map, to_px(wx, wy), (160, 160, 160), cv2.MARKER_TILTED_CROSS, 9, 1)

    # Error panel geometry.
    ex0, ey0, ew, eh = 650, 520, 620, 170
    emax = float(cfg.get('error_ylim', max(0.5, max(np.percentile(s['E'], 99) for s in series if s.get('error', True)))))
    t0, t1 = float(t[0]), float(t[-1])
    to_e = lambda tt, e: (int(ex0 + (tt - t0) / (t1 - t0) * ew), int(ey0 + eh - min(e, emax) / emax * eh))

    speed = float(cfg.get('speed', 2.0))
    fps = 30
    n_frames = int((t1 - t0) / speed * fps) + fps * 2
    out = path(cfg['out'])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                           '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                           '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], stdin=subprocess.PIPE)

    for k in range(n_frames):
        tt = min(t1, t0 + k * speed / fps)
        i = max(1, bisect.bisect_right(t, tt))
        img = base_map.copy()
        img[0:360, 0:640] = frame_at(ct, cp, tt, (640, 360), cache_c)
        img[360:720, 0:640] = frame_at(ot, op, tt, (640, 360), cache_o)
        label(img, 'Chase camera', (12, 26))
        label(img, 'Overhead camera', (12, 386))

        panel = img[my0:my0 + mh, mx0:mx0 + mw].copy()
        rel = lambda q: (q[0] - mx0, q[1] - my0)
        tpts = np.array([rel(to_px(x, y)) for x, y in zip(tx[:i:2], ty[:i:2])], np.int32)
        if len(tpts) > 1:
            cv2.polylines(panel, [tpts], False, (255, 255, 255), 5, cv2.LINE_AA)
        for s in series:
            pts = np.array([rel(to_px(x, y)) for x, y in zip(s['X'][:i:2], s['Y'][:i:2])], np.int32)
            if len(pts) > 1:
                cv2.polylines(panel, [pts], False, tuple(s['color']), int(s.get('width', 2)), cv2.LINE_AA)
        cv2.circle(panel, rel(to_px(tx[i - 1], ty[i - 1])), 8, (255, 255, 255), 2, cv2.LINE_AA)
        for s in series:
            cv2.circle(panel, rel(to_px(s['X'][i - 1], s['Y'][i - 1])), 5, tuple(s['color']), -1, cv2.LINE_AA)
        img[my0:my0 + mh, mx0:mx0 + mw] = panel

        # Legend.
        ly = my0 + mh + 22
        lx = mx0
        for name, color in [('Truth', (255, 255, 255))] + [(s['label'], tuple(s['color'])) for s in series]:
            cv2.line(img, (lx, ly - 5), (lx + 22, ly - 5), color, 3)
            label(img, name, (lx + 28, ly), 0.45)
            lx += 40 + 9 * len(name)

        # Error traces.
        cv2.rectangle(img, (ex0, ey0), (ex0 + ew, ey0 + eh), (36, 36, 42), -1)
        for s in series:
            if not s.get('error', True):
                continue
            pts = np.array([to_e(a, b) for a, b in zip(t[:i:2], s['E'][:i:2])], np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, tuple(s['color']), 2, cv2.LINE_AA)
        for ev in cfg.get('events', []):
            if ev['t'] <= tt:
                x = max(ex0, to_e(ev['t'], 0)[0])
                cv2.line(img, (x, ey0), (x, ey0 + eh), (80, 80, 220), 1)
        label(img, f'Position error (m), axis max {emax:g} m', (ex0 + 6, ey0 - 8), 0.45)

        # Header and live numbers.
        label(img, cfg['title'], (650, 30), float(cfg.get('title_scale', 0.65)), (255, 255, 255), 2)
        stats = [f't = {tt - t0:5.1f} s'] + [f"{s['label']}: {s['E'][i - 1]:.2f} m" for s in series if s.get('error', True)]
        label(img, '   '.join(stats), (650, 60), 0.5, (210, 230, 255))
        active = [ev for ev in cfg.get('events', []) if ev['t'] <= tt < ev.get('until', 1e9)]
        if active:
            text = active[-1]['text']
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
            cv2.rectangle(img, (mx0 + mw - tw - 24, my0 + 8), (mx0 + mw - 8, my0 + 18 + th), (40, 40, 200), -1)
            cv2.putText(img, text, (mx0 + mw - tw - 16, my0 + 14 + th), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        label(img, f'{speed:g}x speed', (W - 110, H - 10), 0.45, (170, 170, 170))
        ff.stdin.write(img.tobytes())

    ff.stdin.close()
    ff.wait()
    print('video:', out, f'({os.path.getsize(out) / 1e6:.1f} MB, {n_frames / fps:.0f} s)')

    gif = cfg.get('gif')
    if gif:
        g = path(gif)
        start = float(cfg.get('gif_start', 0))
        dur = float(cfg.get('gif_seconds', 14))
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(start), '-t', str(dur), '-i', out, '-vf',
                        'fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer',
                        g], check=True)
        print('gif:', g, f'({os.path.getsize(g) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()

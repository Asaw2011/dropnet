"""Fake eye-drop clips from real eye photos.

The clip camera sits next to the nozzle looking at the eye, so a drop shows up big near
the frame edge, shrinks as it falls toward the eye, then either goes in (spreads into the
tear film, often a blink) or misses (beads on the lid/cheek, maybe runs down).
Labels: 0 = went in, 1 = missed, 2 = no drop came out.
"""
import cv2
import numpy as np

W, H = 128, 96
T = 12
IN_EYE, MISSED, NO_DROP = 0, 1, 2
LABELS = ["went_in", "missed", "no_drop"]


def _opening(mask):
    return np.isin(mask, (2, 3, 4))


def augment_base(img, mask, rng):
    """random zoom/shift/flip/color, returns (img HxWx3 uint8, opening bool) at W x H"""
    h0, w0 = img.shape[:2]
    s = rng.uniform(0.75, 1.0)
    cw, ch = w0 * s, h0 * s
    x0, y0 = rng.uniform(0, w0 - cw), rng.uniform(0, h0 - ch)
    ang = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((x0 + cw / 2, y0 + ch / 2), ang, W / cw)
    M[0, 2] += W / 2 - (x0 + cw / 2)
    M[1, 2] += H / 2 - (y0 + ch / 2)
    im = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.5:
        im, m = im[:, ::-1].copy(), m[:, ::-1].copy()

    f = im.astype(np.float32)
    f = f * rng.uniform(0.85, 1.15, 3)  # white balance
    f = (f - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-35, 25)
    f = 255 * (np.clip(f, 0, 255) / 255) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.25:  # noir / ir camera look
        g = f.mean(2, keepdims=True)
        f = np.repeat(g, 3, 2)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        f = cv2.GaussianBlur(f, (k, k), 0)
    return np.clip(f, 0, 255).astype(np.uint8), _opening(m)


def _drop(frame, cx, cy, r, rng_state, alpha=1.0, flat=1.0, blur=0):
    """water drop = little lens: inverted minified view inside, dark rim, specular dot"""
    sp_dx, sp_dy, rim, spec = rng_state
    r = max(r, 1.2)
    pad = int(r * 2 + 4)
    x0, x1 = int(cx - pad), int(cx + pad)
    y0, y1 = int(cy - pad), int(cy + pad)
    if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
        return frame
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = (xx - cx) / r, (yy - cy) / (r * flat)
    d2 = dx * dx + dy * dy
    inside = d2 < 1
    # lens: sample bg from flipped + squeezed spot
    mapx = (cx - dx * r * 0.6).astype(np.float32)
    mapy = (cy - dy * r * flat * 0.6).astype(np.float32)
    src = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32)
    lens = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    edge = np.clip(d2, 0, 1) ** 3
    lens = lens * (1 - rim * edge[..., None]) + 18
    sd = (dx - sp_dx) ** 2 + (dy - sp_dy) ** 2
    lens += spec * np.exp(-sd / 0.03)[..., None]
    a = np.clip((1 - d2) * 6, 0, 1) * alpha
    if blur:
        a = cv2.GaussianBlur(a, (0, 0), blur)
        lens = cv2.GaussianBlur(lens, (0, 0), blur)
    # paste with clipping
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x1, W), min(y1, H)
    sl = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
    out = frame.astype(np.float32)
    patch = out[fy0:fy1, fx0:fx1]
    aa = a[sl][..., None]
    out[fy0:fy1, fx0:fx1] = patch * (1 - aa) + np.clip(lens[sl], 0, 255) * aa
    return out.clip(0, 255).astype(np.uint8)


def _wet(frame, region, strength, rng, n_spec=3):
    """tear film / wet skin: a bit darker + glossy highlights inside region"""
    if strength <= 0 or not region.any():
        return frame
    g = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.2) * strength
    f = frame.astype(np.float32)
    f = f * (1 - 0.12 * g[..., None])
    ys, xs = np.where(region)
    hl = np.zeros((H, W), np.float32)
    for _ in range(n_spec):
        i = rng.integers(len(ys))
        cv2.ellipse(hl, (int(xs[i]), int(ys[i])), (int(rng.integers(1, 4)), int(rng.integers(1, 3))),
                    rng.uniform(0, 180), 0, 360, 1.0, -1)
    hl = cv2.GaussianBlur(hl, (0, 0), 0.8) * region * strength
    f += 140 * hl[..., None]
    return f.clip(0, 255).astype(np.uint8)


def _lid_cover(frame, opening, frac):
    """blink: slide upper lid skin down over the opening. frac 0 open, 1 shut"""
    if frac <= 0:
        return frame
    region = cv2.dilate(opening.astype(np.uint8), np.ones((3, 5), np.uint8)).astype(bool)
    cols = np.where(region.any(0))[0]
    if len(cols) == 0:
        return frame
    # per column top/bottom so the lid edge follows the eye shape
    top = np.full(W, H, np.float32)
    bot = np.zeros(W, np.float32)
    top[cols] = region[:, cols].argmax(0)
    bot[cols] = H - 1 - region[::-1, cols].argmax(0)
    eh = int((bot[cols] - top[cols]).max()) + 4
    lid = cv2.GaussianBlur(np.roll(frame, eh, axis=0), (3, 3), 0)  # skin from just above
    edge = top + frac * (bot - top + 1)
    yy = np.arange(H, dtype=np.float32)[:, None]
    a = np.clip((edge[None, :] - yy) / 1.5, 0, 1) * region
    a = cv2.GaussianBlur(a.astype(np.float32), (3, 3), 0)[..., None]
    out = frame.astype(np.float32) * (1 - a) + lid.astype(np.float32) * a
    lash = (np.abs(yy - edge[None, :]) < 1.3) & region & (frac > 0.08)
    out[lash] *= 0.4
    return out.clip(0, 255).astype(np.uint8)


def _blink_curve(t0, dur):
    def f(t):
        if t < t0 or t > t0 + dur:
            return 0.0
        u = (t - t0) / dur
        return float(np.sin(np.pi * u) ** 0.6)
    return f


def _sample_point(region, rng):
    ys, xs = np.where(region)
    if len(ys) == 0:
        return None
    i = rng.integers(len(ys))
    return float(xs[i]), float(ys[i])


def make_clip(img, mask, label, rng):
    base, opening = augment_base(img, mask, rng)
    if opening.sum() < 40:  # eye cropped away, just call it no drop
        label = NO_DROP

    t_imp = int(rng.integers(4, 8))
    fall = int(rng.integers(2, 5))
    drop_style = (rng.uniform(-0.45, -0.15), rng.uniform(-0.45, -0.15), rng.uniform(0.3, 0.7), rng.uniform(60, 160))
    r_imp = rng.uniform(2.5, 5.5)
    r0 = rng.uniform(10, 22)
    # drop enters from where the nozzle is, mostly top side of frame
    start = (rng.uniform(0.15, 0.85) * W, rng.uniform(-0.25, 0.25) * H)

    blink = lambda t: 0.0
    closed_hit = False
    run_down = False
    overflow = False
    hang = False
    target = None

    eye_dil = cv2.dilate(opening.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if label == IN_EYE:
        # aim mostly at the lower half, that's where drops pool
        ys, xs = np.where(opening)
        lower = opening & (np.arange(H)[:, None] > ys.mean() - 2)
        target = _sample_point(lower if lower.sum() > 10 and rng.random() < 0.7 else opening, rng)
        if rng.random() < 0.6:
            blink = _blink_curve(t_imp + rng.integers(1, 3), rng.integers(3, 6))
        overflow = rng.random() < 0.25
    elif label == MISSED:
        skin = ~cv2.dilate(opening.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        near = skin & cv2.dilate(opening.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
        mode = rng.random()
        if mode < 0.15:
            # eye shut right as the drop lands, it hits the lid
            closed_hit = True
            target = _sample_point(opening, rng)
            blink = _blink_curve(t_imp - 2, rng.integers(5, 8))
        elif mode < 0.55:
            target = _sample_point(near, rng)
        else:
            target = _sample_point(skin, rng)
        run_down = rng.random() < 0.6
        if not closed_hit and rng.random() < 0.3:
            blink = _blink_curve(t_imp + rng.integers(0, 3), rng.integers(3, 6))
    else:
        r = rng.random()
        if r < 0.4:
            blink = _blink_curve(rng.integers(1, T - 3), rng.integers(3, 6))
        elif r < 0.6:
            hang = True  # drop sits on the nozzle edge and never falls
    if target is None and label != NO_DROP:
        label = NO_DROP

    shake = rng.normal(0, rng.uniform(0.2, 1.8), (T, 2)).cumsum(0) * 0.5
    flicker = rng.normal(0, rng.uniform(0, 6), T)
    noise = rng.uniform(1, 7)

    frames = []
    bead = None
    trail = np.zeros((H, W), bool)
    wet_eye = np.zeros((H, W), bool)
    for t in range(T):
        f = base.copy()
        if wet_eye.any():
            f = _wet(f, wet_eye & opening, 1.0, rng, n_spec=2)
        if trail.any():
            f = _wet(f, trail, 0.8, rng, n_spec=1)
        f = _lid_cover(f, opening, blink(t))

        if label != NO_DROP:
            tx, ty = target
            if t_imp - fall <= t < t_imp:
                u = (t - (t_imp - fall)) / fall
                u = u ** 1.4
                cx = start[0] + (tx - start[0]) * u
                cy = start[1] + (ty - start[1]) * u
                r = r0 + (r_imp - r0) * u
                f = _drop(f, cx, cy, r, drop_style, blur=max(0.0, 2.5 * (1 - u)))
            elif t >= t_imp:
                k = t - t_imp
                if label == IN_EYE:
                    # splash then spread into the tear film
                    if k == 0:
                        f = _drop(f, tx, ty, r_imp * 1.3, drop_style, flat=0.6)
                    rr = int(r_imp * (1.5 + 0.8 * k))
                    disk = np.zeros((H, W), np.uint8)
                    cv2.circle(disk, (int(tx), int(ty)), rr, 1, -1)
                    wet_eye |= disk.astype(bool) & eye_dil
                    if k >= 1:
                        # meniscus at the lower lid lights up
                        lower_edge = opening & ~np.roll(opening, -2, axis=0)
                        f = _wet(f, lower_edge, min(1.0, 0.4 * k), rng, n_spec=2)
                    if overflow and k >= 2:
                        if bead is None:
                            ys, xs = np.where(opening)
                            col = xs[(ys == ys.max())][0] if len(xs) else tx
                            bead = [float(col), float(ys.max() + 2)]
                        bead[1] += rng.uniform(1, 3)
                        trail[int(min(bead[1], H - 1)), int(bead[0])] = True
                        trail = cv2.dilate(trail.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool) & ~opening
                        f = _drop(f, bead[0], bead[1], r_imp * 0.7, drop_style, flat=0.8)
                else:
                    if bead is None:
                        bead = [tx, ty]
                    elif run_down:
                        nx = bead[0] + rng.uniform(-0.5, 0.5)
                        ny = bead[1] + rng.uniform(0.8, 3.0)
                        # don't let it roll into the eye, that'd be a different label
                        if not (0 <= int(ny) < H and eye_dil[int(ny), int(np.clip(nx, 0, W - 1))]):
                            if 0 <= int(ny) < H:
                                trail[int(ny), int(np.clip(nx, 0, W - 1))] = True
                            bead = [nx, ny]
                    trail = cv2.dilate(trail.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
                    f = _drop(f, bead[0], bead[1], r_imp * (1.4 if k == 0 else 1.1), drop_style, flat=0.75)
        elif hang:
            f = _drop(f, start[0], max(start[1], 2), r0 * 0.8, drop_style, blur=2.0)

        # handheld shake, flicker, sensor noise
        M = np.float32([[1, 0, shake[t, 0]], [0, 1, shake[t, 1]]])
        f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        f += flicker[t] + rng.normal(0, noise, f.shape)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(frames), label

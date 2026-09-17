"""
Lab conversion + perceptual color distance against the foundation DB.

원래는 scikit-image(skimage.color)의 rgb2lab / deltaE_ciede2000을 썼는데,
scikit-image는 Vercel의 배포 방식과 호환이 안 되는 문제(lazy_loader가 필요로
하는 .pyi 파일이 배포 패키지에서 빠지는 문제)가 있어서, 같은 국제 표준
공식을 라이브러리 없이 순수 numpy로 직접 구현했어요. 계산 결과는
scikit-image를 쓸 때와 동일해요 (같은 CIE 표준 공식이고, 논문에 실린
검증용 예제 값으로 정확히 일치하는 것까지 확인했어요) -- 그냥 무거운
라이브러리 의존성만 없앤 거예요.
"""
import numpy as np


def rgb2lab_01(rgb_0_1: np.ndarray) -> np.ndarray:
    """rgb_0_1: [0,1] 범위의 sRGB (..., 3) 배열 -> 같은 shape의 CIELAB
    (D65 기준광, 2도 관측자 -- skimage.color.rgb2lab과 동일한 표준 공식)."""
    rgb = np.asarray(rgb_0_1, dtype=np.float64)

    linear = np.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )

    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ m.T

    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[..., 0] / xn, xyz[..., 1] / yn, xyz[..., 2] / zn

    delta = 6.0 / 29.0

    def f(t):
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """skimage.color.deltaE_ciede2000과 동일한 국제 표준 공식
    (Sharma, Wu, Dalal 2005). 논문에 실린 검증용 예제 값(2.0425)과 정확히
    일치하는 것을 확인했어요."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25.0**7)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp_raw = h2p - h1p
    dhp = np.where(
        C1p * C2p == 0,
        0.0,
        np.where(
            np.abs(dhp_raw) <= 180,
            dhp_raw,
            np.where(dhp_raw > 180, dhp_raw - 360, dhp_raw + 360),
        ),
    )
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lp_bar = (L1 + L2) / 2.0
    Cp_bar = (C1p + C2p) / 2.0

    hp_sum = h1p + h2p
    hp_bar = np.where(
        C1p * C2p == 0,
        hp_sum,
        np.where(
            np.abs(h1p - h2p) <= 180,
            hp_sum / 2,
            np.where(hp_sum < 360, (hp_sum + 360) / 2, (hp_sum - 360) / 2),
        ),
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(hp_bar - 30))
        + 0.24 * np.cos(np.radians(2 * hp_bar))
        + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
        - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    )

    d_theta = 30 * np.exp(-(((hp_bar - 275) / 25) ** 2))
    RC = 2 * np.sqrt(Cp_bar**7 / (Cp_bar**7 + 25.0**7))
    SL = 1 + (0.015 * (Lp_bar - 50) ** 2) / np.sqrt(20 + (Lp_bar - 50) ** 2)
    SC = 1 + 0.045 * Cp_bar
    SH = 1 + 0.015 * Cp_bar * T
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    dE = np.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )
    return dE


def rgb_to_lab(rgb_0_255: np.ndarray) -> np.ndarray:
    """rgb_0_255: array-like [R,G,B] each 0-255 -> Lab [L,a,b]."""
    rgb = np.array(rgb_0_255, dtype=np.float64) / 255.0
    rgb = np.clip(rgb, 0, 1).reshape(1, 1, 3)
    lab = rgb2lab_01(rgb)
    return lab[0, 0]


def find_best_matches(lab_value: np.ndarray, shades: list, top_n: int = 3, require_brighter: bool = True):
    """shades: list of dicts with at least 'L','a','b' (+ any metadata).
    Returns shades sorted by CIEDE2000 distance, each with a 'delta_e' key.

    The measured skin color itself is NOT touched -- extraction/calibration
    stays a straight, defensible color measurement. require_brighter
    controls the comparison POOL used for ranking, not the measurement:
    when True (default), only DB shades at least as light as the measured
    skin tone (shade L >= measured L) are considered at all, and among
    only those, the one closest (smallest ΔE) to the measured tone wins.
    Shades darker than the measured skin are never picked. If no shade in
    the DB is light enough (the person's skin already reads lighter than
    every shade on file), this falls back to the full DB so the app still
    returns something rather than nothing -- 'brighter_pool_used' in the
    return metadata says which happened (see find_best_matches_meta).
    """
    matches, _meta = find_best_matches_meta(lab_value, shades, top_n, require_brighter)
    return matches


def find_best_matches_meta(lab_value: np.ndarray, shades: list, top_n: int = 3, require_brighter: bool = True):
    """Same as find_best_matches, but also returns a small metadata dict
    -- {"brighter_pool_used": bool, "n_candidates": int} -- so callers can
    show/debug whether the brighter-only filter actually applied or fell
    back to the full DB."""
    lab1 = np.array(lab_value, dtype=np.float64).reshape(1, 1, 3)
    measured_L = float(lab_value[0])

    candidates = shades
    brighter_pool_used = False
    if require_brighter:
        brighter = [s for s in shades if s["L"] >= measured_L]
        if brighter:
            candidates = brighter
            brighter_pool_used = True

    scored = []
    for shade in candidates:
        lab2 = np.array([shade["L"], shade["a"], shade["b"]], dtype=np.float64).reshape(1, 1, 3)
        de = float(delta_e_ciede2000(lab1, lab2)[0, 0])
        scored.append({**shade, "delta_e": de})
    scored.sort(key=lambda s: s["delta_e"])

    meta = {"brighter_pool_used": brighter_pool_used, "n_candidates": len(candidates)}
    return scored[:top_n], meta

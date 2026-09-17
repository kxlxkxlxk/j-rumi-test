"""
Stage 1 검증용 최종 통합 테스트 -- v2 (scikit-image 제거 버전).

scikit-image는 내부적으로 특수한 지연 로딩(lazy loading) 방식을 쓰는데,
이게 Vercel의 배포 패키징 방식과 궁합이 안 좋아서 계속 에러가 났어요
(다른 사람들도 겪는 알려진 문제예요). 우리가 scikit-image에서 실제로 쓰는
기능은 딱 두 가지 -- (1) sRGB -> Lab 색공간 변환, (2) CIEDE2000 색상 차이
계산 -- 뿐이라서, 이 두 계산을 표준 공식 그대로 순수 numpy 코드로 직접
구현해서 라이브러리 의존성 자체를 없앴어요. 계산 결과는 동일해요 (같은
국제 표준 공식이에요), 그냥 무거운 라이브러리를 안 쓰는 것뿐이에요.
"""
from fastapi import FastAPI
import numpy as np

app = FastAPI()


# ---------------------------------------------------------------------------
# sRGB -> CIELAB 변환 (D65 기준광, 2도 관측자) -- skimage.color.rgb2lab과 동일한 표준 공식
# ---------------------------------------------------------------------------
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: [0,1] 범위의 sRGB 값 (..., 3) 배열. 반환: 같은 shape의 Lab 값."""
    rgb = np.asarray(rgb, dtype=np.float64)

    # 1) sRGB -> 선형 RGB (감마 보정 해제)
    linear = np.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )

    # 2) 선형 RGB -> XYZ (sRGB, D65 기준 표준 변환 행렬)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ m.T

    # 3) XYZ -> Lab (D65 기준광 백색점으로 정규화)
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


# ---------------------------------------------------------------------------
# CIEDE2000 색상 차이 -- skimage.color.deltaE_ciede2000과 동일한 국제 표준 공식
# (Sharma, Wu, Dalal 2005)
# ---------------------------------------------------------------------------
def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
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


@app.get("/")
def health():
    return {"status": "ok", "message": "Stage 1 최종 통합 테스트 서버 (scikit-image 없는 버전)가 정상적으로 떠 있어요"}


@app.get("/api/test-deps")
def test_deps():
    results = {}

    # --- OpenCV ---
    import cv2
    dummy_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    dummy_bgr[:] = (10, 20, 30)
    converted = cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB)
    results["opencv"] = {
        "version": cv2.__version__,
        "test_ok": bool(converted.shape == (4, 4, 3)),
    }

    # --- 직접 구현한 색상 변환 (scikit-image 대체) ---
    lab = rgb_to_lab(np.array([0.5, 0.4, 0.3]))
    de_same = delta_e_ciede2000(lab, lab)  # 같은 색끼리는 0에 가까워야 함
    lab2 = rgb_to_lab(np.array([0.9, 0.1, 0.1]))
    de_diff = delta_e_ciede2000(lab, lab2)  # 다른 색끼리는 값이 커야 함
    results["color_math"] = {
        "lab_example": lab.tolist(),
        "delta_e_same_color": float(de_same),
        "delta_e_different_color": float(de_diff),
        "test_ok": bool(de_same < 0.01 and de_diff > 10),
    }

    # --- scipy (Hungarian algorithm) ---
    import scipy
    from scipy.optimize import linear_sum_assignment
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    row_ind, col_ind = linear_sum_assignment(cost)
    results["scipy"] = {
        "version": scipy.__version__,
        "test_ok": bool(len(row_ind) == 3),
    }

    # --- mediapipe ---
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision  # noqa: F401
    from mediapipe.tasks.python import core as mp_core  # noqa: F401
    results["mediapipe"] = {
        "version": mp.__version__,
        "tasks_api_importable": True,
    }

    # --- Pillow ---
    import PIL
    results["pillow"] = {"version": PIL.__version__}

    results["numpy_version"] = np.__version__
    results["all_ok"] = all(
        results[k].get("test_ok", True)
        for k in ("opencv", "color_math", "scipy", "mediapipe")
    )

    return results

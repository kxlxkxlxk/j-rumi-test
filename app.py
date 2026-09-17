"""
Stage 1 검증용 최소 FastAPI 앱 -- 2차 테스트 (mediapipe 제외 버전).

1차 테스트에서 번들 용량이 665MB로 Vercel의 제한(500MB)을 넘어서 배포에
실패했어요. mediapipe가 필요 없는 라이브러리(opencv-contrib-python,
matplotlib, sounddevice)까지 같이 설치하면서 용량을 크게 잡아먹는 게
유력한 원인이라, 이번엔 mediapipe를 완전히 빼고 나머지(opencv,
scikit-image, scipy, numpy, pillow)만 가지고 테스트해요.

이번에 배포가 성공하면: mediapipe가 범인이었다는 게 확정되고, "얼굴 인식
부분만 다른 가벼운 방법으로 바꾸면 Vercel에서도 충분히 돌아간다"는 뜻이
돼요. 이번에도 용량 초과가 나면: mediapipe 말고 다른 라이브러리(특히
scikit-image가 딸려오는 부가 라이브러리들)도 같이 손봐야 한다는 뜻이에요.
"""
from fastapi import FastAPI
import numpy as np

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok", "message": "jerumi API 검증 서버가 정상적으로 떠 있어요 (2차 테스트: mediapipe 제외)"}


@app.get("/api/test-deps")
def test_deps():
    """mediapipe를 제외한 나머지 무거운 라이브러리들을 하나씩 불러오고,
    아주 작은 실제 연산을 하나씩 돌려서 결과를 확인해요."""
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

    # --- scikit-image ---
    import skimage
    from skimage.color import rgb2lab
    lab = rgb2lab(np.array([[[0.5, 0.4, 0.3]]]))
    results["scikit_image"] = {
        "version": skimage.__version__,
        "test_ok": bool(lab.shape == (1, 1, 3)),
    }

    # --- scipy (Hungarian algorithm -- 카드 패치 매칭에 실제로 쓰는 함수) ---
    import scipy
    from scipy.optimize import linear_sum_assignment
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    row_ind, col_ind = linear_sum_assignment(cost)
    results["scipy"] = {
        "version": scipy.__version__,
        "test_ok": bool(len(row_ind) == 3),
    }

    # --- mediapipe: 이번 테스트에서는 의도적으로 제외 ---
    results["mediapipe"] = {
        "skipped": True,
        "note": "1차 테스트에서 용량 초과의 주범으로 의심되어 이번엔 제외하고 테스트해요",
    }

    # --- Pillow ---
    import PIL
    results["pillow"] = {"version": PIL.__version__}

    results["numpy_version"] = np.__version__
    results["all_ok"] = all(
        results[k].get("test_ok", True) for k in ("opencv", "scikit_image", "scipy")
    )

    return results

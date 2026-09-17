"""
Stage 1 검증용 최종 통합 테스트.

지금까지 확인된 것:
1. 용량 문제 -> Vercel의 "Large Functions" 기능을 켜서 해결 (500MB -> 5GB).
2. mediapipe가 화면 있는 컴퓨터용 opencv를 설치해서 나던 실행 에러
   (libxcb.so.1 없음) -> "화면 필요 없는" 버전(opencv-contrib-python-headless)을
   같이 설치해서 해결.

이제 실제 제루미 앱이 쓰는 모든 라이브러리(mediapipe, opencv, scikit-image,
scipy, numpy, pillow)를 한꺼번에 넣고 전부 정상 동작하는지 최종 확인해요.
여기까지 전부 통과하면 Stage 1(배포 가능성 검증)은 완료예요.
"""
from fastapi import FastAPI
import numpy as np

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok", "message": "Stage 1 최종 통합 테스트 서버가 정상적으로 떠 있어요"}


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

    # --- scikit-image ---
    import skimage
    from skimage.color import rgb2lab
    lab = rgb2lab(np.array([[[0.5, 0.4, 0.3]]]))
    results["scikit_image"] = {
        "version": skimage.__version__,
        "test_ok": bool(lab.shape == (1, 1, 3)),
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
        for k in ("opencv", "scikit_image", "scipy", "mediapipe")
    )

    return results

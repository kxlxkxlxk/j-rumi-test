"""
Stage 1 검증용 최소 FastAPI 앱.

목적: 실제 제루미 앱이 쓰는 무거운 라이브러리들(mediapipe, opencv, scikit-image,
scipy)을 Vercel의 Python 서버리스 함수 위에서 문제 없이 불러오고 실행할 수
있는지 -- 특히 번들 용량 제한(Python 함수는 압축 해제 기준 500MB)에 걸리지
않는지 -- 를 먼저 확인하기 위한 것. 아직 실제 얼굴 인식/색상 보정 로직은
옮기지 않았고, 각 라이브러리를 불러와서 아주 작은 연산 하나씩만 돌려봄으로써
"제대로 배포되고 동작하는지"만 확인해요.

이 프로젝트가 문제없이 배포되면, 다음 단계로 진짜 calibration.py /
skin_extraction.py / recommend.py 로직을 이 위에 그대로 옮겨서 실제 API로
만들면 돼요.
"""
from fastapi import FastAPI
import numpy as np

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok", "message": "jerumi API 검증 서버가 정상적으로 떠 있어요"}


@app.get("/api/test-deps")
def test_deps():
    """무거운 라이브러리들을 하나씩 불러오고, 아주 작은 실제 연산을 하나씩
    돌려서 결과를 확인해요. 여기까지 에러 없이 응답이 오면, 용량 제한이나
    플랫폼 호환성 문제는 없다는 뜻이에요."""
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

    # --- mediapipe ---
    # 실제 얼굴 인식(FaceLandmarker.create_from_options)은 모델 파일
    # (face_landmarker.task, 몇 MB)이 있어야 확인 가능해요. 여기서는 아직
    # 그 파일 없이, mediapipe 자체와 Tasks API를 문제없이 불러올 수 있는지만
    # 확인해요 -- 이것만으로도 라이브러리 용량/플랫폼 호환성 검증은 충분해요.
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision  # noqa: F401
    from mediapipe.tasks.python import core as mp_core  # noqa: F401
    results["mediapipe"] = {
        "version": mp.__version__,
        "tasks_api_importable": True,
        "note": "얼굴 인식 자체는 face_landmarker.task 모델 파일이 있어야 다음 단계에서 확인해요",
    }

    # --- Pillow ---
    import PIL
    results["pillow"] = {"version": PIL.__version__}

    results["numpy_version"] = np.__version__
    results["all_ok"] = all(
        results[k].get("test_ok", True) for k in ("opencv", "scikit_image", "scipy")
    )

    return results

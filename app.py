"""
Stage 1 검증용 최소 FastAPI 앱 -- 3차 테스트 (mediapipe 단독 용량 측정).

2차 테스트(mediapipe 제외)는 성공했어요 -- opencv, scikit-image, scipy,
numpy, pillow는 전혀 문제가 없다는 게 확인됐어요. 이번엔 반대로 mediapipe
"딱 하나만" 넣고 다른 건 다 빼서, mediapipe 혼자 용량을 얼마나 차지하는지
정확히 재보는 테스트예요.

이 결과에 따라 다음 전략이 갈려요:
- mediapipe 혼자 넣었는데도 이미 500MB에 가깝거나 넘으면 -> mediapipe
  자체를 가볍게 만드는 작업(불필요한 부속 라이브러리 제거)이 꼭 필요해요.
- mediapipe 혼자는 여유 있게 들어가면 -> 나머지 라이브러리들과 합쳤을 때
  용량을 어디서 좀 더 줄이면 되는지 계산할 수 있어요.
"""
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok", "message": "3차 테스트: mediapipe 단독 용량 측정용 서버예요"}


@app.get("/api/test-deps")
def test_deps():
    """mediapipe만 불러와서 문제없이 동작하는지 확인해요."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision  # noqa: F401
    from mediapipe.tasks.python import core as mp_core  # noqa: F401

    return {
        "mediapipe": {
            "version": mp.__version__,
            "tasks_api_importable": True,
        },
        "all_ok": True,
    }

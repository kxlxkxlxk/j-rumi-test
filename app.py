"""
Stage 2: 제루미 실제 분석 로직(calibration, skin_extraction, recommend)을
그대로 옮겨온 진짜 백엔드 API.

Stage 1에서 확인된 것들이 전부 반영돼 있어요:
- Vercel "Large Functions" 켜짐 (용량 500MB -> 5GB)
- mediapipe용 opencv는 화면 필요 없는 버전(opencv-contrib-python-headless)
- scikit-image 대신 순수 numpy로 직접 구현한 색상 변환 (src/color_match.py)

데이터(파운데이션 색상 목록)는 지금까지 쓰던 방식 그대로, GitHub 저장소의
data/foundation_db.json 파일을 읽고 씁니다 (환경변수 GITHUB_TOKEN,
GITHUB_REPO 설정 필요 -- 아래 read_shades() 참고). 환경변수가 아직
설정되지 않았으면 이 프로젝트에 같이 들어있는 로컬 복사본으로 대신
동작해요 (테스트용 fallback).
"""
import json
import os
import traceback
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

from src.recommend import recommend_foundation
from src import github_storage

app = FastAPI()

LOCAL_DB_PATH = Path(__file__).parent / "data" / "foundation_db.json"


def load_shades():
    """GitHub 저장소에서 읽기를 먼저 시도하고, 환경변수가 없거나 실패하면
    이 프로젝트에 같이 들어있는 로컬 복사본으로 대체해요."""
    try:
        shades, _sha = github_storage.read_shades()
        if shades:
            return shades, "github"
    except Exception:
        pass
    with open(LOCAL_DB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("shades", []), "local_fallback"


def _to_jsonable(obj):
    """numpy 배열/스칼라를 JSON으로 바로 못 바꾸는 문제를 해결하기 위한
    재귀 변환 헬퍼."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


@app.get("/")
def health():
    return {"status": "ok", "message": "jerumi 실제 분석 API가 정상적으로 떠 있어요"}


@app.get("/api/shades")
def api_shades():
    shades, source = load_shades()
    n_verified = sum(1 for s in shades if s.get("verified", True))
    return {
        "source": source,
        "total": len(shades),
        "n_verified": n_verified,
        "n_unverified": len(shades) - n_verified,
        "shades": shades,
    }


@app.post("/api/recommend")
async def api_recommend(
    file: UploadFile = File(...),
    has_card: bool = Form(True),
    use_verified_only: bool = Form(False),
    top_n: int = Form(3),
):
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        bgr_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            return JSONResponse(status_code=400, content={"success": False, "message": "이미지 파일을 읽지 못했어요"})

        shades, source = load_shades()
        shade_pool = [s for s in shades if s.get("verified", True)] if use_verified_only else shades
        if not shade_pool:
            return JSONResponse(status_code=400, content={"success": False, "message": "선택한 범위에 데이터가 없어요"})

        result = recommend_foundation(bgr_img, shade_pool, top_n=top_n, has_card=has_card)

        return _to_jsonable(
            {
                "success": result.success,
                "message": result.message,
                "corrected_lab": result.corrected_lab,
                "matches": result.matches,
                "debug": result.debug,
                "data_source": source,
                "shade_pool_size": len(shade_pool),
            }
        )
    except Exception as e:  # pragma: no cover - surfaced to the caller for debugging
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"서버 오류: {e}",
                "traceback": traceback.format_exc(),
            },
        )


# ---------------------------------------------------------------------------
# 간단한 테스트 페이지 -- 코딩 없이 브라우저에서 바로 사진 업로드해서
# /api/recommend 결과를 확인해볼 수 있어요. (실제 서비스용 화면이 아니라
# Stage 2 백엔드 로직 자체가 잘 동작하는지 확인하기 위한 용도예요.)
# ---------------------------------------------------------------------------
TEST_PAGE_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>jerumi API 테스트</title>
<style>
  body { font-family: sans-serif; max-width: 480px; margin: 40px auto; padding: 0 16px; }
  label { display: block; margin-top: 12px; font-weight: bold; }
  button { margin-top: 16px; padding: 10px 20px; font-size: 16px; }
  pre { background: #f5f5f5; padding: 12px; border-radius: 8px; white-space: pre-wrap; word-break: break-all; }
</style>
</head>
<body>
  <h2>jerumi API 테스트 페이지</h2>
  <form id="f">
    <label>사진 (카드+얼굴 또는 얼굴만)</label>
    <input type="file" name="file" accept="image/*" required>

    <label><input type="checkbox" name="has_card" checked> 색상카드 있음</label>
    <label><input type="checkbox" name="use_verified_only"> 정확측정 데이터만 사용</label>

    <button type="submit">추천 받기</button>
  </form>
  <p id="status"></p>
  <pre id="result"></pre>

<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData();
  fd.append('file', form.file.files[0]);
  fd.append('has_card', form.has_card.checked);
  fd.append('use_verified_only', form.use_verified_only.checked);
  fd.append('top_n', '3');
  document.getElementById('status').textContent = '분석 중... (몇 초 걸릴 수 있어요)';
  document.getElementById('result').textContent = '';
  try {
    const res = await fetch('/api/recommend', { method: 'POST', body: fd });
    const json = await res.json();
    document.getElementById('status').textContent = res.ok ? '완료' : '오류 (아래 내용 확인)';
    document.getElementById('result').textContent = JSON.stringify(json, null, 2);
  } catch (err) {
    document.getElementById('status').textContent = '요청 실패';
    document.getElementById('result').textContent = String(err);
  }
});
</script>
</body>
</html>
"""


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return TEST_PAGE_HTML

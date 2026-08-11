"""
build_portfolio_pptx.py — 포트폴리오 발표 자료 생성 (11슬라이드).

'정리파일/2-1. 개인 프로젝트 climate-forecast_송영상_ver1-1.pptx' 의 서식을
그대로 읽어 확장한다. 템플릿에서 추출한 사양:

  슬라이드 13.33 x 7.5in (16:9), Blank 레이아웃
  상단 바 #2D2D2D / 제목 맑은고딕 22pt Bold / 부제 11pt #5A5A5A
  소제목 15pt Bold #4A148C / 본문 12.5pt #262626
  결론 박스 #F4EFF7 + 좌측 액센트 바 #4A148C
  이미지 자리표시자 8.56, 1.46, 4.08 x 4.41in

**수치는 체크포인트에서 직접 읽는다.** 손으로 옮겨 적으면 재학습 때마다
낡는다 — 실제로 README 결과 요약이 초기 30일 수집본 기준에 멈춰 있다가
2026-08-11 점검에서 발견됐다. 같은 실수를 반복하지 않기 위해 load_metrics()
가 체크포인트를 열어 MAE·F1·기준선·게이트 배분을 가져온다.

실행: python build_portfolio_pptx.py     (torch 필요 — GPU 컨테이너에서 실행)
"""
import os

import torch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

TEMPLATE = "정리파일/2-1. 개인 프로젝트 climate-forecast_송영상_ver1-1.pptx"
OUT      = "정리파일/2-1. 개인 프로젝트 climate-forecast_송영상_ver1-2.pptx"

CKPT_TRI    = "./checkpoints/numerical_trichef.pt"
CKPT_ZONLY  = "./checkpoints/z_only_current_dataset.pt"

FONT     = "맑은 고딕"
INK      = RGBColor(0x2D, 0x2D, 0x2D)
SUB      = RGBColor(0x5A, 0x5A, 0x5A)
BODY     = RGBColor(0x26, 0x26, 0x26)
ACCENT   = RGBColor(0x4A, 0x14, 0x8C)
RULE     = RGBColor(0xE0, 0xE0, 0xE0)
KBOX_BG  = RGBColor(0xF4, 0xEF, 0xF7)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
TBL_HEAD = RGBColor(0xEC, 0xE4, 0xF2)
WARN     = RGBColor(0xB7, 0x3E, 0x0A)

BODY_L        = 0.85
BODY_T        = 1.45
BODY_W_FULL   = 11.65
BODY_W_NARROW = 7.05
IMG_BOX       = (8.56, 1.46, 4.08, 4.41)
KBOX_TOP      = 6.42


# ── 체크포인트에서 실측값 로드 ────────────────────────────────────────

def load_metrics():
    """3축·Z단독 체크포인트에서 발표에 쓸 수치를 읽는다."""
    m = {}
    tri = torch.load(CKPT_TRI, map_location="cpu", weights_only=True)
    m["tri_temp"]   = tri["val_temp_mae"]
    m["tri_precip"] = tri["val_precip_mae"]
    m["epoch"]      = tri.get("epoch")
    m["embed_dim"]  = tri.get("embed_dim", 64)
    d = tri.get("diagnostics") or {}
    m["w_re"], m["w_im"], m["w_z"] = d.get("w_re"), d.get("w_im"), d.get("w_z")
    m["cos_re_z"]  = d.get("cos_re_z_post")
    m["cos_re_im"] = d.get("cos_re_im_post")
    m["cos_im_z"]  = d.get("cos_im_z_post")
    m["extreme"]   = tri.get("extreme_metrics") or {}
    # 기준선 — train.py 수정 이후 체크포인트부터 저장된다
    m["naive_temp"]   = tri.get("val_temp_naive_mae")
    m["naive_precip"] = tri.get("val_precip_naive_mae")
    m["tri_mean"] = tri.get("mean")

    if os.path.exists(CKPT_ZONLY):
        z = torch.load(CKPT_ZONLY, map_location="cpu", weights_only=True)
        m["z_temp"]   = z["val_temp_mae"]
        m["z_precip"] = z["val_precip_mae"]
        m["z_extreme"] = z.get("extreme_metrics") or {}
        m["z_mean"] = z.get("mean")
        # 같은 데이터셋에서 학습됐는지 — 다르면 비교 자체가 성립하지 않는다
        m["same_dataset"] = (m["tri_mean"] == m["z_mean"])
    else:
        m["z_temp"] = m["z_precip"] = None
        m["same_dataset"] = None
    return m


def pct(new, old):
    """old 대비 new 의 변화율(%). 오차 지표이므로 음수가 개선이다."""
    if not old or not new:
        return "—"
    return f"{(new - old) / old * 100:+.1f}%"


def fnum(v, fmt="{:.3f}", dash="—"):
    return fmt.format(v) if isinstance(v, (int, float)) else dash


# ── 저수준 도우미 ────────────────────────────────────────────────────

def _txbox(slide, l, t, w, h):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return box, tf


def _run(para, text, size, bold=False, color=BODY, italic=False):
    r = para.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return r


def _rect(slide, l, t, w, h, fill=None, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                Inches(l), Inches(t), Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def new_slide(prs, title, tagline, runhead, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, 13.33, 0.06, fill=INK)
    _, tf = _txbox(slide, 0.5, 0.16, 10.6, 0.5)
    _run(tf.paragraphs[0], title, 21, bold=True, color=INK)
    _, tf = _txbox(slide, 0.64, 0.6, 11.1, 0.3)
    _run(tf.paragraphs[0], tagline, 11, color=SUB)
    _rect(slide, 0.5, 0.88, 12.3, 0.012, fill=RULE)
    _, tf = _txbox(slide, 9.6, 0.2, 3.3, 0.34)
    tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
    _run(tf.paragraphs[0], f"{runhead}   {page} / 12", 10, color=SUB)
    return slide


def subtitle(slide, text, top=1.02, left=0.59, width=12.24):
    _, tf = _txbox(slide, left, top, width, 0.37)
    _run(tf.paragraphs[0], text, 14, bold=True, color=ACCENT)


def kbox(slide, segs):
    _rect(slide, 0.59, KBOX_TOP, 12.24, 0.66, fill=KBOX_BG)
    _rect(slide, 0.59, KBOX_TOP, 0.05, 0.66, fill=ACCENT)
    _, tf = _txbox(slide, 0.78, KBOX_TOP + 0.08, 11.9, 0.54)
    p = tf.paragraphs[0]
    for seg, bold in segs:
        _run(p, seg, 12, bold=bold, color=ACCENT)


def img_placeholder(slide, title, caption, box=None):
    """이미지 삽입 자리 — 제목 + 설명글을 함께 표시해 무엇을 넣을지 명확히 한다."""
    l, t, w, h = box or IMG_BOX
    sh = _rect(slide, l, t, w, h, fill=None, line=ACCENT)
    sh.line.width = Pt(1.5)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(p, "[ 이미지 삽입 ]\n", 11, bold=True, color=ACCENT)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    _run(p2, title + "\n", 12, bold=True, color=INK)
    p3 = tf.add_paragraph()
    p3.alignment = PP_ALIGN.CENTER
    _run(p3, caption, 9.5, color=SUB)


def bullets(slide, items, left=BODY_L, top=BODY_T, width=BODY_W_FULL,
            size=11.5, bottom=None):
    avail = max(0.4, (bottom or KBOX_TOP - 0.08) - top)
    _, tf = _txbox(slide, left, top, width, avail)
    first = True
    for lvl, segs in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(4)
        p.line_spacing = 1.2
        mark = "• " if lvl == 0 else "– "
        _run(p, ("" if lvl == 0 else "    ") + mark, size,
             color=ACCENT if lvl == 0 else SUB)
        for text, bold in segs:
            _run(p, text, size, bold=bold, color=BODY)
    return tf


def table(slide, rows, left, top, width, col_w=None, size=9.5,
          head_size=9.5, row_h=0.28):
    n_r, n_c = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n_r, n_c, Inches(left), Inches(top),
                                Inches(width), Inches(row_h * n_r))
    tbl = gf.table
    tbl.first_row = True
    if col_w:
        for i, w in enumerate(col_w):
            tbl.columns[i].width = Inches(w)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        for ci, cell_text in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.01)
            cell.margin_bottom = Inches(0.01)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = TBL_HEAD if ri == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            cell.text_frame.word_wrap = True
            segs, buf, bold = [], "", False
            i = 0
            while i < len(cell_text):
                if cell_text[i:i+2] == "**":
                    if buf:
                        segs.append((buf, bold)); buf = ""
                    bold = not bold
                    i += 2
                else:
                    buf += cell_text[i]; i += 1
            if buf:
                segs.append((buf, bold))
            for text, b in segs:
                _run(p, text, head_size if ri == 0 else size,
                     bold=True if ri == 0 else b,
                     color=ACCENT if ri == 0 else BODY)
    return gf


def code_line(slide, text, left, top, width, size=12.5, height=0.38):
    box = _rect(slide, left, top, width, height, fill=KBOX_BG)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name = "Consolas"
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = ACCENT


def stat_row(slide, stats, left, top, width, gap=0.12, h=0.8):
    n = len(stats)
    w = (width - gap * (n - 1)) / n
    for i, (v, k) in enumerate(stats):
        x = left + i * (w + gap)
        _rect(slide, x, top, w, h, fill=KBOX_BG)
        _, tf = _txbox(slide, x + 0.08, top + 0.08, w - 0.16, h - 0.14)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        _run(p, v, 17, bold=True, color=ACCENT)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        _run(p2, k, 9, color=SUB)


# ── 본문 ─────────────────────────────────────────────────────────────

def build():
    if not os.path.exists(TEMPLATE):
        raise SystemExit(f"템플릿을 찾을 수 없습니다: {TEMPLATE}")
    M = load_metrics()

    prs = Presentation(TEMPLATE)
    xml_slides = prs.slides._sldIdLst
    for sld in list(xml_slides):
        prs.part.drop_rel(sld.rId)
        xml_slides.remove(sld)

    E = M["extreme"]
    f1 = lambda k: fnum(E.get(k, {}).get("f1"))
    pr = lambda k: fnum(E.get(k, {}).get("precision"), "{:.1%}")
    rc = lambda k: fnum(E.get(k, {}).get("recall"), "{:.1%}")
    npos = lambda k: f"{E.get(k, {}).get('n_pos', 0):,}건"

    H = ["", "개요", "문제·데이터", "아키텍처", "모델 설계", "모델 설계",
         "적용 효과", "트러블슈팅", "산출 로직", "성능 검증", "활용",
         "대시보드", "운영·한계"]

    # ══════════════════════════════════════ 1. 표지 · 전체 요약
    s = new_slide(prs, "환경 예보 (climate-forecast)",
                  "지상관측 12지점으로 +6시간 뒤를 계산한다 — 논문 수식을 다른 도메인에 이식하고, 실패까지 측정한 기록",
                  H[1], 1)
    stat_row(s, [
        (f"{M['tri_temp']:.2f}°C", "기온 MAE"),
        (f"{M['tri_precip']:.3f}mm", "강수 MAE"),
        (pct(M["tri_temp"], M["z_temp"]), "기온 개선\n(Z단독 대비)"),
        (pct(M["tri_precip"], M["z_precip"]), "강수 개선\n(Z단독 대비)"),
        (f1("heatwave"), "폭염 F1"),
        ("332,195", "학습 관측 건수"),
    ], BODY_L, 1.28, BODY_W_FULL, h=0.78)

    subtitle(s, "Problem → Solution → Result", top=2.22, left=BODY_L, width=7.0)
    table(s, [
        ["구분", "내용"],
        ["문제", "지상관측 12지점은 공간적으로 성기고, 단일 시점 스냅샷만으로는 '다가오는 변화'를 볼 수 없다"],
        ["해결", "Tri-CHEF 3축 융합 이식 — 주변(공간보간)·흐름(시간경향)·지금(수치센서)을 별개 축으로 분리 결합"],
        ["결과", f"Z축 단독 대비 기온 {pct(M['tri_temp'], M['z_temp'])} / 강수 {pct(M['tri_precip'], M['z_precip'])}, "
                 f"극한기상 3종 확률 헤드, Streamlit Cloud 상시 배포"],
    ], BODY_L, 2.62, BODY_W_NARROW, col_w=[0.85, 6.2], row_h=0.5)

    subtitle(s, "Project Overview", top=4.28, left=BODY_L, width=7.0)
    table(s, [
        ["항목", "내용"],
        ["데이터", "기상자료개방포털 ASOS 시간자료 332,195건 (2023-01~2026-08, 12개소)"],
        ["모델", f"3축 Tri-CHEF (embed_dim {M['embed_dim']}) + 동적 게이팅 + 회귀 2·분류 3 헤드"],
        ["파이프라인", "포털 CSV 백필 → GPU 학습 → CPU 서빙 → Actions 6h 자동 갱신"],
        ["환경", "PyTorch 2.6 / Docker / Streamlit Cloud (메모리 227.9MiB 실측)"],
        ["GitHub", "https://github.com/yssong01/climate-forecast"],
    ], BODY_L, 4.68, BODY_W_NARROW, col_w=[1.0, 6.05], row_h=0.28)

    img_placeholder(s, "① 대시보드 머리창",
                    "관측소명 + 현재 기온·강수·습도·기압\n4개 지표가 보이는 상단 영역",
                    box=(8.56, 2.22, 4.08, 3.9))
    kbox(s, [("논문 Tri-CHEF의 3축 융합 수식을 기후 예측에 이식하고, ", False),
             ("적용 효과를 동일 데이터·동일 분할에서 정량 검증", True), ("했다.", False)])

    # ══════════════════════════════════════ 2. 문제 정의 & 데이터
    s = new_slide(prs, "2-1. 문제 정의 & 데이터 | 33만 건을 어떻게 모았나",
                  "관측 데이터의 세 가지 공백, 그리고 API 차단을 만나 수집 경로를 바꾼 과정", H[2], 2)
    subtitle(s, "현장의 문제 — 관측 데이터가 가진 세 가지 공백")
    bullets(s, [
        (0, [("관측망은 성기다. ", True), ("전국 ASOS 주요 지점은 수십 개 수준이고, 지점 사이는 관측 공백이다.", False)]),
        (0, [("단일 시점 스냅샷은 방향을 모른다. ", True),
             ("지금 1005hPa인 것과 '떨어지는 중'인 1005hPa는 전혀 다른 상황인데, 레코드 한 줄엔 그 차이가 없다.", False)]),
        (0, [("극한기상은 희소하다. ", True),
             ("폭염·한파·황사는 전체의 몇 % 수준이라 정확도로 재면 '항상 아님'만 답해도 99%가 나온다.", False)]),
    ], top=1.42, size=11)
    subtitle(s, "수집 경로 — data.kma.go.kr → 파일셋 (시간자료, 2023~2026)", top=2.58)
    table(s, [
        ["포털 내 경로", "사용처"],
        ["데이터 → 기상관측 → 지상 → **종관기상관측(ASOS)**", "Z·Re·Im 3축의 원천 관측값 (332,195건 → 유효 쌍 331,595)"],
        ["데이터 → 기상관측 → 지상 → **황사관측(PM10)**", "황사 헤드 정답 라벨 (1시간 평균 150㎍/㎥ 기준)"],
        ["**날씨 이슈별 데이터** → 폭염·한파·황사·태풍", "폭염·한파 헤드 정답 라벨 (실제 발표된 특보 기록)"],
    ], BODY_L, 2.98, BODY_W_FULL, col_w=[5.5, 6.15], row_h=0.3)
    subtitle(s, "Problem → Solution → Result", top=4.06)
    table(s, [
        ["단계", "내용"],
        ["Problem", "apihub API 대량 수집이 **누적 약 9,800건에서 차단**(동시성 16 실측). 병렬화로 우회되지 않는 요청수 기준 상한"],
        ["Solution", "백필은 포털 CSV 일괄 다운로드로 전환(import_kma_fileset.py), API는 실시간 추론 전용으로 역할 분리"],
        ["Result", "30일치 → **3.6년치(332,195건)**로 확장. 계절 순환을 학습셋에 포함시켜 시간 분할 검증 문제까지 해소"],
    ], BODY_L, 4.46, BODY_W_FULL, col_w=[1.1, 10.55], row_h=0.42)
    kbox(s, [("원본 CSV는 ", False), ("EUC-KR", True),
             (" 인코딩이라 UTF-8로 읽으면 예외 없이 조용히 실패한다. 결측은 기본값으로 채우지 않고 ", False),
             ("레코드 전체를 버린다", True), (" — 결측이 실측으로 위장되는 것을 막기 위해.", False)])

    # ══════════════════════════════════════ 3. 아키텍처 & 논문 검토
    s = new_slide(prs, "2-2. 시스템 아키텍처 & 기후 AI 논문 검토",
                  "학습은 로컬 GPU, 서빙은 CPU 컨테이너, 갱신은 저장소가 원본 — 그리고 최신 기상 AI를 왜 안 썼나", H[3], 3)
    subtitle(s, "4개 경로의 역할 분리")
    table(s, [
        ["단계", "구성", "산출물"],
        ["① 수집", "포털 CSV 일괄(백필) + apihub API(실시간 15회/건)", "historical_data_1y.json (69MB)"],
        ["② 학습", "Docker GPU 컨테이너 · PyTorch 2.6", "numerical_trichef.pt (236KB)"],
        ["③ 서빙", "Streamlit + CPU torch · 메모리 **227.9MiB 실측**", "Streamlit Community Cloud"],
        ["④ 갱신", "GitHub Actions 6시간 주기 → main 되커밋 → 자동 재배포", "recent_window.json (0.6MB)"],
    ], BODY_L, 1.42, BODY_W_FULL, col_w=[0.95, 6.7, 4.0], row_h=0.3)
    subtitle(s, "검토한 최신 기상 AI와 기각 사유", top=3.0)
    table(s, [
        ["논문 / 모델", "핵심 아이디어", "적용 불가 사유"],
        ["**GraphCast · Pangu-Weather\nFourCastNet · ClimaX**",
         "GNN·Transformer·Fourier Neural Operator로 지구 격자 상태를 전파",
         "네 모델 모두 **ERA5 재분석 격자(0.25° 전지구)**를 입력 전제로 한다. 이 프로젝트 입력은 **12개 지점**이라 격자 자체가 없고, 보간해 넣으면 **보간 인공물을 학습**하게 된다"],
        ["**NowcastNet**", "이류–확산 물리 손실로 초단기 강수 예측",
         "이류·확산 항을 우리가 만든 IDW 보간장에 걸면 **자기 인공물을 물리 법칙으로 학습**한다"],
    ], BODY_L, 3.4, BODY_W_FULL, col_w=[2.6, 3.5, 5.55], row_h=0.72)
    subtitle(s, "대신 반영한 기상학·통계학 원리", top=4.98)
    bullets(s, [
        (0, [("IDW 역거리가중 공간 보간 ", True), ("— 지구통계학 고전 기법(power=2). Re축의 근거", False)]),
        (0, [("기압 경향(pressure tendency) ", True),
             ("— 기압 하강률은 저기압 접근의 고전적 선행지표, 기온 하강률은 복사냉각→이슬점 접근 과정. Im축의 근거", False)]),
        (0, [("퍼시스턴스 기준선 ", True), ("— 단기 예보 평가의 표준 기준선 T(t+L)≈T(t)", False),
             ("   ·   Hurdle 모델 ", True), ("— 0 과잉 분포를 '올까/얼마나'로 분해하는 계량경제학·수문학 표준", False)]),
    ], top=5.38, size=10.5)
    kbox(s, [("최신 모델을 쓰지 않은 것이 아니라, ", False),
             ("입력 데이터 구조가 전제를 만족하지 못함을 확인하고 기각", True), ("한 것이다.", False)])

    # ══════════════════════════════════════ 4. Tri-CHEF 이식 & 3축 설계
    s = new_slide(prs, "2-3. Tri-CHEF 이식 | 검색용 수식을 기후 예측으로",
                  "원 논문은 날씨와 무관한 '한국어 멀티모달 검색' 논문이다", H[4], 4)
    subtitle(s, "원 논문 — Complex-Hermitian Embedding Fusion for Korean Multimodal Retrieval (Zenodo, DOI 10.5281/zenodo.20034370)")
    bullets(s, [
        (0, [("문서·이미지·영상·음원을 한 번에 검색할 때 ", False), ("3개 인코더(SigLIP2·BGE-M3·DINOv2)", True),
             ("의 출력을 하나의 점수로 합치는 문제를 다룬다.", False)]),
        (0, [("가장 흔한 ", False), ("가중합", True), ("은 한 채널 점수가 크면 그 채널이 최종 점수를 독점한다. 대안이 ", False),
             ("Hermitian-style modulus", True),
             ("(제곱→합→제곱근) — 한 축이 다른 축을 상쇄해 지우지 못하고 세 축의 근거가 모두 남는다. 이것이 ", False),
             ("논문의 식 1번(Eq.1)", True), (".", False)]),
        (0, [("이 프로젝트가 빌린 것은 ", False), ("합치는 방식(수식 구조)뿐", True),
             ("이고, 세 축에 담는 데이터는 전부 새로 설계했다.", False)]),
    ], top=1.42, size=11)
    subtitle(s, "3축 설계 — 기준은 '단일 시점 스냅샷(Z)에서 유도할 수 없는 정보만 새 축에 담는다'", top=2.72)
    table(s, [
        ["축", "던지는 질문", "실제 입력", "Z에서 유도 불가능한 이유"],
        ["**Z** 지금 여기", "이 지점은 지금 어떤 상태인가?",
         "기온·강수·습도·풍속·풍향(sin/cos)·기압·강수형태·시각(sin/cos)·위도·경도 = **12차원**", "— (기준축)"],
        ["**Re** 주변", "주변 지역은 지금 어떤가?",
         "대상 관측소를 **제외한** 11개소의 같은 시각 실측을 IDW(power=2)로 ±3°(≈300km) 격자 **32×32×4채널**에 보간",
         "다른 지점 관측값은 이 지점 레코드에 물리적으로 없다"],
        ["**Im** 흐름", "어느 쪽으로 움직이는 중인가?",
         "같은 관측소의 **1·3·6시간 전 대비 변화량** × 4변수(기온·기압·습도·풍속) = **12차원**",
         "∂s/∂t 는 s(t) 하나만으로 계산 불가"],
    ], BODY_L, 3.12, BODY_W_FULL, col_w=[1.2, 2.1, 5.3, 3.05], row_h=0.6)
    subtitle(s, "설정창 지도가 관측소 '위치'만 쓰는 이유", top=5.15)
    bullets(s, [
        (0, [("위경도가 실제 모델 입력이다. ", True),
             ("record_to_vec 인덱스 10·11이 위도·경도 — 관측소마다 기후가 달라(강릉의 동해 영향, 제주의 해양성) 좌표 없이는 단일 모델이 12지점을 구분할 수 없다.", False)]),
        (0, [("Re축 IDW 가중치가 관측소 간 거리로 결정", True),
             ("되므로 '어느 이웃이 얼마나 반영되는가'를 지도가 설명한다. 외부 타일 서버 없이 Plotly 내장 지형만 사용(오프라인 렌더링).", False)]),
    ], top=5.55, size=10.5)
    kbox(s, [("Re축은 ", False), ("대상 관측소 자신의 값을 절대 넣지 않는다", True),
             (" — 넣으면 Z축과 중복정보가 되어 '정보량 0' 문제가 재발한다.", False)])

    # ══════════════════════════════════════ 5. 융합 수식 & 동적 게이팅
    s = new_slide(prs, "2-4. 융합 수식 & 동적 게이팅 | Eq.1을 기호 하나씩",
                  "s는 스칼라가 아니라 길이 64 벡터이고, 축 가중치는 매 출력마다 다시 계산된다", H[5], 5)
    code_line(s, "s_i  =  √( (w_Re · Re_i)²  +  (w_Im · Im_i)²  +  (w_Z · Z_i)² )",
              BODY_L, 1.4, BODY_W_FULL, size=13)
    table(s, [
        ["기호", "무엇인가", "이번 학습 결과"],
        ["Re, Im, Z", f"세 축을 각 인코더에 통과시켜 얻은 **숫자 {M['embed_dim']}개짜리 벡터**", f"축마다 {M['embed_dim']}개"],
        ["i", "그 벡터에서 몇 번째 자리인지 (3단계는 자리마다 따로 적용)", f"1 ~ {M['embed_dim']}"],
        ["w", "각 축을 얼마나 믿을지 — **Σw = 1** (softmax)",
         f"Re {fnum(M['w_re'])} / Im {fnum(M['w_im'])} / Z {fnum(M['w_z'])}"],
        ["s", "합쳐진 결과. 6개 헤드가 **공유**하는 입력", f"{M['embed_dim']}개"],
    ], BODY_L, 1.9, BODY_W_FULL, col_w=[1.2, 7.3, 3.15], row_h=0.3)
    subtitle(s, "3단계의 의미 · 동적 게이팅 구조", top=3.35)
    bullets(s, [
        (0, [("① 곱하기 ", True), ("— 신뢰도가 낮은 축은 여기서 작아진다   ", False),
             ("② 제곱 ", True), ("— 부호를 없애 상쇄를 막는다   ", False),
             ("③ 더하고 제곱근 ", True), ("— 직각삼각형 빗변 계산과 동일", False)]),
        (0, [("게이트: ", True), ("표준화된 Z축 12차원 → Linear(12→16) → GELU → Linear(16→3) → softmax → (w_Re, w_Im, w_Z)", False)]),
        (0, [("균등 초기화(1/3씩). ", True),
             ("논문 기본값 (1, 0.4, 0.2)로 두면 Re가 처음부터 1등이 되고 엔트로피 항이 그 순서를 증폭시켜, 어느 축이 유용한지 배우기도 전에 고착됐다.", False)]),
        (0, [("왜 softmax인가 — 스케일 자유도 제거. ", True),
             ("가중치가 자유로우면 k배 줄이고 헤드가 1/k로 상쇄해 어떤 크기 정규화도 무력화된다. Σw=1이면 정해진 예산을 나눠 갖게 되어 '어느 축에 얼마'가 실제 선택이 된다.", False)]),
        (0, [("논문 원식 s = √(A² + (αB)² + (φC)²) 에서 α = w_Im ÷ w_Re, φ = w_Z ÷ w_Re 로 환산", False)]),
    ], top=3.75, size=10.5)
    kbox(s, [("논문과 다르게 바꾼 지점 — 원 논문의 α·φ는 학습 후 고정 상수인데, ", False),
             ("그렇게 뒀더니 정보가 없는 축의 가중치가 오히려 커졌다", True), (" (α 0.4→0.665).", False)])

    # ══════════════════════════════════════ 6. Tri-CHEF 적용 효과 ★
    s = new_slide(prs, "2-5. Tri-CHEF 적용 효과 | 3축 vs Z축 단독",
                  "동일 데이터·동일 분할·동일 조기종료 기준에서 측정한 순수 아키텍처 효과", H[6], 6)
    subtitle(s, "핵심 비교 — Tri-CHEF 적용 전후 (이번 재학습 결과)")
    table(s, [
        ["구성", "기온 MAE", "강수 MAE", "폭염 F1", "한파 F1", "황사 F1"],
        ["**Z축 단독** (Re·Im 입력 차단)",
         f"{fnum(M['z_temp'], '{:.3f}')} °C", f"{fnum(M['z_precip'], '{:.4f}')} mm",
         fnum((M.get("z_extreme") or {}).get("heatwave", {}).get("f1")),
         fnum((M.get("z_extreme") or {}).get("coldwave", {}).get("f1")),
         fnum((M.get("z_extreme") or {}).get("dust", {}).get("f1"))],
        ["**3축 Tri-CHEF** (현행)",
         f"**{M['tri_temp']:.3f} °C**", f"**{M['tri_precip']:.4f} mm**",
         f1("heatwave"), f1("coldwave"), f1("dust")],
        ["개선폭", f"**{pct(M['tri_temp'], M['z_temp'])}**", f"**{pct(M['tri_precip'], M['z_precip'])}**", "—", "—", "—"],
    ], BODY_L, 1.42, BODY_W_FULL, col_w=[3.85, 1.75, 1.85, 1.4, 1.4, 1.4], row_h=0.38)
    _, tf = _txbox(s, BODY_L, 3.05, BODY_W_FULL, 0.8)
    p = tf.paragraphs[0]
    p.line_spacing = 1.2
    _run(p, "공정성 조건 — ", 10.5, bold=True, color=ACCENT)
    _run(p, "같은 데이터 캐시(API 재수집 없음) · 같은 SEED(42, 동일 분할) · 같은 조기 종료 기준(patience 20). "
            "Z단독은 아키텍처·용량은 그대로 두고 Re·Im 입력만 영벡터로 차단해 정보 기여도만 분리했다(인코더는 동결). "
            f"두 체크포인트의 특성 정규화 통계 일치 여부: {M.get('same_dataset')}. "
            "한계: 조건별 학습 길이가 다를 수 있다(각자 최고 검증 시점에서 저장).", 10.5, color=BODY)
    subtitle(s, "축 구성을 바꿔가며 측정한 이력 — 가장 큰 도약은 '데이터 교체'에서 나왔다", top=3.92)
    table(s, [
        ["단계", "Re축", "Im축", "기온 MAE", "강수 MAE", "황사 F1", "게이트 Re/Im/Z"],
        ["v3", "합성 위성", "MiniLM", "1.450", "0.2175", "0.258", ".05 / .14 / .81"],
        ["v4", "합성 위성", "MiniLM", "1.387", "0.2065", "0.257", "**.00** / .12 / .88"],
        ["**v5**", "**실측 보간**", "MiniLM", "1.279", "0.1696", "**0.661**", "**.52** / .04 / .44"],
        ["v7", "실측 보간", "**경향벡터**", "1.270", "0.1735", "0.576", ".37 / .21 / .42"],
    ], BODY_L, 4.32, BODY_W_FULL, col_w=[0.8, 1.75, 1.7, 1.5, 1.6, 1.3, 3.0], row_h=0.3)
    _, tf = _txbox(s, BODY_L, 5.92, BODY_W_FULL, 0.4)
    p = tf.paragraphs[0]
    _run(p, "※ v3~v7은 이전 세대 데이터셋 기준(비교는 그 안에서만 유효). ", 9.5, bold=True, color=WARN)
    _run(p, "합성 위성일 때 게이트 배분이 0.00이었다가 실측 교체 후 0.52로 뛴 것이 "
            "'모델 스스로 볼 가치가 있다고 판단한' 직접 증거다.", 9.5, color=BODY)
    kbox(s, [("아키텍처가 아니라 ", False), ("축에 담는 데이터를 바꿔서", True),
             (" 황사 F1이 0.257→0.661(2.6배)이 됐다.", False)])

    # ══════════════════════════════════════ 7. 트러블슈팅
    s = new_slide(prs, "2-6. 트러블슈팅 | 에러 없이 조용히 망가지던 것들",
                  "전부 '잘 도는 것처럼 보이는' 종류였다 — 실측으로만 잡을 수 있었다", H[7], 7)
    subtitle(s, "① 시뮬레이션 데이터의 정보량은 정확히 0이었다")
    table(s, [
        ["단계", "내용"],
        ["Problem", "초기 Re축(합성 위성)·Im축(MiniLM 예보문)은 **Z축 수치에서 결정론적으로 생성**한 시뮬레이션. 정보이론적으로 Z를 넘을 수 없는 구조인데 겉보기엔 '3축 멀티모달'로 보인다"],
        ["Solution", "추론 시 Re축을 **영벡터로 끄고** 출력 비교(deploy_ablation.py) + 게이트 배분 관측"],
        ["Result", "축을 꺼도 출력이 **소수점까지 완전히 동일**(기여도 0). 게이트 배분도 **0.000** — 모델이 이미 알고 있었다. "
                   "→ 실측 공간보간장·경향벡터로 교체 후 게이트 0.00→0.52, 황사 F1 0.257→0.661"],
    ], BODY_L, 1.42, BODY_W_FULL, col_w=[1.1, 10.55], row_h=0.45)
    subtitle(s, "② 학습을 조용히 망가뜨린 버그들", top=3.2)
    table(s, [
        ["증상", "원인", "조치"],
        ["강수 헤드가 학습 도중 영구 정지",
         "clamp(x,0,200)이 x<0에서 gradient **정확히 0** → 전 샘플 음수 수렴 후 회복 불가", "softplus 교체, 테스트로 회귀 고정"],
        ["강수가 사실상 학습되지 않음", "기온 손실(σ²≈수십)이 강수 손실(O(1))을 **1000배 압도**", "분산 정규화로 손실 스케일 정렬"],
        ["존재하지 않는 '개선'이 관측됨", "baseline을 **전체 데이터**에서 계산해 검증셋과 표본이 달랐다", "검증셋과 **동일 표본**에서 계산"],
        ["서울 관측 → 부산 6h 후와 페어링", "관측소 경계에서 쌍이 다른 관측소로 넘어갈 수 있는 인덱싱", "(관측소, 시각) 키 직접 조회"],
        ["**CI가 항상 초록불**", "테스트가 실패를 **출력만 하고 종료 코드 0** 반환", "sys.exit(0 if n_pass==n_all else 1)"],
        ["9시간 묵은 값을 '실시간'으로 표시", "Docker 기본 TZ가 UTC라 datetime.now()가 KST보다 9시간 뒤처짐", "ZoneInfo('Asia/Seoul') 명시"],
    ], BODY_L, 3.6, BODY_W_FULL, col_w=[3.1, 5.15, 3.4], row_h=0.4)
    kbox(s, [("배포 단계도 같은 성격 — Streamlit Secrets에 ", False), ("따옴표를 빼면", True),
             (" TOML 파싱이 실패해 값을 못 읽는데 화면은 대체값으로 정상처럼 보인다. ", False),
             ("API 호출 카운터를 화면에 노출", True), ("해 둔 덕에 원인을 좁혔다.", False)])

    # ══════════════════════════════════════ 8. 산출 로직 & 극한기상
    s = new_slide(prs, "2-7. +6시간 산출 로직 & 극한기상 3종",
                  "공유 표현 s → 6개 헤드 → 5가지 출력. 물리적 제약을 손실이 아니라 구조에 걸었다", H[8], 8)
    subtitle(s, "기온 — 퍼시스턴스 잔차(Δ)   ·   강수 — Hurdle 구조")
    code_line(s, "T(t+6) = T(t) + head_temp(s)          ŷ_precip = σ(z₁) × softplus(z₂)",
              BODY_L, 1.4, BODY_W_FULL, size=12.5)
    table(s, [
        ["요소", "역할", "선정 이유"],
        ["head_temp(s)", "지금 기온에서 얼마나 변할지(Δ)",
         "각 축이 단위 정규화되고 √(v²)가 부호를 지워 절대 기온 복원은 구조적으로 불리하다. 반면 T(t+6)≈T(t)는 강한 물리적 기준선 → 기준선을 주고 **편차만** 학습"],
        ["σ(z₁)", "비가 올 확률 (0~1)", "무강수가 대부분인 데이터에서 **정확한 0**을 연속적으로 표현하기 위함"],
        ["softplus(z₂)", "온다면 몇 mm (항상 > 0)", "강수량은 음수 불가. clamp는 dying-clamp 위험이 있어 배제. 후처리로 0.2mm 미만은 0으로 반올림"],
    ], BODY_L, 1.9, BODY_W_FULL, col_w=[1.5, 2.6, 7.55], row_h=0.44)
    subtitle(s, "극한기상 3종 — 라벨 정의와 판정선", top=3.62)
    table(s, [
        ["사건", "학습에 쓴 정답(라벨) 정의", "폴백(발표 기록 없을 때)", "판정선", "F1", "검증셋 양성"],
        ["폭염", "**기상청이 실제 발표한 폭염주의보 기록**", "발표기준 **33°C**를 대상 시각 기온에 적용", "0.84", f1("heatwave"), npos("heatwave")],
        ["한파", "**실제 발표된 한파주의보 기록**", "발표기준 **−12°C**를 대상 시각 기온에 적용", "0.85", f1("coldwave"), npos("coldwave")],
        ["황사", "**PM10 1시간 평균 ≥ 150㎍/㎥**(대기환경지수 '매우나쁨')", "없음 — 마스크로 손실에서 제외", "0.91", f1("dust"), npos("dust")],
    ], BODY_L, 4.02, BODY_W_FULL, col_w=[0.75, 4.3, 3.3, 0.95, 0.95, 1.4], row_h=0.44)
    _, tf = _txbox(s, BODY_L, 5.42, BODY_W_FULL, 0.9)
    p = tf.paragraphs[0]
    p.line_spacing = 1.15
    _run(p, "판정선을 0.5가 아닌 값으로 정한 절차 — ", 10.5, bold=True, color=ACCENT)
    _run(p, "① 검증셋을 보정용/평가용 50:50 무작위 분할(별개 시드) ② 보정용에서만 F1 최대화 임계값 탐색 "
            "③ **한 번도 보지 않은** 평가용에 적용해 유지 여부 확인. 실제로 **강수는 통과하지 못해**(순이득 0.01 미만) "
            "보수적 값을 유지했고 폭염·한파·황사만 채택했다.\n", 10.5, color=BODY)
    _run(p, "공식 황사주의보(400㎍/㎥ 2시간)를 안 쓴 이유 — ", 10.5, bold=True, color=WARN)
    _run(p, "보유 데이터의 6개 관측소에서 사례가 거의 없어 학습 불가. 폴백의 33°C/−12°C도 원래 일 최고/최저기온 "
            "기준이라 공식 특보와 같은 정의가 아님을 화면에 명시한다.", 10.5, color=BODY)
    kbox(s, [("물리적 제약(음수 불가·강한 기준선 존재)을 ", False), ("손실 함수가 아니라 아키텍처로", True), (" 걸었다.", False)])

    # ══════════════════════════════════════ 9. 성능 검증
    s = new_slide(prs, "2-8. 성능 검증 | 이긴 것과 진 것을 같이 쓴다",
                  "MAE는 기준선 없이는 좋고 나쁨을 말해주지 않는다", H[9], 9)
    subtitle(s, "회귀 성능 — 기준선 대비 (검증셋 20%, 학습 미사용)")
    table(s, [
        ["지표", "모델", "기준선(학습 없는 자명한 방법)", "판정"],
        ["기온 MAE", f"**{M['tri_temp']:.3f} °C**",
         f"퍼시스턴스 T(t+6)≈T(t) : {fnum(M['naive_temp'], '{:.3f}')} °C",
         f"**{pct(M['tri_temp'], M['naive_temp'])}**"],
        ["강수 MAE", f"**{M['tri_precip']:.4f} mm**",
         f"상시 0mm 예측 : {fnum(M['naive_precip'], '{:.4f}')} mm",
         f"**{pct(M['tri_precip'], M['naive_precip'])}**"],
    ], BODY_L, 1.42, BODY_W_NARROW, col_w=[1.3, 1.6, 2.85, 1.3], row_h=0.42)
    _, tf = _txbox(s, BODY_L, 2.85, BODY_W_NARROW, 1.15)
    p = tf.paragraphs[0]
    p.line_spacing = 1.15
    _run(p, "강수 결과를 가리지 않는다. ", 10.5, bold=True, color=WARN)
    _run(p, "대부분의 시각이 0mm라 '항상 0mm'라고만 답해도 MAE가 매우 낮게 나온다. 이 모델은 그 기준선을 "
            "**근소하게 넘지 못했다**. 원인은 구조적으로 추적됐다 — 극한기상 헤드가 3개로 늘면서 헤드들이 공유 표현 s를 "
            "두고 경쟁해 Hurdle의 무강수 억제력이 약해졌다(초기 −25.6%). EXTREME_BCE_WEIGHT를 1.0→0.3으로 낮추고 "
            "데이터를 3.6년으로 늘려 현재 수준까지 좁혔으며, **새 헤드를 추가할 때마다 강수 MAE를 재확인**하는 것이 "
            "이 프로젝트의 규약이 됐다. 강수는 MAE보다 '비가 올지 안 올지'로 보는 것이 실용적이다.", 10.5, color=BODY)
    subtitle(s, "극한기상 분류 (판정 임계값 0.5 기준)", top=4.05, left=BODY_L, width=7.0)
    table(s, [
        ["사건", "정밀도", "재현율", "F1", "검증셋 양성"],
        ["폭염", pr("heatwave"), rc("heatwave"), f1("heatwave"), npos("heatwave")],
        ["한파", pr("coldwave"), rc("coldwave"), f1("coldwave"), npos("coldwave")],
        ["황사", pr("dust"), rc("dust"), f1("dust"), npos("dust")],
    ], BODY_L, 4.45, BODY_W_NARROW, col_w=[1.0, 1.5, 1.5, 1.4, 1.65], row_h=0.31)
    _, tf = _txbox(s, BODY_L, 5.8, BODY_W_NARROW, 0.55)
    p = tf.paragraphs[0]
    p.line_spacing = 1.15
    _run(p, "희소 사건에 accuracy를 쓰지 않는다 — ", 10, bold=True, color=ACCENT)
    _run(p, "폭염이 1%면 '항상 아님'만 답해도 99%다. F1 상한은 표본 수가 좌우하며, 황사가 낮은 것은 배울 사례가 적기 때문. "
            "출력값 적중률(운영 사후 대조)은 기온 ±1.5°C, 강수 0.1mm 비/무비 일치 기준.", 10, color=BODY)
    img_placeholder(s, "② '성능 검증' 탭",
                    "회귀 MAE 카드 + 기준선 대비 증감\n+ 극한기상 F1 표 + 출력값 적중률")
    kbox(s, [("기준선을 체크포인트에 함께 저장하도록 train.py를 고쳐, ", False),
             ("대시보드가 기준선 대비 증감을 자동 표시", True), ("한다 — 숫자가 낡지 않는다.", False)])

    # ══════════════════════════════════════ 10. 활용 시나리오
    s = new_slide(prs, "2-9. 활용 시나리오 | 12개 도시에서 이 값들이 무엇에 쓰이나",
                  "출력값이 의사결정으로 연결되는 지점, 그리고 쓰면 안 되는 지점", H[10], 10)
    subtitle(s, "출력값별 활용 — 관측소가 놓인 12개 도시는 인구·산업이 밀집한 곳이라 리스크에 노출된 자산이 있다")
    table(s, [
        ["출력값", "무엇을 알려주나", "활용 예"],
        ["**기온**\n(현재 · +6h)", "6시간 뒤 온도 수준. 기준선(퍼시스턴스) 대비 유의미하게 개선된 유일한 회귀 지표",
         "건설·물류 등 **야외작업 온열질환 예방**(작업중지·휴식주기 사전 조정) · **태양광 발전효율** 예측(패널 온도가 오르면 효율이 떨어진다) · 냉난방 피크 수요 대비"],
        ["**강수**\n(비/무비)", "비가 올지 안 올지. 강수량 절대값이 아니라 **이진 판정**으로 쓴다",
         "야외작업·타설 등 **일정 조정** · 태양광 발전량 급감 대비 · 배수 설비 사전 점검"],
        ["**습도**", "체감온도(열지수)의 필수 구성 요소",
         "**기온만으로는 온열질환 위험을 못 잰다** — 습도가 높으면 땀 증발이 막혀 같은 기온에서도 위험이 급증한다. 기온과 함께 읽어야 하는 값"],
        ["**기압**", "기압 **하강률**은 저기압 접근의 고전적 선행지표 (Im축이 실제로 쓰는 신호)",
         "급변 감시 — 값 자체보다 '떨어지는 중인가'가 정보다"],
        ["**폭염 확률**", "폭염특보 수준 상황이 6시간 뒤 발생할 확률",
         "**건설현장 작업중지 판단** · 전력 피크 수요 대비 · 온열질환 감시 대상 선별"],
        ["**한파 확률**", "한파특보 수준 상황 확률",
         "**동파·결빙 대비** · 난방 수요 · 취약계층 보호 자원 배치"],
        ["**황사 확률**", "PM10 '매우나쁨'(150㎍/㎥) 수준 확률",
         "야외작업자 보호구 지급 · **태양광 패널 오염(soiling) → 청소 시점 판단** · 실내 공기질 설비 가동"],
    ], BODY_L, 1.42, BODY_W_FULL, col_w=[1.3, 3.9, 6.45], row_h=0.5)
    _, tf = _txbox(s, BODY_L, 5.12, BODY_W_FULL, 1.2)
    p = tf.paragraphs[0]
    p.line_spacing = 1.15
    _run(p, "쓰면 안 되는 지점 — 한계를 함께 명시한다.  ", 10.5, bold=True, color=WARN)
    _run(p, "① **강수량(mm) 절대값 기반 의사결정**: 기준선('항상 0mm')을 근소하게 넘지 못했다. "
            "비/무비 이진 판정으로만 쓴다.  ② **12개 지점 외 지역**: 지점 사이는 관측 공백이라 값이 없다.  "
            "③ **황사는 6개소만 PM10 라벨**(서울·수원·춘천·대구·전주·광주) — 나머지는 다른 지역 패턴을 옮긴 것이라 신뢰도가 낮다.  "
            "④ **공식 예보·특보를 대체하지 않는다**: 기상산업진흥법상 예보업은 등록 대상이며, 이 산출물은 모델 출력값이다.\n",
         10.5, color=BODY)
    _run(p, "지역 특성도 값 해석에 들어간다 — ", 10.5, bold=True, color=ACCENT)
    _run(p, "강릉(동해 영향)·제주(해양성)·대구(분지 고온)처럼 같은 기온이라도 지역마다 의미가 다르다. "
            "모델도 위도·경도를 입력으로 받아 이 차이를 학습한다.", 10.5, color=BODY)
    kbox(s, [("기후 리스크는 '지금 몇 도인가'가 아니라 ", False),
             ("'몇 시간 뒤 무엇을 해야 하는가'", True), ("로 답해야 실무에 쓰인다.", False)])

    # ══════════════════════════════════════ 11. 대시보드
    s = new_slide(prs, "2-10. 대시보드 | 블랙박스를 열어 보이는 화면 설계",
                  "선·막대는 전부 실측이고 ★ 별표 하나만 모델 출력값이다 — 근거를 화면에서 바로 답할 수 있어야 한다", H[11], 11)
    table(s, [
        ["탭", "무엇을 보여주나", "설계 원칙"],
        ["**출력값 추이**", "기온·강수·습도·기압 최근 72시간 실측 + **★ = +6h 출력값**. 기온만 마지막 실측점과 점선 연결(강수는 마지막이 0이면 선이 무의미)",
         "예측 대상(기온·강수)과 참고용(습도·기압)을 구분해 표기"],
        ["**극한 기상**", "폭염·한파·황사 확률 막대 + **빨간 판정선**(84/85/91%). 각 확률이 무엇을 배운 것인지 라벨 정의 표를 함께 표시",
         "'발령된 경보가 아니라 확률'임을 최상단에 명시(예보업 등록 대상 회피)"],
        ["**성능 검증**", "회귀 MAE·기준선 대비·극한기상 F1·적중률. F1은 임계값 0.5 기준이라 빨간 판정선과 다르다는 점까지 표기",
         "실패(강수 기준선 미달)를 숨기지 않고 경고로 표시"],
        ["**모델 구조**", "축 배분 막대(매 출력마다 재계산) + 3축 질문 표 + 수식 기호별 해설 + 원 논문 관계 + 학습된 표현의 코사인 유사도",
         "'실시간 데이터가 모델을 재학습시키지 않는다'를 명시"],
    ], BODY_L, 1.38, BODY_W_FULL, col_w=[1.5, 6.6, 3.55], row_h=0.52)
    IMG_W, IMG_H, IMG_T = 3.72, 2.05, 3.68
    img_placeholder(s, "③ 출력값 추이 탭", "2×2 그래프 + +6h 출력값 카드",
                    box=(0.85, IMG_T, IMG_W, IMG_H))
    img_placeholder(s, "④ 극한 기상 탭", "확률 막대 3종 + 빨간 판정선",
                    box=(0.85 + IMG_W + 0.24, IMG_T, IMG_W, IMG_H))
    img_placeholder(s, "⑤ 모델 구조 탭", "축 배분 막대 + 3축 표 + 융합 수식",
                    box=(0.85 + (IMG_W + 0.24) * 2, IMG_T, IMG_W, IMG_H))
    _, tf = _txbox(s, BODY_L, IMG_T + IMG_H + 0.12, BODY_W_FULL, 0.4)
    p = tf.paragraphs[0]
    _run(p, "축 배분이 매 출력마다 달라진다 — ", 10, bold=True, color=ACCENT)
    _run(p, f"이번 학습 결과 Re {fnum(M['w_re'])} / Im {fnum(M['w_im'])} / Z {fnum(M['w_z'])}. "
            f"학습된 표현의 코사인 유사도는 Re–Z {fnum(M['cos_re_z'], '{:.2f}')} · Re–Im {fnum(M['cos_re_im'], '{:.2f}')} · "
            f"Im–Z {fnum(M['cos_im_z'], '{:.2f}')} 로, '입력이 독립'과 '표현이 독립'은 다르다는 점도 화면에 공개한다.",
         10, color=BODY)
    kbox(s, [("해석 가능성이 Tri-CHEF를 택한 실질적 이유다 — ", False),
             ("축 단위로 기여도를 분해해 보여줄 수 있다", True), (".", False)])

    # ══════════════════════════════════════ 12. 배포·운영 & 한계
    s = new_slide(prs, "2-11. 배포·운영 & 한계 | 5분 갱신이 왜 부하가 아닌가",
                  "캐시 키를 시계에 묶어 호출량을 접속자 수와 무관하게 만들었다", H[12], 12)
    subtitle(s, "API 트래픽 설계 — 출력값 1건 = API 15회 (대상 1 + 이웃 11 + 과거 3)")
    table(s, [
        ["조건", "시간당", "일일", "판정"],
        ["캐시 없음 · 5분 갱신 · 1개 관측소", "180", "4,320", "위험"],
        ["캐시 없음 · 5분 갱신 · 접속자 3명", "540", "12,960", "**차단 지점(9,800) 초과**"],
        ["**캐시 적용**(관측소·관측시각 키)", "**15**", "**360**", "차단 지점의 3.7%"],
    ], BODY_L, 1.42, BODY_W_NARROW, col_w=[3.0, 0.9, 1.0, 2.15], row_h=0.3)
    bullets(s, [
        (0, [("근거는 물리적 사실 — ", False), ("ASOS는 매시 정각 관측 1회, 약 10분 후 공개", True),
             (". 같은 정시에 재조회해도 같은 값이다. 실측 검증: 같은 정시 2회 실행 → **증가 0건**", False)]),
        (0, [("2차 안전장치로 일일 상한(기본 3,000) + 사이드바 실측 표시. 합산 예산 ", False),
             ("650~1,000회/일(차단 지점의 7~10%)", True)]),
        (0, [("보안: 인증키를 .env(로컬) / Streamlit Secrets(배포) / GitHub Secrets(Actions) ", False),
             ("세 경로로 분리", True), (" — 저장소에 키가 남지 않고 브라우저에도 노출되지 않는다", False)]),
    ], top=2.7, width=BODY_W_NARROW, size=10.5, bottom=4.0)
    subtitle(s, "재학습 자동화 검토", top=4.02, left=BODY_L, width=7.0)
    table(s, [
        ["단계", "자동화", "근거"],
        ["신규 관측 수집", "**이미 자동**", "Actions 6h 주기 → main 되커밋 → Cloud 자동 재배포"],
        ["극한기상 라벨 갱신", "반자동", "날씨이슈·PM10은 **API 미제공** — 포털 수동 다운로드"],
        ["재학습 실행", "로컬 GPU", "Actions 무료 러너에 GPU 없음. **권장 주기 분기~반기**"],
        ["검증·배포 게이트", "가능", "지표가 나빠지면 배포를 막는 CI 게이트(설계 완료)"],
    ], BODY_L, 4.42, BODY_W_NARROW, col_w=[1.7, 1.2, 4.15], row_h=0.3)
    subtitle(s, "남은 한계와 다음 단계", top=1.02, left=8.3, width=4.5)
    bullets(s, [
        (0, [("강수 < 기준선 ", True), ("— 헤드 간 표현 경쟁이 원인. 완화만 했다", False)]),
        (0, [("Im축 성능 손해 ", True), ("— 실측 교체 후 F1 하락. 원칙(가짜 데이터 금지) 우선으로 유지", False)]),
        (0, [("PPO < 지도학습 ", True), ("— 경보 게이트에서 이벤트 214개로 표본 부족", False)]),
        (0, [("실위성 연결 ", True), ("— GK2A/Sentinel-2로 12지점 보간의 성긴 해상도 개선", False)]),
        (0, [("레이더 초단기 ", True), ("— 대류성 강수 예측 시계 확장, 강수 실패를 정면으로", False)]),
        (0, [("이상 탐지(QC) ", True), ("— Autoencoder 복원오차 기반 센서 이상 탐지를 앞단에", False)]),
        (0, [("물리 제약(PINN) ", True), ("— 실측 격자 확보가 선행되어야 함", False)]),
    ], left=8.3, top=1.42, width=4.5, size=10, bottom=6.3)
    kbox(s, [("이 프로젝트는 '만들고 나서 잘 됐다'가 아니라 ", False),
             ("잘못된 것을 실험으로 잡아낸 기록", True), ("에 가깝다.", False)])

    prs.save(OUT)
    print(f"생성 완료: {OUT}  (슬라이드 {len(prs.slides)}장)")
    print(f"  3축   기온 {M['tri_temp']:.4f} / 강수 {M['tri_precip']:.4f}")
    print(f"  Z단독 기온 {fnum(M['z_temp'], '{:.4f}')} / 강수 {fnum(M['z_precip'], '{:.4f}')}")
    print(f"  기준선 기온 {fnum(M['naive_temp'], '{:.4f}')} / 강수 {fnum(M['naive_precip'], '{:.4f}')}")
    print(f"  동일 데이터셋 여부: {M.get('same_dataset')}")


if __name__ == "__main__":
    build()

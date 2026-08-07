"""
rl_phase_filter.py — Phase 3-6: PPO 경보 게이트 (돌발 강수 조기 경보)

동결된 Tri-CHEF 예보 위에 얹는 이산 경보 정책이다.
    행동 a ∈ {0 정상, 1 주의, 2 경보}

왜 이 형태인가 — 원안(initial_plan.md)의 RL 설계를 그대로 쓰지 않은 이유:

  원안은 연속 오프셋 Δ 를 PPO 로 학습하고 보상을 R = −|예측+Δ−실측| 로 두었다.
  그런데 Δ 는 다음 상태에 영향을 주지 않고 ∂R/∂Δ 가 해석적으로 존재한다.
  즉 MDP 가 아니라 '보상이 미분 가능한 1-스텝 밴딧'이고, 이런 문제에서 PPO 는
  경사하강법보다 이론적으로 열등하다(같은 최적해를 더 큰 분산으로 느리게 찾음).

  RL 이 실제로 정당해지는 조건은 셋이고, 경보 문제는 셋을 모두 만족한다:
    (a) 비대칭 비용   — 미탐지 손실 ≫ 오경보 손실. 계단함수라 미분 불가
    (b) 이산 행동     — 정상/주의/경보
    (c) 행동 이력 의존 — 경보 피로(alarm fatigue). 최근에 경보를 많이 냈으면
                        다음 오경보의 비용이 커진다 → 보상이 과거 행동에
                        의존하므로 진짜 MDP 가 된다.

  (c)가 핵심이다. 고정 임계값이나 표본 단위 지도학습은 원리적으로 '지금까지
  경보를 몇 번 냈는가'를 반영할 수 없다. PPO 가 이겨야 할 지점도 정확히 거기다.
  이기지 못하면 그 결과를 그대로 보고한다 — 이 저장소의 다른 ablation 과 같다.

평가 설계 — 왜 시간 분할이 아니라 관측소 홀드아웃인가:

  수집 구간(2026-07-08 ~ 08-07)이 장마 종료 경계를 가로지른다. 관측소별로
  앞 70%/뒤 30% 로 자르면 검증 구간 평균 강수가 0.000mm 가 되어 이벤트가
  2,573개 중 2개(0.08%)로 붕괴한다 — 평가 자체가 성립하지 않는다.
  대신 관측소를 3개씩 4묶음으로 나눠 교차검증한다. 모든 fold 가 같은 기간을
  공유해 계절 편향이 없고, 8,546쌍 전량이 정확히 한 번씩 검증셋이 된다.
  덤으로 '학습에 없던 지점으로 일반화되는가'를 재게 되는데, 이는 관측망
  공백 지대 예측이라는 이 프로젝트의 문제의식과 직접 맞닿는다.

실행:
    python rl_phase_filter.py              # 4-fold 교차검증 전체
    python rl_phase_filter.py --fold 1     # 1개 fold 만 (빠른 확인)
"""
import argparse
import copy
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import collect_historical, record_to_vec, _parse_ts, STATION_NAMES
from satellite_collector import SimulatedSatelliteCollector
from text_collector import SimulatedTextCollector
from predict import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 이벤트 정의 ───────────────────────────────────────────────────
# '돌발 강수 시작' = 현재 사실상 무강수인데 +6시간 뒤 유의미한 강수.
# 이미 내리는 비가 계속되는 경우는 퍼시스턴스로 맞출 수 있어 경보 가치가 없다.
# 조기 경보가 실제로 필요한 것은 이 전이 구간이다 (전체의 2.50%, 214개).
EVENT_DRY_MAX = 0.1    # mm — 현재 '무강수' 판정 상한
EVENT_ON_MIN  = 1.0    # mm — +6h '유의미한 강수' 판정 하한

# ── 보상 (비대칭 운영 비용) ───────────────────────────────────────
# 조기 경보에서 미탐지는 침수·고립으로 이어지고 오경보는 신뢰도 손실에 그친다.
# 실무 비용비는 흔히 20:1 이상이라 그 비율을 그대로 반영한다.
R_MISS      = -20.0   # 이벤트인데 정상 판정 (최악)
R_HIT       = +10.0   # 이벤트에 경보
R_WATCH_HIT =  +4.0   # 이벤트에 주의 (부분 인정)
R_ALARM_FA  =  -1.0   # 비이벤트에 경보  × (1 + 피로)
R_WATCH_FA  =  -0.3   # 비이벤트에 주의  × (1 + 피로)

# 경보 피로 — 최근 경보가 쌓일수록 오경보 비용이 커진다.
# decay=0.9 이면 기억 길이는 약 10스텝이고, 상시 경보 정책의 정상상태 피로는
# 1/(1−0.9)=10 이 되어 오경보 비용이 11배로 뛴다. 이 항이 '무조건 경보'라는
# 자명한 해를 배제하고, 동시에 이 문제를 밴딧이 아닌 MDP 로 만든다.
FATIGUE_DECAY = 0.9
FATIGUE_ALARM = 1.0
FATIGUE_WATCH = 0.4

# ── 상태 ──────────────────────────────────────────────────────────
# 0~13 정적(동결 모델·관측에서 결정), 14 동적(에이전트가 제어)
FEAT_NAMES = [
    "theta",        #  0 논문 위상각 atan2(a_im, a_re) — 퇴화 확인용
    "w_re",         #  1 게이트: 위성
    "w_im",         #  2 게이트: 문서
    "w_z",          #  3 게이트: 수치
    "cos_re_im",    #  4 축 정렬도 — 게이트와 달리 실제 per-sample 축 정보
    "cos_re_z",     #  5
    "cos_im_z",     #  6
    "pred_precip",  #  7 모델 예측 강수 log1p
    "pred_dtemp",   #  8 모델이 퍼시스턴스에 얹은 보정량 Δ
    "cur_precip",   #  9 현재 강수 log1p
    "humidity",     # 10
    "pressure",     # 11
    "temp_vol",     # 12 최근 6시간 기온 변동성
    "pres_trend",   # 13 최근 6시간 기압 변화 — 급강하는 대류 발달 신호
    "fatigue",      # 14 경보 피로 (에이전트 제어) ← MDP 를 만드는 항
]
N_STATIC = 14
N_STATE  = len(FEAT_NAMES)
N_ACTION = 3

# 상태 집합 ablation — 어떤 특성이 관측소 홀드아웃에서 일반화를 깨는가.
#
# 게이트 입력(num_x)에 위도·경도가 들어 있으므로(train.record_to_vec 인덱스
# 10·11) w_re/w_im/w_z 와 그 재표현인 theta 는 사실상 관측소 식별자다. 기압·
# 습도도 고도·해양성에 따라 관측소마다 기준선이 다르다. 이런 '관측소 지문'을
# 상태에 넣으면 정책이 지점별 규칙을 외우고 새 지점에서 무너진다.
# minimal 은 지점 간 의미가 보존되는 물리량과 변화량만 남긴다.
STATE_SETS = {
    "full":    FEAT_NAMES,                      # 15 — 지문 포함
    "minimal": ["pred_precip", "pred_dtemp", "cur_precip",
                "temp_vol", "pres_trend", "fatigue"],   # 6 — 지점 불변
}


def state_cols(name: str) -> list:
    """상태 집합 이름 → 관측 벡터에서 뽑을 열 인덱스."""
    return [FEAT_NAMES.index(f) for f in STATE_SETS[name]]

# ── 학습 ──────────────────────────────────────────────────────────
EPISODE_LEN  = 168     # 1주 — 피로가 감쇠할 만큼 길고 신용할당은 가능한 길이
PPO_ITERS    = 400   # 60회에서는 학습 곡선이 아직 하강 중이었다 (약 0.25초/회)
PPO_EPOCHS   = 4
PPO_CLIP     = 0.2
PPO_LR       = 3e-4
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
ENT_COEF     = 0.01
VF_COEF      = 0.5
MINIBATCH    = 512
SUP_EPOCHS   = 300
SEED         = 42

CHECKPOINT  = "./checkpoints/numerical_trichef.pt"
RESULT_JSON = "./checkpoints/phase36_results.json"

# 4-fold 관측소 그룹 — fold 안에서 지리적으로 흩어지게 묶어, 한 fold 가
# 수도권처럼 한 기상 계통에 몰리지 않게 한다.
FOLDS = [
    ["108", "105", "143"],   # 서울·강릉·대구
    ["112", "133", "159"],   # 인천·대전·부산
    ["119", "131", "156"],   # 수원·청주·광주
    ["101", "146", "184"],   # 춘천·전주·제주
]


# ══ 1. 특성 추출 ══════════════════════════════════════════════════

def build_station_series(records: list, model, ckpt: dict,
                         device: str = DEVICE) -> dict:
    """
    관측소별 시계열 (정적 특성, 이벤트 라벨) 생성.

    동결된 Tri-CHEF 를 한 번 통과시켜 예측·게이트·축 진단을 뽑는다.
    RL 은 이 위에 얹히는 정책이므로 기반 모델은 학습하지 않는다.

    쌍 선별과 순서는 train.py 의 WeatherDataset 과 정확히 일치시킨다.
    text_collector.get_batch() 는 타임스탬프 목록이 완전히 같을 때만 캐시를
    재사용하고 다르면 캐시 파일을 덮어쓴다 — 관측소별로 쪼개 호출하면
    학습이 쓰는 8,546개짜리 임베딩 캐시가 조각난 캐시로 파괴되고, 다음
    train.py 실행에서 MiniLM 재추론이 발생한다.
    """
    L = ckpt["lead_hours"]
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std  = np.array(ckpt["std"],  dtype=np.float32)

    times = [_parse_ts(r["timestamp"]) for r in records]
    stns  = [str(r.get("stn")) for r in records]
    src_idx = [i for i in range(len(records) - L)
               if stns[i] == stns[i + L]
               and (times[i + L] - times[i]).total_seconds() == L * 3600]
    src = [records[i] for i in src_idx]
    tgt = [records[i + L] for i in src_idx]

    sat = SimulatedSatelliteCollector()
    txt = SimulatedTextCollector()

    # ── 동결 모델 순전파 ──────────────────────────────────────────
    vecs  = np.array([record_to_vec(r) for r in src], dtype=np.float32)
    x_num = torch.tensor((vecs - mean) / std, dtype=torch.float32).to(device)
    x_img = torch.tensor(sat.get_batch(src), dtype=torch.float32).to(device)
    x_txt = torch.tensor(txt.get_batch(src), dtype=torch.float32).to(device)

    preds, reports = [], defaultdict(list)
    with torch.no_grad():
        for b in range(0, len(src), 512):
            sl = slice(b, b + 512)
            preds.append(model(num_x=x_num[sl], img_x=x_img[sl],
                               txt_x=x_txt[sl]).cpu())
            for k, v in model.axis_report(x_num[sl], x_img[sl],
                                          x_txt[sl]).items():
                reports[k].append(v.cpu())
    pred = torch.cat(preds).numpy()
    rep  = {k: torch.cat(v).numpy() for k, v in reports.items()}

    # ── 이력 특성 — 원본 시계열에서 동일 관측소 구간만 되돌아본다 ──
    temps_all = np.array([r.get("temperature", 20.0) for r in records], np.float32)
    pres_all  = np.array([r.get("pressure",  1013.0) for r in records], np.float32)
    vol, trend = [], []
    for i in src_idx:
        lo = i
        for k in range(1, L + 1):
            if i - k < 0 or stns[i - k] != stns[i]:
                break
            lo = i - k
        vol.append(temps_all[lo:i + 1].std() if lo < i else 0.0)
        trend.append(pres_all[i] - pres_all[lo])

    cur_temp   = np.array([r.get("temperature",   20.0) for r in src], np.float32)
    cur_precip = np.array([r.get("precipitation",  0.0) for r in src], np.float32)
    tgt_precip = np.array([r.get("precipitation",  0.0) for r in tgt], np.float32)

    feat = np.stack([
        rep["theta"] / (np.pi / 2),
        rep["gate"][:, 0], rep["gate"][:, 1], rep["gate"][:, 2],
        rep["cos_re_im"], rep["cos_re_z"], rep["cos_im_z"],
        np.log1p(np.clip(pred[:, 1], 0, None)),
        (pred[:, 0] - cur_temp) / 5.0,          # 모델이 얹은 보정량 Δ
        np.log1p(cur_precip),
        (np.array([r.get("humidity",  50.0) for r in src], np.float32) - 70.0) / 30.0,
        (np.array([r.get("pressure", 1013.0) for r in src], np.float32) - 1005.0) / 10.0,
        np.array(vol,   np.float32) / 3.0,
        np.array(trend, np.float32) / 5.0,
    ], axis=1).astype(np.float32)

    event = ((cur_precip < EVENT_DRY_MAX) &
             (tgt_precip >= EVENT_ON_MIN)).astype(np.float32)

    # ── 관측소별 분리 (블록이 연속이고 블록 내부는 시간순) ────────
    order = np.array([stns[i] for i in src_idx])
    return {s: {"feat": feat[order == s], "event": event[order == s],
                "pred_precip": pred[order == s, 1].astype(np.float32)}
            for s in sorted(set(order.tolist()))}


def to_episodes(series: dict, stations: list):
    """관측소 시계열을 고정 길이 에피소드로 잘라 (E, T, F)·(E, T) 로 쌓는다."""
    feats, events = [], []
    for s in stations:
        if s not in series:
            continue
        f, e = series[s]["feat"], series[s]["event"]
        n = len(f) // EPISODE_LEN
        for k in range(n):                      # 나머지 구간은 버린다
            sl = slice(k * EPISODE_LEN, (k + 1) * EPISODE_LEN)
            feats.append(f[sl])
            events.append(e[sl])
    return np.stack(feats), np.stack(events)


# ══ 2. 환경 ═══════════════════════════════════════════════════════

class AlarmEnv:
    """
    벡터화 경보 환경 — 모든 에피소드를 같은 시각으로 동시에 진행한다.

    fatigue 는 에이전트의 과거 행동으로만 결정되는 상태 성분이고,
    보상이 여기에 곱해지므로 이 환경은 밴딧이 아니라 MDP 다.
    """

    def __init__(self, feats: np.ndarray, events: np.ndarray):
        self.feats, self.events = feats, events
        self.E, self.T, _ = feats.shape

    def reset(self) -> np.ndarray:
        self.t = 0
        self.fatigue = np.zeros(self.E, dtype=np.float32)
        return self._obs()

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [self.feats[:, self.t], self.fatigue[:, None]], axis=1
        ).astype(np.float32)

    def step(self, a: np.ndarray):
        ev, f = self.events[:, self.t], self.fatigue

        r = np.zeros(self.E, dtype=np.float32)
        hit = ev > 0.5
        r[hit & (a == 0)] = R_MISS
        r[hit & (a == 1)] = R_WATCH_HIT
        r[hit & (a == 2)] = R_HIT
        r[~hit & (a == 1)] = (R_WATCH_FA * (1.0 + f))[~hit & (a == 1)]
        r[~hit & (a == 2)] = (R_ALARM_FA * (1.0 + f))[~hit & (a == 2)]

        self.fatigue = FATIGUE_DECAY * f + np.where(
            a == 2, FATIGUE_ALARM, np.where(a == 1, FATIGUE_WATCH, 0.0)
        ).astype(np.float32)

        self.t += 1
        done = self.t >= self.T
        return (None if done else self._obs()), r, done


def rollout(env: AlarmEnv, act_fn) -> dict:
    """정책을 한 에피소드 끝까지 실행하고 보상·탐지 지표를 집계한다."""
    obs = env.reset()
    total_r, acts, evs = 0.0, [], []
    while True:
        a = act_fn(obs)
        obs, r, done = env.step(a)
        total_r += r.sum()
        acts.append(a)
        evs.append(env.events[:, env.t - 1])
        if done:
            break

    a = np.concatenate(acts)
    e = np.concatenate(evs) > 0.5
    warned = a >= 1                              # 주의 또는 경보

    hits   = int((e & warned).sum())
    misses = int((e & ~warned).sum())
    fas    = int((~e & warned).sum())
    n      = len(a)

    return {
        "reward_per_step": total_r / n,
        "POD": hits / max(hits + misses, 1),
        "FAR": fas / max(hits + fas, 1),
        "CSI": hits / max(hits + misses + fas, 1),
        "alarm_rate": float((a == 2).mean()),
        "watch_rate": float((a == 1).mean()),
        "hits": hits, "misses": misses, "false_alarms": fas,
        "n_events": int(e.sum()), "n_steps": n,
    }


# ══ 3. 정책 ═══════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    공유 트렁크 + 정책 헤드(3) + 가치 헤드(1).

    입력 정규화를 모듈 안에 버퍼로 넣는다. 상태 특성의 스케일이 제각각이라
    (게이트 0~1, 기압 추세 ±수, log1p 강수 0~0.3) 원시값을 그대로 tanh 에
    넣으면 분산이 큰 특성이 트렁크를 포화시켜, 정작 판별력이 있는 예측 강수
    같은 작은 특성이 묻힌다. 실측에서 정규화 없이는 PPO 가 값싼 '주의'를
    26% 남발하는 해에 갇혔다. 기준선은 원시 특성을 그대로 쓰므로 이 정규화가
    PPO 에만 유리하게 작용하지 않는다 — 임계값 탐색은 스케일 불변이다.
    """

    def __init__(self, obs_mean: np.ndarray, obs_std: np.ndarray,
                 cols: list, hidden: int = 64):
        super().__init__()
        # 환경은 항상 15차원 관측을 내보내고, 정책만 자기 상태 집합으로
        # 좁혀 본다 — 기준선은 전체 관측을 그대로 쓰므로 비교가 흔들리지 않는다.
        self.register_buffer("cols", torch.tensor(cols, dtype=torch.long))
        self.register_buffer("obs_mean", torch.tensor(obs_mean))
        self.register_buffer("obs_std",  torch.tensor(obs_std))
        self.trunk = nn.Sequential(
            nn.Linear(len(cols), hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, N_ACTION)
        self.v  = nn.Linear(hidden, 1)
        with torch.no_grad():
            # 정책은 거의 균등하게 출발시키되, 초기 무작위 경보로 피로가
            # 치솟아 학습 초반이 망가지지 않도록 '정상' 쪽으로 살짝 기울인다.
            self.pi.weight.mul_(0.01)
            self.pi.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))

    def forward(self, x):
        x = x[:, self.cols]
        h = self.trunk((x - self.obs_mean) / self.obs_std)
        return self.pi(h), self.v(h).squeeze(-1)


def threshold_act(score_fn, lo: float, hi: float):
    """점수 → 행동. 고정 임계값·지도학습 기준선이 공유하는 판정부."""
    def act(obs):
        s = score_fn(obs)
        return np.where(s >= hi, 2, np.where(s >= lo, 1, 0)).astype(np.int64)
    return act


def tune_thresholds(env: AlarmEnv, score_fn, grid: np.ndarray) -> tuple:
    """
    학습 fold 보상을 최대로 만드는 (lo, hi) 격자 탐색.

    기준선에도 PPO 와 완전히 같은 목적함수(피로 포함 누적 보상)를 주어야
    비교가 공정하다. 기준선을 MAE 나 정확도로 튜닝해 놓고 보상으로 평가하면
    이겨도 의미가 없다.
    """
    best, best_r = (grid[0], grid[-1]), -np.inf
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            r = rollout(env, threshold_act(score_fn, lo, hi))["reward_per_step"]
            if r > best_r:
                best_r, best = r, (float(lo), float(hi))
    return best


def train_supervised(feats: np.ndarray, events: np.ndarray,
                     device: str = DEVICE) -> nn.Module:
    """
    표본 단위 지도학습 분류기 — 가장 강한 근시안 기준선.

    정적 특성 14개만 본다. fatigue 를 넣어도 의미가 없는데, 지도학습은
    '이 표본이 이벤트인가'를 맞추도록 학습될 뿐 '지금 경보를 내는 것이
    이후 비용까지 포함해 이득인가'를 최적화하지 않기 때문이다. 이 구조적
    한계가 PPO 와의 유일한 본질적 차이가 되도록 두는 것이 설계 의도다.
    """
    x = torch.tensor(feats.reshape(-1, N_STATIC)).to(device)
    y = torch.tensor(events.reshape(-1)).to(device)

    net = nn.Sequential(
        nn.Linear(N_STATIC, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    # 이벤트가 2.5% 뿐이라 가중치 없이는 상시 음성으로 붕괴한다.
    pos_w = torch.tensor([(y < 0.5).sum() / max((y > 0.5).sum(), 1)]).to(device)

    net.train()
    for _ in range(SUP_EPOCHS):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            net(x).squeeze(-1), y, pos_weight=pos_w
        )
        loss.backward()
        opt.step()
    net.eval()
    return net


def compute_gae(rew: np.ndarray, val: np.ndarray,
                gamma: float = GAMMA, lam: float = GAE_LAMBDA) -> np.ndarray:
    """
    GAE(λ) 이점 추정. rew/val: (T, E) → 반환 (T, E).

    에피소드는 고정 길이이고 끝이 곧 종료 상태이므로 마지막 스텝의
    부트스트랩 값은 0 이다. 여기서 val[T-1] 을 이어 쓰면 존재하지 않는
    미래 보상을 끌어와 마지막 구간의 이점이 계통적으로 부풀려진다.
    """
    adv = np.zeros_like(rew)
    last = np.zeros(rew.shape[1], dtype=np.float32)
    for t in reversed(range(rew.shape[0])):
        next_v = val[t + 1] if t + 1 < rew.shape[0] else np.zeros_like(last)
        delta = rew[t] + gamma * next_v - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


def train_ppo(env: AlarmEnv, sel_env: AlarmEnv = None, device: str = DEVICE,
              iters: int = PPO_ITERS, state_set: str = "full",
              verbose: bool = True) -> ActorCritic:
    """
    클리핑 대리목적 + GAE(λ) PPO.

    sel_env 를 주면 주기적으로 평가해 보상이 가장 좋은 반복을 되돌린다.
    학습 보상은 반복마다 크게 출렁이므로 '마지막 반복'을 그냥 쓰면 정책의
    실력이 아니라 뽑기 운을 재게 된다. 선택셋은 검증 관측소와 겹치지 않는
    별도 관측소여야 한다 — 그렇지 않으면 검증셋에 맞춰 고른 것이 된다.
    """
    cols = state_cols(state_set)
    # 정규화 통계는 학습 fold 에서만 뽑는다 — 검증 관측소 분포를 보면 누설이다.
    # 피로(마지막 열)는 이미 O(1) 이고 환경이 만드는 값이라 그대로 둔다.
    flat = env.feats.reshape(-1, N_STATIC)
    full_mean = np.concatenate([flat.mean(0), [0.0]]).astype(np.float32)
    full_std  = np.concatenate([flat.std(0) + 1e-6, [1.0]]).astype(np.float32)

    net = ActorCritic(full_mean[cols], full_std[cols], cols).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=PPO_LR)
    T, E = env.T, env.E
    best_sel, best_state = -np.inf, None

    for it in range(1, iters + 1):
        # ── 롤아웃 ────────────────────────────────────────────────
        obs_buf = np.zeros((T, E, N_STATE), dtype=np.float32)
        act_buf = np.zeros((T, E), dtype=np.int64)
        lgp_buf = np.zeros((T, E), dtype=np.float32)
        val_buf = np.zeros((T, E), dtype=np.float32)
        rew_buf = np.zeros((T, E), dtype=np.float32)

        obs = env.reset()
        with torch.no_grad():
            for t in range(T):
                ot = torch.tensor(obs).to(device)
                logits, value = net(ot)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()

                obs_buf[t], act_buf[t] = obs, a.cpu().numpy()
                lgp_buf[t] = dist.log_prob(a).cpu().numpy()
                val_buf[t] = value.cpu().numpy()

                obs, r, done = env.step(act_buf[t])
                rew_buf[t] = r

        # ── GAE ───────────────────────────────────────────────────
        adv = compute_gae(rew_buf, val_buf)
        ret = adv + val_buf

        b_obs = torch.tensor(obs_buf.reshape(-1, N_STATE)).to(device)
        b_act = torch.tensor(act_buf.reshape(-1)).to(device)
        b_lgp = torch.tensor(lgp_buf.reshape(-1)).to(device)
        b_adv = torch.tensor(adv.reshape(-1)).to(device)
        b_ret = torch.tensor(ret.reshape(-1)).to(device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ── 갱신 ──────────────────────────────────────────────────
        n = b_obs.size(0)
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, MINIBATCH):
                mb = perm[s:s + MINIBATCH]
                logits, value = net(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                lgp = dist.log_prob(b_act[mb])

                ratio = torch.exp(lgp - b_lgp[mb])
                a_mb = b_adv[mb]
                pg = -torch.min(
                    ratio * a_mb,
                    torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * a_mb,
                ).mean()
                vf = F.mse_loss(value, b_ret[mb])
                ent = dist.entropy().mean()

                opt.zero_grad()
                (pg + VF_COEF * vf - ENT_COEF * ent).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        # ── 선택셋 기준 최고 반복 보존 ────────────────────────────
        if sel_env is not None and (it % 5 == 0 or it == iters):
            r_sel = rollout(sel_env, ppo_act(net, device))["reward_per_step"]
            if r_sel > best_sel:
                best_sel = r_sel
                best_state = copy.deepcopy(net.state_dict())

        if verbose and (it % 10 == 0 or it == 1):
            # 경보율을 함께 찍는다 — '항상 정상'(경보율 0)이 강한 국소최적이라
            # 보상만 봐서는 학습 중인지 그 해로 붕괴했는지 구분되지 않는다.
            print(f"      PPO[{state_set:<7}] it{it:3d} | "
                  f"보상/스텝 {rew_buf.sum()/(T*E):+.4f} "
                  f"| 경보율 {(act_buf == 2).mean():.3f} "
                  f"| 주의율 {(act_buf == 1).mean():.3f} "
                  f"| 엔트로피 {ent.item():.3f}")

    if best_state is not None:
        net.load_state_dict(best_state)
        if verbose:
            print(f"      PPO[{state_set:<7}] 선택셋 최고 반복 복원 "
                  f"(보상 {best_sel:+.4f})")
    return net


def ppo_act(net: "ActorCritic", device: str = DEVICE):
    """평가 시에는 표집하지 않고 최빈 행동을 쓴다 (결정론적)."""
    def act(obs):
        with torch.no_grad():
            logits, _ = net(torch.tensor(obs).to(device))
        return logits.argmax(dim=-1).cpu().numpy()
    return act


# ══ 4. 교차검증 ═══════════════════════════════════════════════════

def run_fold(series: dict, val_stns: list, fold_id: int,
             iters: int = PPO_ITERS, verbose: bool = True) -> dict:
    """
    한 fold — 관측소를 세 겹으로 나눈다.

        검증 3개 : 최종 평가. 어떤 방법도 학습·선택에 쓰지 않는다.
        선택 2개 : 임계값·PPO 반복 등 '고르는' 결정에만 쓴다.
        적합 7개 : 파라미터를 맞추는 데만 쓴다.

    선택 겹을 따로 두는 이유는 PPO 때문이다. 반복마다 보상이 출렁여서
    마지막 반복을 그냥 쓰면 정책 실력이 아니라 뽑기 운이 찍힌다. 다만
    PPO 에만 선택셋을 주면 데이터·보정 기회가 달라져 비교가 깨지므로,
    기준선의 임계값 탐색도 같은 선택셋에서 하고 적합 겹도 똑같이 7개로
    맞춘다. 그래야 남는 차이가 방법의 차이가 된다.
    """
    all_stns = sorted(series.keys())
    tr_stns = [s for s in all_stns if s not in val_stns]
    # 선택 겹은 목록에서 떨어진 두 지점을 골라 한 기상 계통에 몰리지 않게 한다
    sel_stns = [tr_stns[1], tr_stns[6]]
    fit_stns = [s for s in tr_stns if s not in sel_stns]

    fit_f, fit_e = to_episodes(series, fit_stns)
    sel_f, sel_e = to_episodes(series, sel_stns)
    va_f,  va_e  = to_episodes(series, val_stns)
    fit_env, sel_env, va_env = (AlarmEnv(fit_f, fit_e),
                                AlarmEnv(sel_f, sel_e),
                                AlarmEnv(va_f, va_e))

    if verbose:
        nm = lambda ss: "·".join(STATION_NAMES.get(s, s) for s in ss)
        print(f"\n  [fold {fold_id}] 검증 {nm(val_stns)}")
        print(f"      적합 {nm(fit_stns)} — {fit_f.shape[0]}에피소드 / "
              f"이벤트 {int(fit_e.sum())}개")
        print(f"      선택 {nm(sel_stns)} — {sel_f.shape[0]}에피소드 / "
              f"이벤트 {int(sel_e.sum())}개")
        print(f"      검증 {nm(val_stns)} — {va_f.shape[0]}에피소드 / "
              f"이벤트 {int(va_e.sum())}개")

    policies = []

    # ── 기준선 1: 항상 정상 ───────────────────────────────────────
    policies.append(("항상 정상", lambda o: np.zeros(len(o), np.int64)))

    # ── 기준선 2: 고정 임계값 (모델 예측 강수 하나만) ─────────────
    precip_score = lambda o: o[:, FEAT_NAMES.index("pred_precip")]
    lo, hi = tune_thresholds(sel_env, precip_score, np.linspace(0.0, 1.2, 25))
    policies.append(("고정 임계값", threshold_act(precip_score, lo, hi)))

    # ── 기준선 3: 지도학습 분류기 (정적 14특성) ───────────────────
    clf = train_supervised(fit_f, fit_e)

    def clf_score(o):
        with torch.no_grad():
            x = torch.tensor(o[:, :N_STATIC]).to(DEVICE)
            return torch.sigmoid(clf(x).squeeze(-1)).cpu().numpy()

    lo2, hi2 = tune_thresholds(sel_env, clf_score, np.linspace(0.05, 0.95, 19))
    policies.append(("지도학습 분류기", threshold_act(clf_score, lo2, hi2)))

    # ── PPO — 상태 집합 ablation ──────────────────────────────────
    # 적합 보상만 보면 full 이 이긴다(관측소 지문을 외울 수 있으니 당연하다).
    # 두 조건을 같은 검증 관측소에서 나란히 재야, 그 이득이 암기였는지
    # 일반화였는지 구분된다.
    for name in ("full", "minimal"):
        net = train_ppo(fit_env, sel_env=sel_env, iters=iters,
                        state_set=name, verbose=verbose)
        policies.append((f"PPO ({name})", ppo_act(net)))

    # 모든 정책을 적합·검증 양쪽에서 동일하게 재어 일반화 격차를 드러낸다.
    out = {}
    for nm_, fn in policies:
        m = rollout(va_env, fn)
        t = rollout(fit_env, fn)
        m["train_reward"] = t["reward_per_step"]
        m["train_warn_rate"] = t["alarm_rate"] + t["watch_rate"]
        out[nm_] = m
    return out


def print_table(title: str, res: dict) -> None:
    """
    학습 보상을 검증 보상 옆에 나란히 찍는다 — 이 문제에서 둘의 격차가
    곧 관측소 지문 암기량이고, 검증 열만 봐서는 왜 졌는지 알 수 없다.
    """
    W = 96
    print(f"\n{'='*W}")
    print(f" {title}")
    print(f"{'='*W}")
    print(f"{'정책':<18} {'학습보상':>9} {'검증보상':>9} {'격차':>7} "
          f"{'POD':>6} {'FAR':>6} {'CSI':>6} {'경보율':>7} "
          f"{'정탐':>5} {'미탐':>5} {'오경보':>7}")
    print("-" * W)
    for k, m in res.items():
        tr = m.get("train_reward", float("nan"))
        va = m["reward_per_step"]
        print(f"{k:<18} {tr:+9.4f} {va:+9.4f} {tr - va:7.3f} "
              f"{m['POD']:6.3f} {m['FAR']:6.3f} {m['CSI']:6.3f} "
              f"{m['alarm_rate']+m['watch_rate']:7.3f} "
              f"{m['hits']:5d} {m['misses']:5d} {m['false_alarms']:7d}")
    print("-" * W)
    best = max(res, key=lambda k: res[k]["reward_per_step"])
    print(f" 검증 최고 보상: {best}  ({res[best]['reward_per_step']:+.4f})")
    print(f"{'='*W}")


def main():
    ap = argparse.ArgumentParser(description="Phase 3-6 PPO 경보 게이트")
    ap.add_argument("--fold", type=int, default=None,
                    help="특정 fold 만 실행 (1~4). 생략 시 4-fold 전체")
    ap.add_argument("--iters", type=int, default=PPO_ITERS,
                    help=f"PPO 반복 수 (기본 {PPO_ITERS})")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 84)
    print(" Phase 3-6 — PPO 경보 게이트 (돌발 강수 조기 경보)")
    print("=" * 84)
    print(f" 디바이스: {DEVICE}")
    print(f" 이벤트  : 현재 강수 < {EVENT_DRY_MAX}mm  →  +6h 강수 ≥ {EVENT_ON_MIN}mm")
    print(f" 보상    : 미탐지 {R_MISS} | 정탐 {R_HIT:+} | 주의정탐 {R_WATCH_HIT:+} | "
          f"오경보 {R_ALARM_FA}×(1+피로)")
    print(f" 피로    : decay {FATIGUE_DECAY} — 상시 경보 시 정상상태 "
          f"{FATIGUE_ALARM/(1-FATIGUE_DECAY):.0f} (오경보 비용 11배)")

    if not os.path.exists(args.checkpoint):
        print(f"\n[ERROR] 체크포인트가 없습니다: {args.checkpoint}")
        print("        먼저 python train.py 를 실행하세요.")
        return

    model, ckpt = load_model(args.checkpoint, DEVICE)
    print(f"\n동결 모델 로드 — 기온 MAE {ckpt['val_temp_mae']:.4f}°C / "
          f"강수 MAE {ckpt['val_precip_mae']:.4f}mm (+{ckpt['lead_hours']}h)")

    records = collect_historical()
    print("동결 모델 순전파 — 상태 특성 추출 중...")
    series = build_station_series(records, model, ckpt)

    n_ev = sum(int(v["event"].sum()) for v in series.values())
    n_st = sum(len(v["event"]) for v in series.values())
    print(f"  관측소 {len(series)}개 | 표본 {n_st}개 | "
          f"이벤트 {n_ev}개 ({n_ev/n_st*100:.2f}%)")

    folds = [args.fold] if args.fold else list(range(1, len(FOLDS) + 1))
    all_res = {}
    for fid in folds:
        res = run_fold(series, FOLDS[fid - 1], fid, iters=args.iters)
        all_res[fid] = res
        names = "·".join(STATION_NAMES.get(s, s) for s in FOLDS[fid - 1])
        print_table(f"fold {fid} — 검증 {names}", res)

    # ── 전 fold 합산 ──────────────────────────────────────────────
    if len(all_res) > 1:
        agg = {}
        for k in next(iter(all_res.values())):
            hits = sum(all_res[f][k]["hits"] for f in all_res)
            mis  = sum(all_res[f][k]["misses"] for f in all_res)
            fas  = sum(all_res[f][k]["false_alarms"] for f in all_res)
            steps = sum(all_res[f][k]["n_steps"] for f in all_res)
            agg[k] = {
                # fold 마다 스텝 수가 달라 스텝 가중 평균을 쓴다
                "reward_per_step": sum(all_res[f][k]["reward_per_step"]
                                       * all_res[f][k]["n_steps"]
                                       for f in all_res) / steps,
                "POD": hits / max(hits + mis, 1),
                "FAR": fas / max(hits + fas, 1),
                "CSI": hits / max(hits + mis + fas, 1),
                "alarm_rate": sum(all_res[f][k]["alarm_rate"]
                                  * all_res[f][k]["n_steps"]
                                  for f in all_res) / steps,
                "watch_rate": sum(all_res[f][k]["watch_rate"]
                                  * all_res[f][k]["n_steps"]
                                  for f in all_res) / steps,
                "train_reward": sum(all_res[f][k]["train_reward"]
                                    * all_res[f][k]["n_steps"]
                                    for f in all_res) / steps,
                "hits": hits, "misses": mis, "false_alarms": fas,
                "n_events": hits + mis, "n_steps": steps,
            }
        print_table("4-fold 합산 (8,546쌍 전량이 한 번씩 검증셋)", agg)
        all_res["합산"] = agg

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in all_res.items()}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {RESULT_JSON}")


if __name__ == "__main__":
    main()

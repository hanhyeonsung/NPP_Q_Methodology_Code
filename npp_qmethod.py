# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm as normal_dist
import plotly.graph_objs as go

st.set_page_config(page_title="Q-Analysis", layout="wide")

RNG_SEED = 42

Q_SET = [
    "우리나라 에너지 정책은 공정한 절차에 따라 추진되고 있다.",
    "정부의 에너지 정책은 다양한 이해관계자의 의견을 반영해 사회적 합의가 충분히 이루어지고 있다.",
    "에너지 정책은 전문적인 영역이므로 전문가가 결정하도록 해야 한다.",
    "우리나라 에너지 정책은 정치적 성향과 이념의 영향을 받고 있다.",
    "정부가 원자력 정책을 공정하고 책임감 있게 추진하고 있다.",
    "정부가 원자력 발전 정책에 대한 정보를 투명하게 공개하고 있다.",
    "원자력 발전 정책에 대한 정보를 신뢰할 수 있다.",
    "원자력 정책은 다양한 이해관계자의 의견을 충분히 반영하고 있다.",
    "정부가 신재생에너지 정책을 공정하고 책임감 있게 추진하고 있다.",
    "정부가 신재생에너지 정책에 대한 정보를 투명하게 공개하고 있다.",
    "신재생에너지 정책은 다양한 이해관계자의 의견을 충분히 반영하고 있다.",
    "원자력발전은 우리나라에 꼭 필요한 발전원이다.",
    "원자력발전은 에너지 비용을 낮추는 데 효과적이라고 생각한다.",
    "원자력발전은 나에게 경제적인 이득을 줄 것이다.",
    "원자력발전은 에너지 자립도를 강화하는 데 중요한 역할을 한다.",
    "향후 원자력발전의 비중을 지금보다 더 높여야 한다.",
    "우리나라 원자력발전은 안전하다고 생각한다.",
    "사용후핵연료 최종처분시설은 안전하다고 생각한다.",
    "원자력 발전은 온실가스 배출을 줄이는 데 효과적이다.",
    "원자력 발전은 환경에 미치는 부정적 영향이 다른 발전원에 비해 낮다.",
    "원자력 발전은 지속가능한 에너지원이다.",
    "우리나라의 원자력발전기술은 타 발전기술 대비 우수하다.",
    "우리나라 원자력발전의 기술경쟁력을 지속적으로 키워야 한다.",
    "우리나라 원자력발전의 기술우위로 우리 경제가 더 발전할 것이다.",
    "후쿠시마 오염수 방류는 나를 불안하게 만든다.",
    "후쿠시마 오염수 방류는 내 건강에 유의미한 악영향을 미칠 것이다.",
    "나는 후쿠시마 오염수 방류 이후 수산물을 더 적게 소비할 것이다.",
    "신재생에너지는 에너지 비용을 낮추는 데 효과적이라고 생각한다.",
    "신재생에너지는 나에게 경제적인 이득을 줄 것이다.",
    "신재생에너지는 에너지 자립도를 강화하는 데 중요한 역할을 한다.",
    "신재생에너지발전은 온실가스 배출을 줄이는 데 효과적이다.",
    "신재생에너지발전은 환경에 미치는 부정적 영향이 다른 발전원에 비해 낮다.",
    "신재생에너지발전은 지속가능한 에너지원이다.",
    "우리나라의 신재생에너지기술은 타 발전기술 대비 우수하다.",
    "우리나라 신재생에너지의 기술경쟁력을 지속적으로 키워야 한다.",
]

Non_common = {
    "C36": "원자력 정책 결정과정에서 형식적인 여론수렴은 갈등을 악화할 수 있다",
    "C37": "원자력 발전소 건설에서 금전 보상만으로는 안전·신뢰에 따른 수용성 문제를 해결할 수 없다",
    "C38": "원전의 잦은 정지와 고장은 사고 위험을 높일 것이다",
    "C39": "원자력발전의 연료주기와 발전소 해체까지 고려하면 환경영향이 더 커질 것이다",
    "C40": "오염수 방류 위험이 있는 지역에선 어업·관광 생계 전환에 대한 지원이 필요하다",
    "C41": "원자력 발전 규제기관의 독립성이 부족하면 어떤 대책도 신뢰하기 어렵다",
    "C42": "소형모듈형원자력발전(SMR)은 상업 운영 데이터가 부족해 신뢰하기 어렵다",
    "C43": "에너지 안보를 위해 핵연료·부품 공급망 지정학 리스크를 상시 점검하고 대체 경로를 확보해야 한다",
    "C44": "원자력 발전 확대를 위해 기존 부지 내부에 대체·증설을 우선 검토해야한다",
    "C45": "원자력 발전 건설에 앞서 원자력 발전에 대한 사고나 피해에 대한 책임보험·배상 절차를 사전 확정해야 수용성을 높일 수 있다",
    "C46": "원자력 발전 확대는 국민들의 의견을 충분히 수렴하는 숙의·질의응답 과정이 있어야만 갈등을 줄일 수 있다",
    "C47": "내가 거주하는 지역에 원자력 발전소가 들어온다면 주택 가치 하락 보상과 토지 매입 옵션이 있어야 수용할 수 있다",
    "C48": "원자력 발전의 정지·고장률에 대한 정보가 투명하게 공개되면 원자력 발전 정책을 수용할 수 있다",
    "C49": "원자력 발전의 연료주기·발전소 해체 영향 공개·탄소감축이 이뤄지면 수용할 수 있다",
    "C50": "후쿠시마 오염수 방류문제는 장기적으로 수산·해양관광에 타격을 줄 것이다",
    "C51": "원자력 발전의 현장 상주 규제자, 상시감사와 상시감사에 대한 데이터가 더 투명하게 공개되면 신뢰도가 높아질 것이다",
    "C52": "소형모듈형원자력발전(SMR)의 상업운전 데이터가 축적되면 수용할 수 있다",
    "C53": "원전 운영의 사이버·물리 보안을 통합 관리하고 침해 대응 훈련을 정기적으로 시행해야 한다",
    "C54": "원자력 발전소 인근지역의 반경별 환경 데이터 공개와 주민 참여 감시는 필수적으로 이루어져야 한다",
    "C55": "원자력 발전의 수용성을 높이기 위하여 지역 기업 참여·상주 고용·하청 의무를 제도화 해야한다",
    "C56": "원자력발전 정책 수립의 여론수렴은 대국민 합의 형성에 도움이 된다",
    "C57": "지역내 원자력 발전소 건설에서 지역사회 발전을 위한 상생 패키지(보상·일자리·인프라)가 있으면 수용성이 높아진다",
    "C58": "현재의 원자력 발전소의 안전관리 수준은 사고 위험에 대한 걱정이 없을만큼 높은 수준이다",
    "C59": "원자력 발전은 수명주기 전과정의 탄소배출이 낮아 국가감축 목표에 기여한다",
    "C60": "원자력 발전에 대한 정밀 모니터링·지역 수산물에 대한 브랜딩·수산물 판로지원이 병행되면 어업 피해를 최소화할 수 있다",
    "C61": "현재 원자력 규제위원회의 규제·감사 운영 구조는 충분히 독립적으로 이루어지고 있다",
    "C62": "분산형 전원인 소형 모듈형 원전(SMR)으로 정전에 대한 대응력을 높일 수 있다",
    "C63": "공급망에 대한 리스크 관리가 이루어지면 에너지소비의 해외 의존은 수용 가능하다",
    "C64": "원자력 발전은 수명주기 배출이 낮아 감축 목표에 기여한다",
    "C65": "원자력 발전 건설에 앞서 원자력 발전에 대한 사고나 피해에 대한 책임보험·배상 절차를 사전 확정해야 수용성을 높일 수 있다",
}

C_COLS = [f"C{i:02d}" for i in range(1, len(Q_SET) + 1)]
Q_dict = dict(zip(C_COLS, Q_SET))
merged_dict = {**Q_dict, **Non_common}

DEMO_MAP = {
    "성별": {1: "남자", 2: "여자"},
    "연령": {1: "18~29세", 2: "30~39세", 3: "40~49세", 4: "50~59세", 5: "60세이상"},
    "거주지역": {1: "서울", 2: "인천/경기", 3: "대전/충청/세종", 4: "광주/전라", 5: "대구/경북", 6: "부산/울산/경남", 7: "강원/제주"},
    "거주기간": {1: "2년 미만", 2: "2~5년", 3: "5~10년", 4: "10년 이상"},
    "학력": {1: "중졸이하", 2: "고졸", 3: "전문대재학이상"},
    "직업": {1: "농/임/어업", 2: "자영업", 3: "판매/서비스직", 4: "생산/기능/노무", 5: "사무/관리/전문", 6: "주부", 7: "학생", 8: "무직/기타"},
    "종사자여부": {1: "있음", 2: "없음", 3: "모름"},
    "가구소득": {1: "200만원미만", 2: "200~400만원", 3: "400~600만원", 4: "600~800만원", 5: "800만원 이상", 6: "모름/무응답"},
    "개인소득": {1: "200만원미만", 2: "200~400만원", 3: "400~600만원", 4: "600~800만원", 5: "800만원 이상", 6: "모름/무응답"},
    "이념성향": {1: "진보", 2: "중도", 3: "보수"},
}

원전_거주지역_MAPPING = {
    1: "부산광역시/기장군",
    2: "울산광역시/울주군",
    3: "전라남도/영광군",
    4: "경상북도/경주시",
    5: "경상북도/울진군",
}

TAB_HELP = {
    "tab1": (
        "- **Explained Variance**: 요인이 전체 변동을 얼마나 설명하는지(고유값/비율/누적).\n"
        "- **Scree Plot**: 고유값이 급격히 꺾이는 지점(Elbow) 전까지를 요인 수 후보로 봅니다.\n"
        "- **Loadings**: 응답자(Q-sort)가 각 요인에 얼마나 강하게 연결되는지(부호는 방향, 크기는 강도).\n"
        "- **Factor Distribution**: |loading| 기준으로 유형(Type) 할당 결과 분포(표본 편중 여부 확인)."
    ),
    "tab2": (
        "- **Humphrey's Rule**: 요인 ‘유의성’ 점검용 휴리스틱입니다.\n"
        "- 각 요인에서 **가장 큰 적재량 2개의 곱**이 임계값(대개 `2*(1/√Nitems)`)을 넘으면 Pass.\n"
        "- Pass가 False인 요인은 해석 시 보수적으로(또는 제외) 다루는 게 안전합니다."
    ),
    "tab3": (
        "- **Distinguishing Statements**: 특정 요인이 ‘다른 모든 요인’과 통계적으로 유의하게 차이나는 문항.\n"
        "- 이 문항들이 해당 유형(Type)을 설명하는 **핵심 해석 단서**가 됩니다.\n"
        "- 표에서 **Min Difference**는 (타겟 요인 z - 비교 요인 z) 중 ‘가장 작은 절대차’를 의미.\n"
        "- **Z-Stat/P-Value**가 기준(예: p<.01, p<.05)을 만족하면 구별진술로 판단합니다."
    ),
    "tab4": (
        "- **Cross-Set Congruence**: A/B/C 세트 간 요인배열이 얼마나 유사한지 Tucker’s φ로 비교.\n"
        "- 일반적으로 **0.90 이상**이면 매우 높은 유사성(거의 동일 요인), **0.80~0.90**은 높은 유사성.\n"
        "- 공통 문항(C01~C35)만으로 계산하므로 ‘프레이밍(비공통)’ 영향은 배제된 비교입니다."
    ),
    "tab5": (
        "- **Bootstrap Stability**: 응답자 재표집으로 요인배열이 얼마나 ‘흔들리는지’를 확인합니다.\n"
        "- **z_mean**: 부트스트랩 반복에서의 평균 요인배열(z).\n"
        "- **z_sd(z_std)**: 반복에서의 표준편차(클수록 해당 문항이 불안정).\n"
        "- **Tucker’s φ**: 표준(원분석) 요인배열과 부트스트랩 요인배열의 유사도(높을수록 안정).\n"
        "- `thr/sep`가 빡빡할수록 각 요인에 ‘확실히’ 속한 사람만 쓰므로 n_used가 줄 수 있습니다."
    ),
    "tab6": (
        "- **Framing ATT**: 공통 문항을 제외한 **비공통 문항(Unique)** 평균을 세트별로 비교합니다.\n"
        "- 세트 간 비공통 평균 차이가 크면 **문항 구성(프레이밍)** 에 따른 응답 편향 가능성을 시사.\n"
        "- Cohen’s d는 효과크기(표준편차 가정에 따라 값이 달라짐)로 참고용입니다."
    ),
    "tab7": (
        "- **Demographic Analysis**: 요인(Type)별로 인구통계 분포가 어떻게 다른지 교차표로 확인.\n"
        "- 여기서 Type은 기본적으로 **가장 큰 |loading| 요인**으로 할당됩니다.\n"
        "- (선택) Unassigned 제외하고 보기를 해제하면 ‘애매한’ 응답자(|loading| < 0.4)를 포함한 분포를 볼 수 있습니다."
    ),
}


def show_tab_help(key: str):
    msg = TAB_HELP.get(key)
    if msg:
        st.info(msg)


def find_demo_key(col_name: str):
    for key in DEMO_MAP.keys():
        if key in col_name or (key == "거주기간" and "거주" in col_name and "기간" in col_name):
            return key
        if "종사자" in col_name and key == "종사자여부":
            return key
        if "가구" in col_name and "소득" in col_name and key == "가구소득":
            return key
        if "개인" in col_name and "소득" in col_name and key == "개인소득":
            return key
    return None


def standardize_rows(X: np.ndarray):
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def tuckers_phi(vec_a: np.ndarray, vec_b: np.ndarray):
    den = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if den == 0:
        return 0.0
    return float(vec_a @ vec_b / den)


class QEngine:
    def __init__(self, data_df: pd.DataFrame, n_factors=3, rotation=True, corr_method="spearman"):
        self.raw_df = data_df
        self.n_factors = int(n_factors)
        self.rotation = bool(rotation)
        self.corr_method = corr_method

        temp = data_df.apply(pd.to_numeric, errors="coerce").values
        row_means = np.nanmean(temp, axis=1)
        inds = np.where(np.isnan(temp))
        temp[inds] = np.take(row_means, inds[0])
        self.data = np.nan_to_num(temp, nan=0.0)

        self.n_persons, self.n_items = self.data.shape
        self.loadings = None
        self.factor_arrays = None
        self.explained_variance = None
        self.eigenvalues = None

    def fit(self):
        if self.corr_method == "spearman":
            R, _ = spearmanr(self.data, axis=1)
            z_data = standardize_rows(self.data)
        else:
            z_data = standardize_rows(self.data)
            R = np.corrcoef(z_data)
        R = np.nan_to_num(R, nan=0.0)

        eigvals, eigvecs = np.linalg.eigh(R)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.eigenvalues = eigvals

        k = self.n_factors
        valid = np.maximum(eigvals[:k], 0)
        L = eigvecs[:, :k] * np.sqrt(valid)

        if self.rotation and k > 1:
            L = self._varimax(L, normalize=False)

        for f in range(L.shape[1]):
            imax = int(np.argmax(np.abs(L[:, f])))
            if L[imax, f] < 0:
                L[:, f] *= -1

        self.loadings = L
        self.explained_variance = eigvals[:k]
        self.factor_arrays = self._calculate_factor_arrays(L, z_data)
        return self

    def _varimax(self, Phi: np.ndarray, normalize=True, gamma=1.0, q=20, tol=1e-6):
        if normalize:
            h2 = np.sum(Phi**2, axis=1, keepdims=True)
            h2[h2 == 0] = 1
            Phi = Phi / np.sqrt(h2)

        p, k = Phi.shape
        R = np.eye(k)
        d = 0.0
        for _ in range(q):
            d_old = d
            Lambda = Phi @ R
            u, s, vh = np.linalg.svd(Phi.T @ (Lambda**3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))))
            R = u @ vh
            d = float(np.sum(s))
            if d_old != 0 and d / d_old < 1 + tol:
                break

        Phi = Phi @ R
        if normalize:
            Phi = Phi * np.sqrt(h2)
        return Phi

    def _calculate_factor_arrays(self, loadings: np.ndarray, z_data: np.ndarray):
        n_items = z_data.shape[1]
        arrays = np.zeros((n_items, self.n_factors), dtype=float)
        for f in range(self.n_factors):
            l_vec = loadings[:, f]
            l_clean = np.clip(l_vec, -0.95, 0.95)
            weights = l_clean / (1 - l_clean**2)
            if np.sum(np.abs(weights)) < 1e-6:
                arrays[:, f] = 0.0
                continue
            ws = weights @ z_data
            mu = ws.mean()
            sd = ws.std(ddof=1)
            if sd == 0:
                sd = 1.0
            arrays[:, f] = (ws - mu) / sd
        return arrays


def check_humphreys_rule(loadings: np.ndarray, n_items: int):
    n_factors = loadings.shape[1]
    threshold = 2 * (1 / np.sqrt(n_items))
    rows = []
    for f in range(n_factors):
        abs_loads = np.sort(np.abs(loadings[:, f]))[::-1]
        if len(abs_loads) >= 2:
            prod = float(abs_loads[0] * abs_loads[1])
            rows.append({"Factor": f"F{f+1}", "Product": round(prod, 4), "Threshold": threshold, "Pass": prod > threshold})
        else:
            rows.append({"Factor": f"F{f+1}", "Product": 0.0, "Threshold": threshold, "Pass": False})
    return pd.DataFrame(rows)


def find_distinguishing_items_r_logic(factor_arrays, n_factors, item_labels=None, se=0.30, alpha=0.01):
    col_names = [f"F{i+1}" for i in range(n_factors)]
    df_arrays = pd.DataFrame(factor_arrays, columns=col_names)
    if item_labels is not None:
        df_arrays.index = item_labels

    crit_z = normal_dist.ppf(1 - alpha / 2)
    out = {}

    for i in range(n_factors):
        target_col = f"F{i+1}"
        other_cols = [c for c in df_arrays.columns if c != target_col]
        if not other_cols:
            continue

        is_higher_all = pd.Series(True, index=df_arrays.index)
        is_lower_all = pd.Series(True, index=df_arrays.index)
        min_diff_val = pd.Series(np.inf, index=df_arrays.index)

        for other in other_cols:
            diff = df_arrays[target_col] - df_arrays[other]
            z_stat = diff / (np.sqrt(2) * se)
            is_higher_all &= (z_stat > crit_z)
            is_lower_all &= (z_stat < -crit_z)

            cur_abs = np.abs(diff)
            update = cur_abs < np.abs(min_diff_val)
            min_diff_val[update] = diff[update]

        dist_mask = is_higher_all | is_lower_all
        dist_items = df_arrays[dist_mask].copy()
        if dist_items.empty:
            continue

        dist_items["Distinction"] = np.where(is_higher_all[dist_mask], "Higher", "Lower")
        dist_items["Min Difference"] = min_diff_val[dist_mask]
        dist_items["Z-Stat"] = dist_items["Min Difference"] / (np.sqrt(2) * se)
        dist_items["P-Value"] = 2 * (1 - normal_dist.cdf(np.abs(dist_items["Z-Stat"])))
        dist_items = dist_items.sort_values("Min Difference", ascending=False, key=abs)

        round_cols = col_names + ["Min Difference", "Z-Stat", "P-Value"]
        dist_items[round_cols] = dist_items[round_cols].round(3)
        out[target_col] = dist_items

    return out


def _corr_mat_cols(A: np.ndarray, B: np.ndarray, eps=1e-12):
    A0 = A - A.mean(axis=0, keepdims=True)
    B0 = B - B.mean(axis=0, keepdims=True)
    An = np.linalg.norm(A0, axis=0, keepdims=True) + eps
    Bn = np.linalg.norm(B0, axis=0, keepdims=True) + eps
    return (A0.T @ B0) / (An.T @ Bn)


def qindtest_conservative_align(loa_boot: np.ndarray, target: np.ndarray, nfactors: int):
    L0 = np.asarray(loa_boot, float).copy()
    T = np.asarray(target, float)

    C = _corr_mat_cols(T, L0)
    absC = np.abs(C)
    row_argmax = absC.argmax(axis=1)
    problematic = [i for i in range(nfactors) if row_argmax[i] != i]

    L = L0.copy()
    if len(problematic) == 2:
        i, k = problematic
        if row_argmax[i] == k and row_argmax[k] == i:
            L_swapped = L.copy()
            L_swapped[:, [i, k]] = L_swapped[:, [k, i]]
            C2 = _corr_mat_cols(T, L_swapped)
            absC2 = np.abs(C2)
            row_argmax2 = absC2.argmax(axis=1)
            problematic2 = [r for r in range(nfactors) if row_argmax2[r] != r]
            if len(problematic2) == 0:
                L = L_swapped

    C_final_pre = _corr_mat_cols(T, L)
    signs = np.sign(np.diag(C_final_pre))
    signs[signs == 0] = 1.0
    L = L * signs
    return L


def bootstrap_stability_qmethod(
    df_values: np.ndarray,
    n_factors: int,
    n_boot: int = 500,
    corr_method: str = "spearman",
    thr: float = 0.40,
    sep: float = 0.10,
    rng_seed: int = 42,
    do_align: bool = True,
    abs_phi: bool = True,
    best_match_when_no_align: bool = True,
):
    rng = np.random.default_rng(rng_seed)

    base_engine = QEngine(pd.DataFrame(df_values), n_factors=n_factors, corr_method=corr_method).fit()
    L_target_full = base_engine.loadings.copy()
    base_arrays = base_engine.factor_arrays.copy()
    N, n_items = df_values.shape

    arr_stack = np.zeros((n_boot, n_items, n_factors), dtype=float)
    phi_mat = np.zeros((n_boot, n_factors), dtype=float)
    used_counts = np.zeros((n_boot, n_factors), dtype=int)

    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        sample = df_values[idx, :]

        boot_engine = QEngine(pd.DataFrame(sample), n_factors=n_factors, corr_method=corr_method).fit()
        L_boot = boot_engine.loadings.copy()

        L_target_res = L_target_full[idx, :]

        if do_align:
            L_proc = qindtest_conservative_align(L_boot, L_target_res, nfactors=n_factors)
        else:
            L_proc = L_boot

        z_data = standardize_rows(sample)

        absL = np.abs(L_proc)
        max_idx = absL.argmax(axis=1)
        sorted_abs = np.sort(absL, axis=1)[:, ::-1]
        max_abs = sorted_abs[:, 0]
        second_abs = sorted_abs[:, 1] if n_factors >= 2 else np.zeros(N)

        ok = (max_abs >= thr) & ((max_abs - second_abs) >= sep)

        arrays_b = np.zeros((n_items, n_factors), dtype=float)
        for f in range(n_factors):
            sel = np.where(ok & (max_idx == f))[0]
            used_counts[b, f] = int(sel.size)
            if sel.size == 0:
                arrays_b[:, f] = 0.0
                continue

            l_vec = np.clip(L_proc[sel, f], -0.95, 0.95)
            w = l_vec / (1 - l_vec**2)
            if np.sum(np.abs(w)) < 1e-8:
                arrays_b[:, f] = 0.0
                continue

            ws = w @ z_data[sel, :]
            mu = ws.mean()
            sd = ws.std(ddof=1)
            if sd == 0:
                sd = 1.0
            arrays_b[:, f] = (ws - mu) / sd

        arr_stack[b, :, :] = arrays_b

        if (not do_align) and best_match_when_no_align:
            corr = np.zeros((n_factors, n_factors), dtype=float)
            for i in range(n_factors):
                a = base_arrays[:, i]
                na = np.linalg.norm(a)
                for j in range(n_factors):
                    bb = arrays_b[:, j]
                    nb = np.linalg.norm(bb)
                    den = na * nb
                    corr[i, j] = 0.0 if den == 0 else float((a @ bb) / den)
            for i in range(n_factors):
                phi = float(corr[i, np.argmax(np.abs(corr[i, :]))])
                phi_mat[b, i] = abs(phi) if abs_phi else phi
        else:
            for f in range(n_factors):
                a = base_arrays[:, f]
                bb = arrays_b[:, f]
                den = np.linalg.norm(a) * np.linalg.norm(bb)
                phi = 0.0 if den == 0 else float((a @ bb) / den)
                phi_mat[b, f] = abs(phi) if abs_phi else phi

    z_mean = arr_stack.mean(axis=0)
    z_sd = arr_stack.std(axis=0, ddof=1)

    return {
        "base_arrays": base_arrays,
        "z_mean": z_mean,
        "z_sd": z_sd,
        "phi_mat": phi_mat,
        "n_used_mean": used_counts.mean(axis=0),
        "n_used_min": used_counts.min(axis=0),
        "n_used_p10": np.percentile(used_counts, 10, axis=0),
    }


@st.cache_data(show_spinner=False)
def calculate_cross_set_congruence(parts_q, common_cols, n_factors=3, corr_method="spearman"):
    engines = {}
    for name, df in parts_q.items():
        df_common = df[common_cols]
        engine = QEngine(df_common, n_factors=n_factors, corr_method=corr_method).fit()
        engines[name] = engine.factor_arrays

    results = []
    for s1, s2 in [("A", "B"), ("A", "C"), ("B", "C")]:
        if s1 not in engines or s2 not in engines:
            continue
        arr1, arr2 = engines[s1], engines[s2]
        phis = [abs(tuckers_phi(arr1[:, f], arr2[:, f])) for f in range(n_factors)]
        results.append({"Pair": f"{s1}-{s2}", "Factors": phis})
    return results


def parse_uploaded_file(file):
    xls = pd.ExcelFile(file)
    valid_names = ["PARTA", "PARTB", "PARTC"]

    q_parts = {}
    d_parts = {}

    meta_keywords = ["time", "date", "duration", "ip", "token"]
    q_pattern = re.compile(r"^C(0[1-9]|[12][0-9]|3[0-9]|4[0-9]|5[0-9]|6[0-9])$", re.IGNORECASE)

    for sname in valid_names:
        if sname not in xls.sheet_names:
            continue

        df = pd.read_excel(xls, sname)
        id_col = next((c for c in df.columns if str(c).lower() in ["email", "id", "respondent", "pid"]), None)
        if not id_col:
            df["ID"] = [f"P{i+1}" for i in range(len(df))]
            id_col = "ID"
        else:
            df[id_col] = df[id_col].astype(str)

        df = df.set_index(id_col)
        numeric_df = df.apply(pd.to_numeric, errors="coerce")

        q_cols, d_cols = [], []
        for c in df.columns:
            if q_pattern.match(str(c)):
                q_cols.append(c)
            elif c != id_col and not any(k in str(c).lower() for k in meta_keywords):
                if numeric_df[c].notna().sum() > 0:
                    d_cols.append(c)

        if q_cols:
            q_df = numeric_df[q_cols].dropna(thresh=len(q_cols) * 0.8)
            key = sname.replace("PART", "")
            q_parts[key] = q_df
            d_parts[key] = df.loc[q_df.index, d_cols]

    return q_parts, d_parts


def get_common_columns(parts):
    pat = re.compile(r"^C(0[1-9]|[12][0-9]|3[0-5])$", re.IGNORECASE)
    sets_cols = []
    for df in parts.values():
        cols = {c for c in df.columns if pat.match(str(c))}
        sets_cols.append(cols)
    if not sets_cols:
        return []
    return sorted(list(set.intersection(*sets_cols)))


def calculate_framing_att(parts, common_cols):
    summary = []
    for name, df in parts.items():
        non_common_cols = [c for c in df.columns if c not in common_cols]
        if not non_common_cols:
            mean_val, n_items, items_str = 0.0, 0, "-"
        else:
            mean_val = float(np.nanmean(df[non_common_cols].values))
            n_items = len(non_common_cols)
            if len(non_common_cols) > 5:
                items_str = f"{non_common_cols[0]}...{non_common_cols[-1]} ({len(non_common_cols)})"
            else:
                items_str = ", ".join(non_common_cols)
        summary.append({"Set": name, "Non-Common Mean": mean_val, "N_Items": n_items, "Items": items_str})
    return pd.DataFrame(summary).set_index("Set")


st.title("Q-방법론 분석 엔진 v2.0")
st.markdown("""
> **분석 엔진 개요**
> 본 도구는 Q-방법론에 기반하여 응답자 유형을 도출하고,  
> 요인 구조의 해석·안정성·집단 간 일관성을 체계적으로 검증하기 위해 설계되었습니다.
>
> - 요인 추출 및 회전 결과에 대한 **구조적 해석 지원**
> - 통계적 기준에 따른 **구별 진술문 식별**
> - 부트스트랩 재표집을 통한 **요인 배열 안정성 검증**
> - 공통 문항 기반 **집단 간 요인 비교**
> - 유형별 **인구통계학적 특성 탐색**
""")

uploaded_file = st.sidebar.file_uploader("Upload Excel (PARTA/B/C)", type="xlsx")

if not uploaded_file:
    st.info("분석할 엑셀 파일(PARTA, PARTB, PARTC 시트 포함)을 업로드해주세요.")
    st.stop()

q_parts, d_parts = parse_uploaded_file(uploaded_file)
if not q_parts:
    st.error("데이터를 읽을 수 없습니다. 시트명(PARTA...)과 문항 코드(C01...)를 확인해주세요.")
    st.stop()

st.sidebar.success(f"Loaded Sets: {list(q_parts.keys())}")
common_cols = get_common_columns(q_parts)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "1. Basic Q-Analysis",
        "2. Humphrey's Rule",
        "3. Distinguishing Items",
        "4. Cross-Set Congruence",
        "5. Bootstrap Stability",
        "6. Framing ATT",
        "7. Demographic Analysis",
    ]
)

target_set = st.sidebar.selectbox("Target Set (Tabs 1,2,3,5,7)", list(q_parts.keys()))
n_factors = int(st.sidebar.number_input("Number of Factors", 1, 7, 3))
corr_method = st.sidebar.selectbox("Correlation Method", ["pearson", "spearman"], index=1)

df = q_parts[target_set]
engine = QEngine(df, n_factors=n_factors, corr_method=corr_method).fit()

with tab1:
    st.header("Basic Q-Analysis Result")
    show_tab_help("tab1")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Explained Variance")
        r1 = engine.explained_variance
        r2 = engine.explained_variance / engine.eigenvalues.sum()
        r3 = r2.cumsum()
        df_ev = pd.DataFrame(
            np.vstack([r1, r2, r3]),
            columns=[f"F{i+1}" for i in range(n_factors)],
            index=["Eigenvalue", "Eigenvalue_ratio", "Cumsum"],
        )
        st.dataframe(df_ev.round(3))

        st.subheader("Scree Plot")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(engine.eigenvalues) + 1), engine.eigenvalues, "bo-", markersize=6)
        ax.axhline(y=1.0, color="r", linestyle="--", linewidth=0.8, label="Eigenvalue=1")
        ax.set_title("Scree Plot (Eigenvalues)")
        ax.set_xlabel("Factor Number")
        ax.set_ylabel("Eigenvalue")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

    with c2:
        st.subheader("Factor Distribution")
        max_vals = np.max(np.abs(engine.loadings), axis=1)
        max_idxs = np.argmax(np.abs(engine.loadings), axis=1)
        valid_types = [f"Type {i+1}" if v > 0.4 else "None" for i, v in zip(max_idxs, max_vals)]
        s_counts = pd.Series(valid_types).value_counts().sort_index()
        st.bar_chart(s_counts)
        st.caption(f"Total Participants: {len(df)}")

    st.subheader("Factor Loadings (Rotated)")
    st.dataframe(
        pd.DataFrame(engine.loadings, index=df.index, columns=[f"F{i+1}" for i in range(n_factors)])
        .style.background_gradient(cmap="Blues")
    )

with tab2:
    st.header("Humphrey's Rule Validation")
    show_tab_help("tab2")
    res_hum = check_humphreys_rule(engine.loadings, engine.n_items)
    st.dataframe(res_hum.style.applymap(lambda x: "color: green; font-weight: bold" if x else "color: red", subset=["Pass"]))

with tab3:
    st.header("Distinguishing Statements")
    show_tab_help("tab3")

    c1, c2 = st.columns(2)
    with c1:
        alpha_level = st.selectbox("Alpha Level", [0.01, 0.05], index=0)
    with c2:
        se_val = st.number_input("Standard Error", 0.1, 1.0, 0.30, step=0.01)

    dist_dict = find_distinguishing_items_r_logic(engine.factor_arrays, n_factors, item_labels=df.columns, se=se_val, alpha=alpha_level)
    subtabs = st.tabs([f"Type {i+1}" for i in range(n_factors)])

    for i, stab in enumerate(subtabs):
        with stab:
            f_key = f"F{i+1}"
            items_df = dist_dict.get(f_key)
            if items_df is None or items_df.empty:
                st.warning("No distinguishing items found for this factor.")
                continue

            st.write(f"Found {len(items_df)} distinguishing items (p < {alpha_level})")

            df_show = items_df.copy()
            df_show["Statement"] = df_show.index.map(Q_dict)

            if df_show["Statement"].isnull().any():
                target_nums = df_show[df_show["Statement"].isnull()].index
                for t in target_nums:
                    base_num = int(t.split("C")[1])
                    if target_set == "B":
                        target_key = "C" + str(base_num + 10)
                    elif target_set == "C":
                        target_key = "C" + str(base_num + 20)
                    else:
                        target_key = "C" + str(base_num)
                    df_show.loc[t, "Statement"] = Non_common.get(target_key, target_key)

            cols = ["Statement"] + [c for c in df_show.columns if c != "Statement"]
            df_show = df_show[cols]

            st.dataframe(
                df_show.style.background_gradient(cmap="coolwarm", subset=["Min Difference"], vmin=-2, vmax=2),
                use_container_width=True,
            )

with tab4:
    st.header("Cross-Set Congruence")
    show_tab_help("tab4")
    if len(common_cols) < 5:
        st.warning("공통 문항(C01~C35)이 충분하지 않아 교차 분석을 수행할 수 없습니다.")
    else:
        results = calculate_cross_set_congruence(q_parts, common_cols, n_factors, corr_method)
        for res in results:
            st.subheader(f"Comparison: {res['Pair']}")
            cols = st.columns(len(res["Factors"]))
            for i, phi in enumerate(res["Factors"]):
                cols[i].metric(f"Factor {i+1}", f"{phi:.3f}", delta_color="normal" if phi > 0.9 else "off")
            st.divider()

with tab5:
    st.header("Paper-style Bootstrap (Z-mean / Z-sd for Factor Arrays)")
    show_tab_help("tab5")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        n_boot_paper = st.number_input("n_boot", 50, 5000, 200, step=50, key="t5_nboot")
    with c2:
        thr_loading = st.number_input("|loading| thr", 0.10, 0.90, 0.40, step=0.01, key="t5_thr")
    with c3:
        sep_loading = st.number_input("sep(|max|-|2nd|)", 0.00, 0.50, 0.10, step=0.01, key="t5_sep")
    with c4:
        do_align = st.checkbox("Alignment correction (swap+reflect)", value=True, key="t5_align")
    with c5:
        best_match = st.checkbox("Best-match (only when alignment OFF)", value=True, key="t5_best_match")
    with c6:
        abs_phi = st.checkbox("Use abs(phi)", value=True, key="t5_abs_phi")

    run = st.button("Run Bootstrap", key="t5_run")
    if run:
        with st.spinner(f"Running {int(n_boot_paper)} iterations..."):
            res = bootstrap_stability_qmethod(
                df_values=df.values,
                n_factors=int(n_factors),
                n_boot=int(n_boot_paper),
                corr_method=corr_method,
                thr=float(thr_loading),
                sep=float(sep_loading),
                rng_seed=RNG_SEED,
                do_align=bool(do_align),
                abs_phi=bool(abs_phi),
                best_match_when_no_align=bool(best_match),
            )
        st.session_state["paper_boot_res"] = res

    if "paper_boot_res" not in st.session_state:
        st.info("위 버튼을 눌러 Bootstrap 결과를 먼저 생성하세요.")
    else:
        res = st.session_state["paper_boot_res"]
        base_arrays = res["base_arrays"]
        z_mean = res["z_mean"]
        z_sd = res["z_sd"]
        phi_mat = res["phi_mat"]

        summary = pd.DataFrame(
            {
                "phi_mean": phi_mat.mean(axis=0),
                "phi_sd": phi_mat.std(axis=0, ddof=1),
                "phi_rate_90": (phi_mat >= 0.90).mean(axis=0),
                "mean_statement_sd": z_sd.mean(axis=0),
                "n_used_mean": res["n_used_mean"],
                "n_used_p10": res["n_used_p10"],
                "n_used_min": res["n_used_min"],
            },
            index=[f"F{i+1}" for i in range(int(n_factors))],
        )
        st.subheader("Factor-level stability summary")
        st.dataframe(summary.round(3), use_container_width=True)

        factor_cols = [f"F{i+1}" for i in range(int(n_factors))]
        df_base = pd.DataFrame(base_arrays, index=df.columns, columns=factor_cols)
        df_mean = pd.DataFrame(z_mean, index=df.columns, columns=factor_cols)
        df_sd = pd.DataFrame(z_sd, index=df.columns, columns=factor_cols)

        n_items = df_sd.shape[0]
        cA, cB, cC = st.columns([1, 1, 2])
        with cA:
            order_mode = st.selectbox("정렬 기준", ["(원본 순서)", "누적 표준오차(합)", "누적 표준오차(L2)"], index=1, key="t5_order")
        with cB:
            topn_rank = st.number_input("표시 Top N", 10, 300, 60, step=10, key="t5_topn_rank")
        with cC:
            st.caption("누적 표준오차(합)=Σ z_sd, L2=√(Σ z_sd²). 값이 클수록 문항이 더 흔들림")

        if order_mode == "누적 표준오차(합)":
            cum_sd = df_sd.sum(axis=1)
            order = cum_sd.sort_values(ascending=False).index.to_list()
        elif order_mode == "누적 표준오차(L2)":
            cum_sd = np.sqrt((df_sd**2).sum(axis=1))
            order = cum_sd.sort_values(ascending=False).index.to_list()
        else:
            cum_sd = pd.Series(np.nan, index=df_sd.index)
            order = df.columns.tolist()

        rank_df = pd.DataFrame(
            {
                "StatementCode": order,
                "Statement": [merged_dict.get(i, i) for i in order],
            }
        )
        for f in factor_cols:
            rank_df[f"{f} z_sd"] = df_sd.loc[order, f].values
        if order_mode != "(원본 순서)":
            rank_df["Cum z_sd"] = cum_sd.loc[order].values

        st.subheader("누적 표준오차 큰 문항 랭킹")
        st.dataframe(rank_df.head(int(topn_rank)).round(3), use_container_width=True)

        st.subheader("Fig3-style: z_mean ± z_sd with Standard z overlay (all factors)")
        fig = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        y_ord = order

        for fi in range(int(n_factors)):
            color = colors[fi % len(colors)]
            fname = f"F{fi+1}"

            x_mean = df_mean.loc[order, fname].values
            x_sd = df_sd.loc[order, fname].values
            x_base = df_base.loc[order, fname].values

            fig.add_trace(
                go.Scatter(
                    x=x_mean,
                    y=y_ord,
                    mode="markers",
                    name=f"{fname} — bootstrap z_mean ± z_sd",
                    error_x=dict(type="data", array=x_sd, visible=True, thickness=1.2, width=0),
                    marker=dict(symbol="circle", size=7, color=color, opacity=0.55),
                    legendgroup=fname,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_base,
                    y=y_ord,
                    mode="markers",
                    name=f"{fname} — standard z",
                    marker=dict(symbol="diamond", size=7, color=color, line=dict(width=1, color="black")),
                    legendgroup=fname,
                )
            )

        fig.add_vline(x=0, line_dash="dash", line_width=1)
        fig.update_layout(
            xaxis_title="z-score",
            yaxis_title="Statement",
            height=max(500, int(26 * n_items)),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.header("Framing ATT (Non-Common Items)")
    show_tab_help("tab6")
    att_df = calculate_framing_att(q_parts, common_cols)
    st.dataframe(att_df)

    if len(att_df) >= 2:
        st.subheader("Pairwise Differences")
        sets = att_df.index.tolist()
        pairs = [(a, b) for idx, a in enumerate(sets) for b in sets[idx + 1 :]]
        diffs = [
            {"Pair": f"{s2} - {s1}", "Difference": att_df.loc[s2, "Non-Common Mean"] - att_df.loc[s1, "Non-Common Mean"]}
            for s1, s2 in pairs
        ]
        st.dataframe(pd.DataFrame(diffs))

        st.subheader("Cohen's d")
        sd = st.selectbox("SD Likert", [1.2, 1.4, 1.6], index=0)
        diffs_d = [
            {"Pair": f"{s2} - {s1}", "d": (att_df.loc[s2, "Non-Common Mean"] - att_df.loc[s1, "Non-Common Mean"]) / sd}
            for s1, s2 in pairs
        ]
        st.dataframe(pd.DataFrame(diffs_d))

with tab7:
    st.header("Demographic Analysis")
    show_tab_help("tab7")


    absL = np.abs(engine.loadings)
    max_vals = absL.max(axis=1)
    max_idxs = absL.argmax(axis=1)

    factor_labels = [
        f"Type {i+1}" if v >= 0.4 else "Unassigned"
        for i, v in zip(max_idxs, max_vals)
    ]
    if target_set not in d_parts:
        st.warning("이 데이터셋(Set)에는 인구통계 정보가 없습니다.")
        st.stop()

    demo_df = d_parts[target_set].copy()
    common_indices = demo_df.index.intersection(df.index)
    if len(common_indices) == 0:
        st.error("Q-sort 데이터와 인구통계 데이터의 ID(PID)가 일치하지 않습니다.")
        st.stop()

    demo_subset = demo_df.loc[common_indices].copy()
    drop_unassigned = st.checkbox("Unassigned 제외하고 보기", value=True)

    demo_subset = demo_subset.copy()
    demo_subset["Assigned_Factor"] = pd.Series(factor_labels, index=df.index).loc[common_indices]

    if drop_unassigned:
        demo_subset = demo_subset[demo_subset["Assigned_Factor"] != "Unassigned"]

    found = False
    for col in demo_subset.columns:
        if col == "Assigned_Factor":
            continue

        map_key = find_demo_key(str(col))
        if col == "거주지역" and ("원전지역" in str(uploaded_file.name)):
            DEMO_MAP[map_key] = 원전_거주지역_MAPPING

        if not map_key:
            continue

        found = True
        st.markdown(f"### {col}")

        mapped_col = demo_subset[col].map(DEMO_MAP[map_key]).fillna("Unknown/Other")
        ct = pd.crosstab(mapped_col, demo_subset["Assigned_Factor"])

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(ct)
        with c2:
            st.bar_chart(ct)

        st.divider()

    if not found:
        st.warning("지정된 인구통계 컬럼(성별, 연령 등)을 찾을 수 없습니다. 엑셀 파일의 컬럼명을 확인해주세요.")
        st.dataframe(demo_subset.head())

"""FreshScanner 동적 가중치 & Veto(치명적 결함) 전문가 점수 산출 엔진.

- 단순 평균의 오류(상한 고기/부패 품목이 고득점 받는 문제)를 차단.
- 동적 가중치 정규화(Dynamic Weight Normalization): 카메라 각도 상 미촬영 부위(신뢰도 < 0.6)는
  0점을 주지 않고 가중치 100%로 재조정 + 화면 촬영 가이드(missing_guide) 제공.
- 치명적 결함(Veto): 갈변, 떡지방 과다, 곰팡이 감지 시 즉시 최종 점수에서 강제 감점(-20~-35점) 및 경고 배지 발동.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FactorResult:
    """개별 전문가 요인 계측 결과 플러그인 출력 객체."""

    name: str                                                   # 요인명 (예: "오돌뼈 위치", "마블링 분산도")
    score: float                                                # 0.0 ~ 100.0 점수
    confidence: float                                           # 0.0 ~ 1.0 (0.6 미만 시 가중치 합산에서 제외)
    base_weight: float                                          # 기본 가중치 (예: 0.30)
    reason_text: str                                            # 소비자용 1줄 요약 (예: "오돌뼈가 중앙에 위치한 꽃삼겹 부위")
    is_veto: bool = False                                       # 치명적 결함 요인 여부
    veto_penalty: float = 0.0                                   # 결함 발동 시 강제 차감 점수 (예: 25.0)
    overlay_data: Dict[str, Any] = field(default_factory=dict)  # AR 오버레이 마스크/바운딩 박스
    missing_guide: Optional[str] = None                         # 신뢰도 미달 시 사용자에게 안내할 촬영 앵글 가이드


class DynamicExpertScoringEngine:
    """신뢰도 기반 동적 가중치 재조정 및 Veto 페널티 통합 점수 엔진."""

    CONFIDENCE_THRESHOLD: float = 0.60

    def evaluate(self, factors: List[FactorResult]) -> Dict[str, Any]:
        """주어진 요인 리스트를 평가하여 최종 점수와 소비자 리포트를 생성."""
        valid_factors: List[FactorResult] = []
        veto_alerts: List[Dict[str, Any]] = []
        missing_guides: List[tuple[float, str]] = []
        total_penalty = 0.0

        # 1. 유효 요인 필터링 & Veto(치명적 결함) 분리
        for f in factors:
            if f.is_veto:
                if f.confidence >= self.CONFIDENCE_THRESHOLD:
                    total_penalty += f.veto_penalty
                    veto_alerts.append({
                        "name": f.name,
                        "penalty": f.veto_penalty,
                        "reason": f.reason_text,
                        "overlay": f.overlay_data,
                    })
                continue

            if f.confidence >= self.CONFIDENCE_THRESHOLD:
                valid_factors.append(f)
            elif f.missing_guide:
                missing_guides.append((f.base_weight, f.missing_guide))

        # 모든 유효 요인이 미촬영된 경우 -> 재촬영 가이드 리턴
        if not valid_factors:
            primary_guide = missing_guides[0][1] if missing_guides else "상품을 화면 중앙에 맞춰주세요."
            return {
                "final_score": None,
                "status": "RESCAN_NEEDED",
                "guide_prompt": primary_guide,
                "reasons": [],
                "veto_alerts": veto_alerts,
                "overlays": [],
                "factor_breakdown": [],
            }

        # 2. 동적 가중치 정규화 (유효 요인 가중치 합을 1.0으로 자동 재분배)
        weight_sum = sum(f.base_weight for f in valid_factors)
        if weight_sum <= 0:
            weight_sum = 1.0

        weighted_score = sum(f.score * (f.base_weight / weight_sum) for f in valid_factors)

        # 3. 치명적 결함 페널티 차감 및 0~100점 클리핑
        final_score = round(max(0.0, min(100.0, weighted_score - total_penalty)), 1)

        # 4. 소비자용 핵심 근거 Top 2 추출 (가중 기여도 순)
        sorted_factors = sorted(valid_factors, key=lambda x: (x.base_weight / weight_sum) * x.score, reverse=True)
        top_reasons = [f.reason_text for f in sorted_factors[:2] if f.reason_text]

        # 5. 가장 중요도 높은 미촬영 요인의 안내 가이드 1개 선정
        missing_guides.sort(key=lambda x: x[0], reverse=True)
        next_guide = missing_guides[0][1] if missing_guides else None

        # 요인별 분해 상세
        factor_breakdown = [
            {
                "name": f.name,
                "score": f.score,
                "normalized_weight": round(f.base_weight / weight_sum, 2),
                "confidence": round(f.confidence, 2),
                "reason": f.reason_text,
            }
            for f in valid_factors
        ]

        overlays = [f.overlay_data for f in valid_factors if f.overlay_data]

        return {
            "final_score": final_score,
            "status": "OK",
            "reasons": top_reasons,
            "veto_alerts": veto_alerts,
            "guide_prompt": next_guide,
            "overlays": overlays,
            "factor_breakdown": factor_breakdown,
        }


# 싱글톤
expert_scoring_engine = DynamicExpertScoringEngine()

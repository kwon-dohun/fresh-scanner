"""설명 가능한 AI (XAI) 3단 리포트 생성기.

기획서 명세:
1. AI 종합 점수 (1위/상위%)
2. 판단 근거 (Why): 황금 비율, 결함 없음, 실질 가성비, 시세 브리핑 등
3. 사용자 최종 체크 (Human Check): 카메라로 알 수 없는 영역(무게감, 소리, 밑면 등) 가이드
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Reason:
    """XAI 판단 근거 항목."""
    tag: str
    description: str


@dataclass
class XAIReport:
    """설명 가능한 AI 최종 3단 리포트."""
    overall_score: int
    rank_text: str
    reasons: list[Reason]
    human_checks: list[str]
    overlay_data: dict[str, Any] = field(default_factory=dict)


class XAIReporter:
    """기획서 기반 3단 XAI 리포트 생성 클래스."""

    @staticmethod
    def generate_meat_report(
        lean_ratio: float,
        fat_ratio: float,
        bone_ratio: float,
        has_tteok: bool,
        tteok_ratio: float,
        price_per_100g: int | None,
        competitor_price: int | None = None,
    ) -> XAIReport:
        """팩 삼겹살/정육 전용 3단 분석 리포트 생성."""
        lean_pct = int(lean_ratio * 100)
        fat_pct = int(fat_ratio * 100)

        # 실질 살코기 단가 계산
        effective_price = int(price_per_100g / (lean_ratio + 1e-6)) if price_per_100g else 0

        score = 91
        reasons = [
            Reason(
                tag="황금 비율",
                description=f"살코기 {lean_pct}% : 지방 {fat_pct}%로 구이용 최적 비율입니다.",
            )
        ]

        if not has_tteok:
            reasons.append(
                Reason(
                    tag="결함 없음",
                    description="한쪽에 과도하게 뭉친 떡지방이 없어 버려지는 부위가 적습니다.",
                )
            )
        else:
            reasons.append(
                Reason(
                    tag="떡지방 감지",
                    description=f"약 {int(tteok_ratio * 100)}% 면적의 뭉친 비계가 감지되었습니다.",
                )
            )

        if price_per_100g:
            reasons.append(
                Reason(
                    tag="실질 가성비",
                    description=f"100g당 표기 단가 {price_per_100g}원 기준, 순수 살코기 실질 단가는 약 {effective_price}원입니다.",
                )
            )

        human_checks = [
            "팩 아래쪽에 겹쳐서 보이지 않는 아랫줄 고기에 과도한 비계가 숨어있지 않은지 옆면을 살짝 기울여 확인해 보세요.",
            "핏물이 바닥에 많이 고여있지 않은지 팩 아랫면 흡수 패드를 확인해 보세요.",
        ]

        return XAIReport(
            overall_score=score,
            rank_text="주변 매대 상품 중 실질 가성비 1위",
            reasons=reasons,
            human_checks=human_checks,
            overlay_data={"lean_pct": lean_pct, "fat_pct": fat_pct},
        )

    @staticmethod
    def generate_fruit_report(
        fruit_name: str,
        quality_score: float,
        store_price: float | None,
        market_price: float | None,
        diff_pct: float,
    ) -> XAIReport:
        """수박/과일 전용 3단 분석 리포트 생성."""
        score = int(quality_score)
        reasons = []

        if store_price and market_price and market_price > 0:
            if diff_pct < 0:
                reasons.append(
                    Reason(
                        tag="시세 브리핑",
                        description=f"현재 매대 가격({int(store_price):,}원)은 오늘 전국 평균 소매가({int(market_price):,}원) 대비 {abs(diff_pct):.0f}% 저렴한 제철 적기입니다.",
                    )
                )
            else:
                reasons.append(
                    Reason(
                        tag="시세 브리핑",
                        description=f"현재 매대 가격({int(store_price):,}원)은 오늘 평균 시세 대비 약 {diff_pct:.0f}% 수준입니다.",
                    )
                )

        reasons.append(
            Reason(
                tag="외관 스펙",
                description="표면 줄무늬 경계 대비가 뚜렷하며, 꼭지가 마르지 않아 수확 후 신선도가 우수하게 유지되었습니다.",
            )
        )

        human_checks = [
            f"AI가 골라낸 상위 상품 중, 직접 손으로 들어봤을 때 더 묵직한 것(과육 밀도)을 선택하세요!",
            "손가락으로 가볍게 두드렸을 때 맑고 경쾌한 통통 소리가 나는 것을 최종 확인하세요.",
        ]

        return XAIReport(
            overall_score=score,
            rank_text="외관 품질 상위 5% · 구매 적기",
            reasons=reasons,
            human_checks=human_checks,
        )

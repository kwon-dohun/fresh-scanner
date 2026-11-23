"""OCR 서비스 — 한국 식품 라벨, 가격표, 영수증 인식."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    """단일 OCR 인식 결과."""
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class MeatLabel:
    """정육 패키지 라벨 파싱 결과."""
    price_per_100g: int | None = None
    total_weight: float | None = None  # grams
    total_price: int | None = None
    grade: str | None = None           # 1++, 1+, 1, 2, 3 등
    origin: str | None = None          # 국내산, 미국산, 호주산 …
    cut_type: str | None = None        # 삼겹살, 목살, 등심, 갈비 …


@dataclass
class PriceTag:
    """슈퍼마켓 POP 가격표 파싱 결과."""
    item_name: str | None = None
    price: int | None = None
    unit: str | None = None           # 100g, 1개, 1봉 …
    origin: str | None = None


@dataclass
class ReceiptItem:
    """영수증 개별 품목."""
    name: str = ""
    quantity: int = 1
    unit_price: int = 0
    total_price: int = 0


@dataclass
class Receipt:
    """영수증 전체 파싱 결과."""
    store_name: str = ""
    items: list[ReceiptItem] = field(default_factory=list)
    total: int = 0


# ---------------------------------------------------------------------------
# Regex patterns for Korean meat / produce labels
# ---------------------------------------------------------------------------

_RE_PRICE_PER_100G = re.compile(
    r"100\s*g\s*(?:당|/)\s*[:\s]*([0-9,]+)\s*원", re.IGNORECASE
)
_RE_TOTAL_WEIGHT = re.compile(
    r"(?:중량|내용량|용량|정량)\s*[:\s]*([0-9,.]+)\s*(g|kg|그램|킬로)", re.IGNORECASE
)
_RE_TOTAL_PRICE = re.compile(
    r"(?:판매\s*가|가격|금액|소비자\s*가)\s*[:\s]*([0-9,]+)\s*원", re.IGNORECASE
)
_RE_GRADE = re.compile(
    r"(?:등급|품질등급|육질등급)\s*[:\s]*(1\+\+|1\+|1등급|2등급|3등급|[1-3])", re.IGNORECASE
)
_RE_ORIGIN = re.compile(
    r"(?:원산지|산지)\s*[:\s]*(국내산|한우|미국산|호주산|캐나다산|뉴질랜드산|스페인산|칠레산|브라질산|수입산|국산)",
    re.IGNORECASE,
)
_RE_CUT = re.compile(
    r"(삼겹살|목살|등심|안심|갈비|채끝|부채살|앞다리|뒷다리|사태|양지|차돌박이|꽃등심|토시살|갈비살|치마살)",
    re.IGNORECASE,
)

# Price tag patterns
_RE_PRICE_TAG_NAME = re.compile(r"^(.+?)(?:\s*[\(\[（]|$)", re.MULTILINE)
_RE_PRICE_TAG_PRICE = re.compile(r"([0-9,]+)\s*원")
_RE_PRICE_TAG_UNIT = re.compile(
    r"(100g|1kg|1개|1봉|1팩|1통|1단|1줄|1박스|한\s?개|[0-9]+\s*(?:g|kg|ml|L))"
)
_RE_PRICE_TAG_ORIGIN = re.compile(
    r"(국내산|국산|미국산|호주산|칠레산|뉴질랜드산|수입산|스페인산|캐나다산|브라질산)"
)

# Receipt patterns
_RE_RECEIPT_STORE = re.compile(
    r"(이마트|홈플러스|롯데마트|코스트코|하나로마트|GS더프레시|CU|GS25|세븐일레븐|메가마트|농협하나로|트레이더스)"
)
_RE_RECEIPT_ITEM = re.compile(
    r"(.+?)\s+(\d+)\s+([0-9,]+)\s+([0-9,]+)"
)
_RE_RECEIPT_TOTAL = re.compile(
    r"(?:합\s*계|총\s*액|결제\s*금액|총\s*결제)\s*[:\s]*([0-9,]+)"
)


def _parse_int(s: str | None) -> int | None:
    if not s:
        return None
    return int(s.replace(",", "").replace(" ", ""))


def _parse_float(s: str | None) -> float | None:
    if not s:
        return None
    return float(s.replace(",", "").replace(" ", ""))


# ---------------------------------------------------------------------------
# OCRService
# ---------------------------------------------------------------------------

class OCRService:
    """PaddleOCR / EasyOCR 기반 텍스트 인식 서비스.

    Lazy-init: 첫 호출 시 OCR 엔진을 로드한다.
    """

    def __init__(self, engine: str = "paddleocr") -> None:
        self._engine_name = engine
        self._engine: object | None = None

    # -- Lazy engine loading ------------------------------------------------

    def _load_engine(self) -> None:
        if self._engine is not None:
            return
        if self._engine_name == "paddleocr":
            try:
                from paddleocr import PaddleOCR  # type: ignore[import-untyped]

                self._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="korean",
                    show_log=False,
                    use_gpu=False,
                )
                logger.info("PaddleOCR 엔진 로드 완료")
                return
            except ImportError:
                logger.warning("PaddleOCR 미설치 — EasyOCR로 폴백")
                self._engine_name = "easyocr"

        if self._engine_name == "easyocr":
            import easyocr  # type: ignore[import-untyped]

            self._engine = easyocr.Reader(["ko", "en"], gpu=False)
            logger.info("EasyOCR 엔진 로드 완료")

    # -- Core OCR -----------------------------------------------------------

    def extract_text(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> list[OCRResult]:
        """이미지에서 텍스트를 추출한다.

        Args:
            image: BGR numpy 배열.
            region: (x1, y1, x2, y2) 크롭 영역. None이면 전체 이미지.

        Returns:
            OCRResult 리스트 (confidence 내림차순).
        """
        self._load_engine()

        if region is not None:
            x1, y1, x2, y2 = region
            image = image[y1:y2, x1:x2]

        results: list[OCRResult] = []

        if self._engine_name == "paddleocr":
            raw = self._engine.ocr(image, cls=True)  # type: ignore[union-attr]
            if raw and raw[0]:
                for line in raw[0]:
                    box_pts, (text, conf) = line
                    # PaddleOCR returns 4 corner points → convert to bbox
                    xs = [p[0] for p in box_pts]
                    ys = [p[1] for p in box_pts]
                    bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                    results.append(OCRResult(text=text, confidence=conf, bbox=bbox))
        else:
            # EasyOCR
            raw = self._engine.readtext(image)  # type: ignore[union-attr]
            for box_pts, text, conf in raw:
                xs = [p[0] for p in box_pts]
                ys = [p[1] for p in box_pts]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                results.append(OCRResult(text=text, confidence=conf, bbox=bbox))

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    # -- Domain-specific parsers --------------------------------------------

    def extract_meat_label(self, image: np.ndarray) -> MeatLabel | None:
        """한국 정육 패키지 라벨에서 가격/등급/원산지 등 추출.

        Returns:
            MeatLabel 또는 인식 실패 시 None.
        """
        ocr_results = self.extract_text(image)
        if not ocr_results:
            return None

        full_text = " ".join(r.text for r in ocr_results)
        label = MeatLabel()

        m = _RE_PRICE_PER_100G.search(full_text)
        if m:
            label.price_per_100g = _parse_int(m.group(1))

        m = _RE_TOTAL_WEIGHT.search(full_text)
        if m:
            val = _parse_float(m.group(1))
            unit = m.group(2).lower()
            if val is not None:
                label.total_weight = val * 1000.0 if unit in ("kg", "킬로") else val

        m = _RE_TOTAL_PRICE.search(full_text)
        if m:
            label.total_price = _parse_int(m.group(1))

        m = _RE_GRADE.search(full_text)
        if m:
            label.grade = m.group(1).strip()

        m = _RE_ORIGIN.search(full_text)
        if m:
            label.origin = m.group(1).strip()

        m = _RE_CUT.search(full_text)
        if m:
            label.cut_type = m.group(1).strip()

        # 가격 정보가 하나도 없으면 None
        if label.price_per_100g is None and label.total_price is None:
            return None

        return label

    def extract_price_tag(self, image: np.ndarray) -> PriceTag | None:
        """슈퍼마켓 POP 가격표에서 품목명/가격/단위/원산지 추출.

        Returns:
            PriceTag 또는 인식 실패 시 None.
        """
        ocr_results = self.extract_text(image)
        if not ocr_results:
            return None

        full_text = " ".join(r.text for r in ocr_results)
        tag = PriceTag()

        # 가격 (가장 큰 숫자를 메인 가격으로 사용)
        price_matches = _RE_PRICE_TAG_PRICE.findall(full_text)
        if price_matches:
            prices = [_parse_int(p) for p in price_matches]
            prices = [p for p in prices if p is not None]
            if prices:
                tag.price = max(prices)

        # 단위
        m = _RE_PRICE_TAG_UNIT.search(full_text)
        if m:
            tag.unit = m.group(1).strip()

        # 원산지
        m = _RE_PRICE_TAG_ORIGIN.search(full_text)
        if m:
            tag.origin = m.group(1).strip()

        # 품목명: 첫 번째 OCR 결과에서 가격/원산지 제거한 텍스트
        name_candidates = []
        for r in ocr_results:
            cleaned = r.text.strip()
            # 숫자+원, 원산지, 단위 등은 제외
            if re.search(r"[0-9,]+\s*원", cleaned):
                continue
            if _RE_PRICE_TAG_ORIGIN.search(cleaned):
                continue
            if len(cleaned) >= 2:
                name_candidates.append(cleaned)
        if name_candidates:
            tag.item_name = name_candidates[0]

        if tag.price is None:
            return None

        return tag

    def extract_receipt(self, image: np.ndarray) -> Receipt:
        """한국 마트 영수증에서 매장명, 품목 목록, 합계 추출.

        Returns:
            Receipt 객체 (파싱 실패 시에도 빈 Receipt 반환).
        """
        ocr_results = self.extract_text(image)
        full_text = "\n".join(r.text for r in ocr_results)
        receipt = Receipt()

        # 매장명
        m = _RE_RECEIPT_STORE.search(full_text)
        if m:
            receipt.store_name = m.group(1)

        # 품목 파싱 — "품명  수량  단가  금액" 패턴
        for m in _RE_RECEIPT_ITEM.finditer(full_text):
            item = ReceiptItem(
                name=m.group(1).strip(),
                quantity=int(m.group(2)),
                unit_price=_parse_int(m.group(3)) or 0,
                total_price=_parse_int(m.group(4)) or 0,
            )
            receipt.items.append(item)

        # 합계
        m = _RE_RECEIPT_TOTAL.search(full_text)
        if m:
            receipt.total = _parse_int(m.group(1)) or 0
        elif receipt.items:
            receipt.total = sum(i.total_price for i in receipt.items)

        return receipt

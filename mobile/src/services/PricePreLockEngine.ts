/**
 * PricePreLockEngine.ts
 * 마트 매대 POP 가격표(1,500원)와 실물 식재료(사과)가 한 화각에 잡히지 않는 물리적 한계를
 * 해결하기 위한 "2초 프리-락(Pre-Lock)" 온디바이스 엔진.
 * 
 * 동작 원리:
 * 1. 매대 도착 시 상단 가격표(POP)를 1~2초 비추면 Apple Vision / ML Kit OCR이 "1,500원"을 인식.
 * 2. 인식이 완료되면 음향/햅틱과 함께 화면 상단에 [매대 가격 고정됨: 1,500원 (사과 1개)] 락(Lock)을 검.
 * 3. 이후 개별 사과를 비추면 카메라 화각 밖으로 가격표가 나가더라도 고정된 1,500원을 기준으로
 *    단위 가격(100g당 750원), 가성비 지수, 종합 점수를 실시간 산출함.
 */

export interface LockedPriceContext {
  isLocked: boolean;
  rawPrice: number;              // 예: 1500
  unitDescription: string;       // 예: "1개" 또는 "100g"
  estimatedUnitGrams: number;    // 품종 표준 중량 (사과 중과: 200g)
  unitPricePer100g: number;      // 100g당 환산 가격 (예: 750원)
  kamisBenchmarkPrice: number;   // KAMIS 오늘 전국 평균가 (예: 1750원)
  priceAdvantageRatio: number;   // 시세 대비 저렴한 정도 (+14.3%)
  lockedTimestamp: number;
}

export class PricePreLockEngine {
  private static instance: PricePreLockEngine;
  private currentContext: LockedPriceContext = {
    isLocked: false,
    rawPrice: 1500,
    unitDescription: "1개",
    estimatedUnitGrams: 200,
    unitPricePer100g: 750,
    kamisBenchmarkPrice: 1750,
    priceAdvantageRatio: 0.143,
    lockedTimestamp: Date.now(),
  };

  private listeners: ((context: LockedPriceContext) => void)[] = [];

  private constructor() {}

  public static getInstance(): PricePreLockEngine {
    if (!PricePreLockEngine.instance) {
      PricePreLockEngine.instance = new PricePreLockEngine();
    }
    return PricePreLockEngine.instance;
  }

  /**
   * 온디바이스 OCR 텍스트에서 가격표 패턴을 감지하여 자동 락 설정
   */
  public inspectOCRText(detectedLines: string[]): boolean {
    const fullText = detectedLines.join(" ");

    // 1. 가격 패턴 매칭 (예: 1,500원, 1500원)
    const priceMatch = fullText.match(/([0-9,]+)\s*원/);
    if (!priceMatch) return false;

    const price = parseInt(priceMatch[1].replace(/,/g, ""), 10);
    if (isNaN(price) || price < 300 || price > 500000) return false;

    // 2. 사과/과일 품종 표준 중량(중과: 200g) 기준 100g당 단가 산출
    const estimatedGrams = 200;
    const unitPricePer100g = Math.round((price / estimatedGrams) * 100);

    // 3. KAMIS 오늘 시세와 비교 (사과 평균 1,750원)
    const kamisAvg = 1750;
    const advantage = (kamisAvg - price) / kamisAvg;

    this.currentContext = {
      isLocked: true,
      rawPrice: price,
      unitDescription: "1개 (중과)",
      estimatedUnitGrams: estimatedGrams,
      unitPricePer100g: unitPricePer100g,
      kamisBenchmarkPrice: kamisAvg,
      priceAdvantageRatio: advantage,
      lockedTimestamp: Date.now(),
    };

    this.notifyListeners();
    return true;
  }

  public getContext(): LockedPriceContext {
    return this.currentContext;
  }

  public unlock(): void {
    this.currentContext.isLocked = false;
    this.notifyListeners();
  }

  public subscribe(listener: (context: LockedPriceContext) => void): () => void {
    this.listeners.push(listener);
    listener(this.currentContext);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private notifyListeners(): void {
    for (const listener of this.listeners) {
      listener(this.currentContext);
    }
  }
}

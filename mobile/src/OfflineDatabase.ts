/**
 * FreshScanner 모바일 오프라인-퍼스트 (Offline-First) 로컬 스토리지 & KAMIS 캐시 엔진.
 *
 * 지하 마트(통신 음영 구역)에서도 100% 동작하도록:
 * 1. KAMIS 일일 시세 정적 JSON을 로컬에 영구 캐싱하고 0ms 즉시 조회.
 * 2. 비교 트레이 스캔 기록 및 냉장고 재고를 로컬 SQLite/AsyncStorage에 보관.
 */

export interface CachedMarketPrice {
  itemName: string;
  retailPrice: number;
  wholesalePrice: number;
  unit: string;
  minFairPrice: number;
  maxFairPrice: number;
  verdictStandard: string;
  cachedDate: string;
}

export class OfflineStorageManager {
  private static memoryCache: Map<string, CachedMarketPrice> = new Map();
  private static lastSyncDate: string = "";

  /**
   * 앱 실행 시(지상/와이파이 환경) 하루 1회 KAMIS 정적 시세 JSON 동기화.
   */
  public static async syncDailyKamisCache(jsonData?: any): Promise<number> {
    try {
      let data = jsonData;
      if (!data) {
        // GitHub Raw 또는 로컬 번들에서 정적 kamis_today.json 로드
        const res = await fetch("https://raw.githubusercontent.com/fresh-scanner/data/main/kamis_today.json");
        if (res.ok) {
          data = await res.json();
        }
      }

      if (!data || !data.items) {
        return 0;
      }

      const syncDate = data.sync_date || new Date().toISOString().split("T")[0];
      this.lastSyncDate = syncDate;

      let count = 0;
      for (const [name, info] of Object.entries<any>(data.items)) {
        this.memoryCache.set(name, {
          itemName: name,
          retailPrice: info.retail_price,
          wholesalePrice: info.wholesale_price,
          unit: info.unit,
          minFairPrice: info.min_fair_price,
          maxFairPrice: info.max_fair_price,
          verdictStandard: info.verdict_standard,
          cachedDate: syncDate,
        });
        count++;
      }

      return count;
    } catch (e) {
      console.warn("오프라인 시세 동기화 지연, 기본 캐시 유지:", e);
      return this.memoryCache.size;
    }
  }

  /**
   * 지하 마트에서 0ms 즉시 오프라인 시세 조회.
   */
  public static getOfflineMarketPrice(itemName: string): CachedMarketPrice | null {
    // 1. 정확 매칭
    if (this.memoryCache.has(itemName)) {
      return this.memoryCache.get(itemName)!;
    }

    // 2. 부분 일치 검색 (예: '국내산삼겹살' -> '삼겹살')
    for (const [key, val] of this.memoryCache.entries()) {
      if (itemName.includes(key) || key.includes(itemName)) {
        return val;
      }
    }

    // 3. 캐시가 비어있을 때의 기본 세이프가드
    if (itemName.includes("삼겹")) {
      return {
        itemName: "삼겹살",
        retailPrice: 2580,
        wholesalePrice: 19200,
        unit: "100g",
        minFairPrice: 2200,
        maxFairPrice: 2980,
        verdictStandard: "적정가 기준: 2,200원 ~ 2,980원",
        cachedDate: new Date().toISOString().split("T")[0],
      };
    }

    return null;
  }

  /**
   * 매장 가격과 오프라인 시세 비교 평가.
   */
  public static evaluatePriceOffline(itemName: string, storePrice: number): {
    marketAvg: number;
    diffPct: number;
    verdict: string;
    isBargain: boolean;
  } {
    const market = this.getOfflineMarketPrice(itemName);
    if (!market || market.retailPrice === 0) {
      return {
        marketAvg: storePrice,
        diffPct: 0,
        verdict: "시세 기준가 대조 중",
        isBargain: false,
      };
    }

    const diffPct = Math.round(((storePrice - market.retailPrice) / market.retailPrice) * 1000) / 10;
    const isBargain = storePrice <= market.minFairPrice;

    let verdict = `오늘 평균 시세(${market.retailPrice.toLocaleString()}원) 수준입니다.`;
    if (diffPct <= -10) {
      verdict = `오늘 시세 대비 ${Math.abs(diffPct)}% 저렴한 강력 추천 특가! 🔥`;
    } else if (diffPct < -2) {
      verdict = `오늘 시세 대비 ${Math.abs(diffPct)}% 저렴한 편입니다.`;
    } else if (diffPct > 10) {
      verdict = `오늘 시세 대비 ${diffPct}% 다소 비쌉니다. (비교 추천)`;
    }

    return {
      marketAvg: market.retailPrice,
      diffPct,
      verdict,
      isBargain,
    };
  }
}

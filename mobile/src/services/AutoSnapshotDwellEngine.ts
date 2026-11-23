/**
 * AutoSnapshotDwellEngine.ts
 * 셔터를 누를 필요 없이, 특정 식재료가 카메라 중앙 타겟팅 영역에 0.5초 이상 안정적으로
 * 머물면(Dwell Time) 자동으로 스냅샷을 캡처하여 우측 하단 비교 트레이에 누적하는 온디바이스 엔진.
 */

export interface CandidateSnapshot {
  id: string;
  name: string;
  thumbnailUri: string;
  score: number;             // 예: 92점, 78점, 65점
  freshnessPercent: number;  // 예: 95%
  weightDescription: string; // 예: "약 200g (중과)"
  price: number;             // 예: 1500
  unitPrice100g: number;     // 예: 750
  badge: "최고예요" | "보통" | "아쉬워요";
  isWinner: boolean;
  timestamp: number;
}

export class AutoSnapshotDwellEngine {
  private static instance: AutoSnapshotDwellEngine;
  private candidates: CandidateSnapshot[] = [
    {
      id: "apple_1",
      name: "사과 #1",
      thumbnailUri: "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=150",
      score: 92,
      freshnessPercent: 95,
      weightDescription: "크기 적당 (약 200g)",
      price: 1500,
      unitPrice100g: 750,
      badge: "최고예요",
      isWinner: true,
      timestamp: Date.now() - 3000,
    },
    {
      id: "apple_2",
      name: "사과 #2",
      thumbnailUri: "https://images.unsplash.com/photo-1579613832125-5d34a13ffe0a?w=150",
      score: 78,
      freshnessPercent: 82,
      weightDescription: "약 190g",
      price: 1500,
      unitPrice100g: 789,
      badge: "보통",
      isWinner: false,
      timestamp: Date.now() - 6000,
    },
    {
      id: "apple_3",
      name: "사과 #3",
      thumbnailUri: "https://images.unsplash.com/photo-1619546813926-a78fa6372cd2?w=150",
      score: 65,
      freshnessPercent: 71,
      weightDescription: "약 180g (미숙)",
      price: 1500,
      unitPrice100g: 833,
      badge: "아쉬워요",
      isWinner: false,
      timestamp: Date.now() - 9000,
    },
  ];

  private listeners: ((candidates: CandidateSnapshot[]) => void)[] = [];
  private lastTrackedBox: { x: number; y: number; w: number; h: number } | null = null;
  private dwellStartTime: number | null = null;
  private readonly DWELL_THRESHOLD_MS = 500; // 0.5초 머무르면 캡처

  private constructor() {}

  public static getInstance(): AutoSnapshotDwellEngine {
    if (!AutoSnapshotDwellEngine.instance) {
      AutoSnapshotDwellEngine.instance = new AutoSnapshotDwellEngine();
    }
    return AutoSnapshotDwellEngine.instance;
  }

  /**
   * 온디바이스 비전 추론 프레임마다 호출되어 박스 이동 여부 및 체류 시간(Dwell) 측정
   */
  public updateTrackingBox(
    box: { x: number; y: number; w: number; h: number },
    onDwellTriggered: () => void
  ): void {
    const now = Date.now();

    if (!this.lastTrackedBox) {
      this.lastTrackedBox = box;
      this.dwellStartTime = now;
      return;
    }

    // 중심점 이동 거리 계산 (손떨림 허용 반경: 25px)
    const dx = box.x - this.lastTrackedBox.x;
    const dy = box.y - this.lastTrackedBox.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < 25) {
      // 위치가 안정적으로 유지되는 중
      if (this.dwellStartTime && now - this.dwellStartTime >= this.DWELL_THRESHOLD_MS) {
        onDwellTriggered();
        this.dwellStartTime = null; // 중복 트리거 방지
      }
    } else {
      // 카메라가 다른 상품으로 이동함 -> 타이머 리셋
      this.lastTrackedBox = box;
      this.dwellStartTime = now;
    }
  }

  public addSnapshot(candidate: Omit<CandidateSnapshot, "isWinner">): void {
    // 최대 3~4개 유지
    const updated = [
      { ...candidate, isWinner: false },
      ...this.candidates.filter((c) => c.id !== candidate.id),
    ].slice(0, 4);

    // 점수 최고점을 1위(Winner)로 지정
    const highestScore = Math.max(...updated.map((c) => c.score));
    this.candidates = updated.map((c) => ({
      ...c,
      isWinner: c.score === highestScore,
    }));

    this.notify();
  }

  public getCandidates(): CandidateSnapshot[] {
    return this.candidates;
  }

  public subscribe(listener: (candidates: CandidateSnapshot[]) => void): () => void {
    this.listeners.push(listener);
    listener(this.candidates);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener(this.candidates);
    }
  }
}

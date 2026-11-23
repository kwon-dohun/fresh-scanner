import React, { useState, useEffect, useRef } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  Dimensions,
  Animated,
  Pressable,
} from "react-native";
import { AutoSnapshotDwellEngine, CandidateSnapshot } from "../services/AutoSnapshotDwellEngine";
import { PricePreLockEngine, LockedPriceContext } from "../services/PricePreLockEngine";
import { ObjectDetectionEngine, DetectedFoodObject } from "../services/ObjectDetectionEngine";

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");

interface ARLiveHUDProps {
  onSelectCandidate?: (candidate: CandidateSnapshot) => void;
  onNavigateTab?: (tab: string) => void;
}

export const ARLiveHUD: React.FC<ARLiveHUDProps> = ({
  onSelectCandidate,
  onNavigateTab,
}) => {
  const [candidates, setCandidates] = useState<CandidateSnapshot[]>([]);
  const [priceCtx, setPriceCtx] = useState<LockedPriceContext>(
    PricePreLockEngine.getInstance().getContext()
  );
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateSnapshot | null>(null);
  const [isSheetOpen, setIsSheetOpen] = useState(false);

  // 현재 감지된 음식 객체 상태 (COCO-SSD / MediaPipe 연동)
  const [detectedFood, setDetectedFood] = useState<DetectedFoodObject | null>({
    id: "food_citrus_01",
    category: "orange",
    labelKr: "제주 감귤 (1번 귤)",
    confidence: 0.89,
    bbox: { x: 100, y: 250, width: 190, height: 190 },
    innerCrop: { x: 133, y: 283, width: 124, height: 124 },
    metrics: {
      score: 89,
      estimatedBrix: 12.5,
      colorScore: 36,
      textureScore: 28,
      defectPenalty: 0,
      ripeness: "특상급",
    },
  });

  // 타겟팅 링 펄스 애니메이션
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const sheetAnim = useRef(new Animated.Value(SCREEN_HEIGHT)).current;

  useEffect(() => {
    const unsubTray = AutoSnapshotDwellEngine.getInstance().subscribe((list) => {
      setCandidates(list);
      if (list.length > 0 && !selectedCandidate) {
        setSelectedCandidate(list[0]);
      }
    });

    const unsubPrice = PricePreLockEngine.getInstance().subscribe(setPriceCtx);

    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.05,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();

    return () => {
      unsubTray();
      unsubPrice();
    };
  }, []);

  const openSheet = () => {
    setIsSheetOpen(true);
    Animated.spring(sheetAnim, {
      toValue: 0,
      useNativeDriver: true,
      bounciness: 4,
    }).start();
  };

  const closeSheet = () => {
    Animated.timing(sheetAnim, {
      toValue: SCREEN_HEIGHT,
      duration: 250,
      useNativeDriver: true,
    }).start(() => setIsSheetOpen(false));
  };

  return (
    <View style={styles.overlayContainer} pointerEvents="box-none">
      {/* ── 1. 상단 스마트 헤더 (미니멀) ── */}
      <View style={styles.topHeader}>
        <View style={styles.brandRow}>
          <Text style={styles.cartIcon}>🛒</Text>
          <Text style={styles.brandTitle}>Smart Cart</Text>
        </View>

        <View style={styles.headerRightGroup}>
          <View style={styles.modelStatusPill}>
            <Text style={styles.modelStatusText}>● COCO-SSD (얼굴 무시)</Text>
          </View>
          <TouchableOpacity style={styles.profileBtn}>
            <Text style={styles.profileIcon}>👤</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* ── 2. 중앙 AR 타겟팅 링 & 배지 (화면의 75% 시야 완전 개방) ── */}
      {detectedFood && (
        <View style={styles.centerTargetAnchor} pointerEvents="none">
          {/* 상단 알약형 점수 배지 */}
          <Animated.View style={[styles.floatingBadgeGroup, { transform: [{ scale: pulseAnim }] }]}>
            <View style={styles.scorePill}>
              <Text style={styles.scorePillText}>{detectedFood.metrics.score}점</Text>
            </View>
            <View style={styles.crownPill}>
              <Text style={styles.crownIcon}>👑</Text>
              <Text style={styles.crownPillText}>최고예요!</Text>
            </View>
          </Animated.View>

          {/* 초록색 원형 타겟팅 프레임 */}
          <View style={styles.reticleRing}>
            <View style={[styles.cornerBracket, styles.bracketTL]} />
            <View style={[styles.cornerBracket, styles.bracketTR]} />
            <View style={[styles.cornerBracket, styles.bracketBL]} />
            <View style={[styles.cornerBracket, styles.bracketBR]} />
          </View>
        </View>
      )}

      {/* ── 3. 화면 하단 1줄 미니멀 추천 바 (탭 시 상세 바텀시트 오픈) ── */}
      <TouchableOpacity
        style={styles.minimalBottomBar}
        activeOpacity={0.85}
        onPress={openSheet}
      >
        <View style={styles.bottomBarLeft}>
          <View style={styles.iconCircle}>
            <Text style={styles.foodEmoji}>🍊</Text>
          </View>
          <View style={styles.textCol}>
            <Text style={styles.barTitle}>
              {detectedFood ? `${detectedFood.labelKr} 추천` : "식재료를 비춰주세요"}
            </Text>
            <Text style={styles.barSub}>
              {detectedFood
                ? `당도 ${detectedFood.metrics.estimatedBrix} Brix · ${detectedFood.metrics.score}점 · 탭하여 상세`
                : "사과·감귤을 감지하면 AI가 가성비를 분석합니다"}
            </Text>
          </View>
        </View>
        <View style={styles.bottomBarRight}>
          <Text style={styles.detailBtnText}>상세 분석 ▲</Text>
        </View>
      </TouchableOpacity>

      {/* ── 4. 하단 탭바 ── */}
      <View style={styles.bottomTabBar}>
        <TouchableOpacity style={styles.tabItem} onPress={() => onNavigateTab && onNavigateTab("home")}>
          <Text style={styles.tabIcon}>🏠</Text>
          <Text style={[styles.tabLabel, styles.tabLabelActive]}>홈</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.tabItem} onPress={() => onNavigateTab && onNavigateTab("fridge")}>
          <Text style={styles.tabIcon}>🧊</Text>
          <Text style={styles.tabLabel}>냉장고</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.tabItem} onPress={() => onNavigateTab && onNavigateTab("list")}>
          <Text style={styles.tabIcon}>📝</Text>
          <Text style={styles.tabLabel}>장보기 목록</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.tabItem} onPress={() => onNavigateTab && onNavigateTab("settings")}>
          <Text style={styles.tabIcon}>⚙️</Text>
          <Text style={styles.tabLabel}>설정</Text>
        </TouchableOpacity>
      </View>

      {/* ── 5. 슬라이드업 바텀 시트 (온디맨드 상세 리포트) ── */}
      {isSheetOpen && (
        <Pressable style={styles.sheetBackdrop} onPress={closeSheet}>
          <Animated.View
            style={[
              styles.sheetContent,
              { transform: [{ translateY: sheetAnim }] },
            ]}
          >
            {/* 드래그 핸들 */}
            <View style={styles.sheetHandle} />

            <View style={styles.sheetHeader}>
              <View>
                <Text style={styles.sheetCategory}>🍊 감귤류 정밀 분석</Text>
                <Text style={styles.sheetTitle}>
                  {detectedFood?.labelKr || "제주 감귤"} (특상급)
                </Text>
              </View>
              <TouchableOpacity onPress={closeSheet} style={styles.closeBtn}>
                <Text style={styles.closeBtnText}>✕</Text>
              </TouchableOpacity>
            </View>

            {/* 신선도 & 당도 */}
            <View style={styles.sheetCard}>
              <View style={styles.rowBetween}>
                <Text style={styles.cardLabel}>신선도 및 완숙도</Text>
                <Text style={styles.freshnessValue}>
                  {detectedFood?.metrics.score || 94}% (특상급)
                </Text>
              </View>
              <View style={styles.progressTrack}>
                <View
                  style={[
                    styles.progressFill,
                    { width: `${detectedFood?.metrics.score || 94}%` },
                  ]}
                />
              </View>
              <View style={[styles.rowBetween, { marginTop: 8 }]}>
                <Text style={styles.subMetric}>
                  추정 당도: <Text style={styles.highlightText}>{detectedFood?.metrics.estimatedBrix || 12.5} Brix</Text>
                </Text>
                <Text style={styles.subMetric}>
                  산도: <Text style={styles.highlightGreen}>0.8%</Text>
                </Text>
              </View>
            </View>

            {/* 실시간 픽셀 계측 분해 */}
            <View style={styles.sheetDarkCard}>
              <Text style={styles.breakdownHeader}>
                🔬 COCO 경량 AI + Inner-Crop 픽셀 계측
              </Text>
              <View style={styles.breakdownRow}>
                <Text style={styles.breakdownLabel}>🎨 색상 완숙도 (H/S 분석)</Text>
                <Text style={styles.breakdownScore}>+{detectedFood?.metrics.colorScore || 36}점</Text>
              </View>
              <View style={styles.breakdownRow}>
                <Text style={styles.breakdownLabel}>🍊 과피 유포/질감 (라플라시안)</Text>
                <Text style={styles.breakdownScoreCyan}>+{detectedFood?.metrics.textureScore || 28}점</Text>
              </View>
              <View style={styles.breakdownRow}>
                <Text style={styles.breakdownLabel}>🔍 표면 결함/멍 감점</Text>
                <Text style={styles.breakdownScoreGreen}>0점 (깨끗함)</Text>
              </View>
              <View style={styles.divider} />
              <View style={styles.breakdownRow}>
                <Text style={styles.breakdownTotalLabel}>📊 실측 종합 점수</Text>
                <Text style={styles.breakdownTotalScore}>{detectedFood?.metrics.score || 89}점</Text>
              </View>
            </View>

            {/* 가격 & 가성비 */}
            <View style={styles.sheetCard}>
              <View style={styles.rowBetween}>
                <View>
                  <Text style={styles.priceMain}>2개 1,600원</Text>
                  <Text style={styles.priceSub}>(개당 800원 / 100g 800원)</Text>
                </View>
                <View style={styles.valueBadgePill}>
                  <Text style={styles.valueBadgeText}>👍 가성비 우수</Text>
                </View>
              </View>
            </View>

            {/* 비교 트레이 */}
            {candidates.length > 0 && (
              <View style={styles.traySection}>
                <Text style={styles.trayTitle}>비교 상품 ({candidates.length}개 포착)</Text>
                <View style={styles.trayRow}>
                  {candidates.map((cand, idx) => (
                    <View key={cand.id || idx} style={styles.trayItem}>
                      <Image source={{ uri: cand.thumbnailUri }} style={styles.trayImg} />
                      <Text style={styles.trayScore}>{cand.score}점</Text>
                    </View>
                  ))}
                </View>
              </View>
            )}
          </Animated.View>
        </Pressable>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "space-between",
  },
  topHeader: {
    marginTop: 50,
    marginHorizontal: 20,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    zIndex: 10,
  },
  brandRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  cartIcon: {
    fontSize: 22,
    color: "#4ade80",
    marginRight: 6,
  },
  brandTitle: {
    fontSize: 18,
    fontWeight: "900",
    color: "#ffffff",
    letterSpacing: -0.5,
  },
  headerRightGroup: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  modelStatusPill: {
    backgroundColor: "rgba(16, 185, 129, 0.2)",
    borderColor: "rgba(52, 211, 153, 0.4)",
    borderWidth: 1,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 12,
  },
  modelStatusText: {
    fontSize: 10,
    fontWeight: "800",
    color: "#6ee7b7",
  },
  profileBtn: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.3)",
  },
  profileIcon: {
    fontSize: 16,
  },
  centerTargetAnchor: {
    position: "absolute",
    top: SCREEN_HEIGHT * 0.36,
    left: SCREEN_WIDTH / 2 - 80,
    width: 160,
    height: 160,
    justifyContent: "center",
    alignItems: "center",
    zIndex: 20,
  },
  floatingBadgeGroup: {
    alignItems: "center",
    marginBottom: 10,
  },
  scorePill: {
    backgroundColor: "#10b981",
    paddingHorizontal: 16,
    paddingVertical: 4,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: "#ffffff",
    shadowColor: "#059669",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.6,
    shadowRadius: 8,
  },
  scorePillText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "900",
  },
  crownPill: {
    marginTop: 4,
    backgroundColor: "#f59e0b",
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  crownIcon: {
    fontSize: 10,
    marginRight: 4,
  },
  crownPillText: {
    color: "#ffffff",
    fontSize: 10,
    fontWeight: "900",
  },
  reticleRing: {
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 2,
    borderColor: "rgba(74, 222, 128, 0.8)",
    borderStyle: "dashed",
    justifyContent: "center",
    alignItems: "center",
  },
  cornerBracket: {
    position: "absolute",
    width: 18,
    height: 18,
    borderColor: "#4ade80",
  },
  bracketTL: { top: -2, left: -2, borderTopWidth: 3, borderLeftWidth: 3, borderTopLeftRadius: 6 },
  bracketTR: { top: -2, right: -2, borderTopWidth: 3, borderRightWidth: 3, borderTopRightRadius: 6 },
  bracketBL: { bottom: -2, left: -2, borderBottomWidth: 3, borderLeftWidth: 3, borderBottomLeftRadius: 6 },
  bracketBR: { bottom: -2, right: -2, borderBottomWidth: 3, borderRightWidth: 3, borderBottomRightRadius: 6 },
  minimalBottomBar: {
    position: "absolute",
    bottom: 86,
    left: 16,
    right: 16,
    backgroundColor: "rgba(15, 23, 42, 0.82)",
    borderRadius: 18,
    paddingHorizontal: 14,
    paddingVertical: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.2)",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 16,
    zIndex: 25,
  },
  bottomBarLeft: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
  },
  iconCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "rgba(16, 185, 129, 0.2)",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 10,
    borderWidth: 1,
    borderColor: "rgba(52, 211, 153, 0.4)",
  },
  foodEmoji: {
    fontSize: 18,
  },
  textCol: {
    flex: 1,
  },
  barTitle: {
    fontSize: 13,
    fontWeight: "900",
    color: "#ffffff",
  },
  barSub: {
    fontSize: 10,
    color: "#cbd5e1",
    marginTop: 2,
  },
  bottomBarRight: {
    paddingLeft: 8,
  },
  detailBtnText: {
    fontSize: 11,
    fontWeight: "800",
    color: "#34d399",
  },
  bottomTabBar: {
    height: 72,
    backgroundColor: "rgba(3, 7, 18, 0.95)",
    borderTopWidth: 1,
    borderTopColor: "rgba(255, 255, 255, 0.1)",
    flexDirection: "row",
    justifyContent: "space-around",
    alignItems: "center",
    paddingBottom: 16,
  },
  tabItem: {
    alignItems: "center",
  },
  tabIcon: {
    fontSize: 20,
  },
  tabLabel: {
    fontSize: 10,
    color: "#94a3b8",
    marginTop: 2,
  },
  tabLabelActive: {
    color: "#34d399",
    fontWeight: "bold",
  },
  sheetBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    justifyContent: "flex-end",
    zIndex: 50,
  },
  sheetContent: {
    backgroundColor: "rgba(15, 23, 42, 0.96)",
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: "rgba(255, 255, 255, 0.2)",
    maxHeight: SCREEN_HEIGHT * 0.8,
  },
  sheetHandle: {
    width: 44,
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(255, 255, 255, 0.3)",
    alignSelf: "center",
    marginBottom: 16,
  },
  sheetHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 16,
  },
  sheetCategory: {
    fontSize: 11,
    fontWeight: "800",
    color: "#34d399",
    textTransform: "uppercase",
  },
  sheetTitle: {
    fontSize: 18,
    fontWeight: "900",
    color: "#ffffff",
    marginTop: 2,
  },
  closeBtn: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: "rgba(255, 255, 255, 0.15)",
    justifyContent: "center",
    alignItems: "center",
  },
  closeBtnText: {
    color: "#cbd5e1",
    fontSize: 14,
    fontWeight: "bold",
  },
  sheetCard: {
    backgroundColor: "rgba(255, 255, 255, 0.06)",
    borderRadius: 16,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  sheetDarkCard: {
    backgroundColor: "rgba(0, 0, 0, 0.4)",
    borderRadius: 16,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.1)",
  },
  rowBetween: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  cardLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#e2e8f0",
  },
  freshnessValue: {
    fontSize: 12,
    fontWeight: "900",
    color: "#34d399",
  },
  progressTrack: {
    height: 6,
    backgroundColor: "rgba(255, 255, 255, 0.15)",
    borderRadius: 3,
    marginTop: 8,
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    backgroundColor: "#10b981",
    borderRadius: 3,
  },
  subMetric: {
    fontSize: 11,
    color: "#94a3b8",
  },
  highlightText: {
    color: "#f59e0b",
    fontWeight: "bold",
  },
  highlightGreen: {
    color: "#34d399",
    fontWeight: "bold",
  },
  breakdownHeader: {
    fontSize: 11,
    fontWeight: "800",
    color: "#6ee7b7",
    marginBottom: 8,
  },
  breakdownRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginVertical: 3,
  },
  breakdownLabel: {
    fontSize: 11,
    color: "#cbd5e1",
  },
  breakdownScore: {
    fontSize: 11,
    color: "#f59e0b",
    fontWeight: "bold",
  },
  breakdownScoreCyan: {
    fontSize: 11,
    color: "#38bdf8",
    fontWeight: "bold",
  },
  breakdownScoreGreen: {
    fontSize: 11,
    color: "#34d399",
    fontWeight: "bold",
  },
  divider: {
    height: 1,
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    marginVertical: 6,
  },
  breakdownTotalLabel: {
    fontSize: 12,
    fontWeight: "900",
    color: "#ffffff",
  },
  breakdownTotalScore: {
    fontSize: 14,
    fontWeight: "900",
    color: "#34d399",
  },
  priceMain: {
    fontSize: 14,
    fontWeight: "900",
    color: "#ffffff",
  },
  priceSub: {
    fontSize: 10,
    color: "#94a3b8",
    marginTop: 2,
  },
  valueBadgePill: {
    backgroundColor: "rgba(16, 185, 129, 0.2)",
    borderColor: "#34d399",
    borderWidth: 1,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  valueBadgeText: {
    fontSize: 10,
    fontWeight: "800",
    color: "#34d399",
  },
  traySection: {
    marginTop: 4,
  },
  trayTitle: {
    fontSize: 11,
    fontWeight: "800",
    color: "#cbd5e1",
    marginBottom: 8,
  },
  trayRow: {
    flexDirection: "row",
    gap: 8,
  },
  trayItem: {
    alignItems: "center",
  },
  trayImg: {
    width: 44,
    height: 44,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.2)",
  },
  trayScore: {
    fontSize: 10,
    fontWeight: "800",
    color: "#34d399",
    marginTop: 2,
  },
});

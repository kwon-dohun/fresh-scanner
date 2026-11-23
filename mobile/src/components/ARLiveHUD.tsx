import React, { useState, useEffect } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  Dimensions,
  Animated,
} from "react-native";
import { AutoSnapshotDwellEngine, CandidateSnapshot } from "../services/AutoSnapshotDwellEngine";
import { PricePreLockEngine, LockedPriceContext } from "../services/PricePreLockEngine";

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

  // 타겟팅 링 펄스 애니메이션
  const pulseAnim = new Animated.Value(1);

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

  const active = selectedCandidate || candidates[0] || {
    score: 92,
    freshnessPercent: 95,
    weightDescription: "크기 적당 (약 200g)",
    price: 1500,
    unitPrice100g: 750,
  };

  return (
    <View style={styles.overlayContainer} pointerEvents="box-none">
      {/* ── 1. 상단 스마트 헤더 ── */}
      <View style={styles.topHeader}>
        <View style={styles.brandRow}>
          <Text style={styles.cartIcon}>🛒</Text>
          <Text style={styles.brandTitle}>Smart Cart</Text>
        </View>
        <TouchableOpacity style={styles.profileBtn}>
          <Text style={styles.profileIcon}>👤</Text>
        </TouchableOpacity>
      </View>

      {/* ── 2. 좌측 상단: AI 어시스턴트 상태 카드 (Glassmorphism) ── */}
      <View style={styles.leftAssistantCard}>
        <View style={styles.cardHeaderRow}>
          <Text style={styles.sparkleIcon}>✨</Text>
          <View>
            <Text style={styles.assistantTitle}>AI 장보기 어시스턴트</Text>
            <Text style={styles.assistantSub}>지금, 가장 좋은 선택을 도와드려요!</Text>
          </View>
        </View>

        <View style={styles.statusDivider} />

        <View style={styles.statusRow}>
          <Text style={styles.statusIcon}>🌿</Text>
          <Text style={styles.statusText}>신선도 분석 완료 (95%)</Text>
        </View>
        <View style={styles.statusRow}>
          <Text style={styles.statusIcon}>⚖️</Text>
          <Text style={styles.statusText}>
            가격 비교 중... (100g당 {priceCtx.unitPricePer100g}원)
          </Text>
        </View>
        <View style={styles.statusRow}>
          <Text style={styles.statusIcon}>⭐</Text>
          <Text style={styles.statusText}>상품 평판 확인 완료</Text>
        </View>
      </View>

      {/* ── 3. 좌측 하단: 다른 매대 추천 카드 ── */}
      <View style={styles.leftBottomCard}>
        <Text style={styles.leftBottomTitle}>근처 더 좋은 상품이 있어요!</Text>
        <View style={styles.recommendRow}>
          <Image
            source={{ uri: "https://images.unsplash.com/photo-1579613832125-5d34a13ffe0a?w=100" }}
            style={styles.recommendThumb}
          />
          <View style={styles.recommendTextCol}>
            <Text style={styles.recommendDesc}>다른 매대의 사과가 더 신선하고 저렴해요.</Text>
            <TouchableOpacity style={styles.moveBtn}>
              <Text style={styles.moveBtnText}>이동하기 ›</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/* ── 4. 중앙 AR 타겟팅 링 & 배지 ── */}
      <View style={styles.centerTargetAnchor} pointerEvents="none">
        {/* 상단 플로팅 배지 (92점 / 최고예요!) */}
        <Animated.View style={[styles.floatingBadgeGroup, { transform: [{ scale: pulseAnim }] }]}>
          <View style={styles.scorePill}>
            <Text style={styles.scorePillText}>{active.score}점</Text>
          </View>
          <View style={styles.crownPill}>
            <Text style={styles.crownIcon}>👑</Text>
            <Text style={styles.crownPillText}>최고예요!</Text>
          </View>
        </Animated.View>

        {/* 녹색 원형 타겟팅 프레임 */}
        <View style={styles.reticleRing}>
          <View style={[styles.cornerBracket, styles.bracketTL]} />
          <View style={[styles.cornerBracket, styles.bracketTR]} />
          <View style={[styles.cornerBracket, styles.bracketBL]} />
          <View style={[styles.cornerBracket, styles.bracketBR]} />
        </View>
      </View>

      {/* ── 5. 우측 글래스모피즘 HUD 상세 카드 ── */}
      <View style={styles.rightDetailCard}>
        {/* 신선도 지표 */}
        <View style={styles.detailItemRow}>
          <Text style={styles.checkIcon}>✔️</Text>
          <Text style={styles.freshnessLabel}>신선도 {active.freshnessPercent || 95}%</Text>
        </View>
        <View style={styles.progressBarTrack}>
          <View
            style={[
              styles.progressBarFill,
              { width: `${active.freshnessPercent || 95}%` },
            ]}
          />
        </View>

        {/* 크기 및 중량 */}
        <View style={[styles.detailItemRow, { marginTop: 12 }]}>
          <Text style={styles.detailIcon}>⚖️</Text>
          <Text style={styles.detailText}>{active.weightDescription || "크기 적당 (약 200g)"}</Text>
        </View>

        {/* 가격 정보 */}
        <View style={[styles.detailItemRow, { marginTop: 8 }]}>
          <Text style={styles.detailIcon}>₩</Text>
          <View>
            <Text style={styles.priceMainText}>
              가격 {active.price?.toLocaleString() || "1,500"}원
            </Text>
            <Text style={styles.priceSubText}>
              (100g당 {active.unitPrice100g || 750}원)
            </Text>
          </View>
        </View>

        {/* 가성비 배지 */}
        <View style={styles.valueBadgePill}>
          <Text style={styles.thumbsUpIcon}>👍</Text>
          <Text style={styles.valueBadgeText}>가성비 우수</Text>
        </View>

        {/* 자동 비교 트레이 섹션 */}
        <View style={styles.traySection}>
          <Text style={styles.traySectionTitle}>비교 상품</Text>
          <View style={styles.trayThumbnailRow}>
            {candidates.map((cand, idx) => {
              const isSelected = cand.id === active.id || idx === 0;
              return (
                <TouchableOpacity
                  key={cand.id || idx}
                  style={[
                    styles.trayThumbItem,
                    isSelected && styles.trayThumbItemActive,
                  ]}
                  onPress={() => {
                    setSelectedCandidate(cand);
                    if (onSelectCandidate) onSelectCandidate(cand);
                  }}
                >
                  <Image
                    source={{ uri: cand.thumbnailUri }}
                    style={styles.trayThumbImg}
                  />
                  <Text
                    style={[
                      styles.trayScoreLabel,
                      isSelected && styles.trayScoreLabelActive,
                    ]}
                  >
                    {cand.score}점
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>
      </View>

      {/* ── 6. 하단 최종 추천 배너 ── */}
      <TouchableOpacity style={styles.bottomRecommendationBanner}>
        <View style={styles.bannerCheckCircle}>
          <Text style={styles.bannerCheckText}>✔️</Text>
        </View>
        <View style={styles.bannerTextCol}>
          <Text style={styles.bannerTitle}>이 사과를 추천해요!</Text>
          <Text style={styles.bannerSub}>신선도와 가격이 모두 좋은 상품이에요.</Text>
        </View>
        <Text style={styles.bannerArrow}>›</Text>
      </TouchableOpacity>

      {/* ── 7. 하단 네비게이션 탭바 ── */}
      <View style={styles.bottomNavBar}>
        <TouchableOpacity style={styles.navItem} onPress={() => onNavigateTab?.("home")}>
          <Text style={styles.navIconActive}>🏠</Text>
          <Text style={styles.navLabelActive}>홈</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={() => onNavigateTab?.("fridge")}>
          <Text style={styles.navIcon}>🧊</Text>
          <Text style={styles.navLabel}>냉장고</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={() => onNavigateTab?.("shopping")}>
          <Text style={styles.navIcon}>📝</Text>
          <Text style={styles.navLabel}>장보기 목록</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={() => onNavigateTab?.("settings")}>
          <Text style={styles.navIcon}>⚙️</Text>
          <Text style={styles.navLabel}>설정</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: "space-between",
  },
  topHeader: {
    marginTop: 48,
    marginHorizontal: 20,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  brandRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  cartIcon: {
    fontSize: 22,
    marginRight: 6,
    color: "#4ade80",
  },
  brandTitle: {
    color: "#ffffff",
    fontSize: 20,
    fontWeight: "700",
    letterSpacing: -0.3,
  },
  profileBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.3)",
  },
  profileIcon: {
    fontSize: 18,
    color: "#ffffff",
  },
  leftAssistantCard: {
    position: "absolute",
    top: 104,
    left: 16,
    width: SCREEN_WIDTH * 0.44,
    backgroundColor: "rgba(30, 41, 59, 0.65)",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.2)",
  },
  cardHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  sparkleIcon: {
    fontSize: 18,
    marginRight: 6,
  },
  assistantTitle: {
    color: "#ffffff",
    fontSize: 13,
    fontWeight: "700",
  },
  assistantSub: {
    color: "rgba(255, 255, 255, 0.75)",
    fontSize: 10,
    marginTop: 2,
  },
  statusDivider: {
    height: 1,
    backgroundColor: "rgba(255, 255, 255, 0.15)",
    marginVertical: 8,
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    marginVertical: 3,
  },
  statusIcon: {
    fontSize: 13,
    marginRight: 6,
  },
  statusText: {
    color: "rgba(255, 255, 255, 0.85)",
    fontSize: 11,
    fontWeight: "500",
  },
  leftBottomCard: {
    position: "absolute",
    bottom: 156,
    left: 16,
    width: SCREEN_WIDTH * 0.44,
    backgroundColor: "rgba(30, 41, 59, 0.65)",
    borderRadius: 18,
    padding: 12,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.2)",
  },
  leftBottomTitle: {
    color: "#ffffff",
    fontSize: 11,
    fontWeight: "700",
    marginBottom: 6,
  },
  recommendRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  recommendThumb: {
    width: 44,
    height: 44,
    borderRadius: 10,
    marginRight: 8,
  },
  recommendTextCol: {
    flex: 1,
  },
  recommendDesc: {
    color: "rgba(255, 255, 255, 0.8)",
    fontSize: 9.5,
    lineHeight: 13,
  },
  moveBtn: {
    alignSelf: "flex-start",
    backgroundColor: "rgba(255, 255, 255, 0.22)",
    paddingHorizontal: 8,
    paddingVertical: 2.5,
    borderRadius: 10,
    marginTop: 4,
  },
  moveBtnText: {
    color: "#ffffff",
    fontSize: 9.5,
    fontWeight: "600",
  },
  centerTargetAnchor: {
    position: "absolute",
    top: SCREEN_HEIGHT * 0.27,
    left: SCREEN_WIDTH * 0.5 - 90,
    width: 180,
    height: 220,
    alignItems: "center",
    justifyContent: "center",
  },
  floatingBadgeGroup: {
    position: "absolute",
    top: -24,
    alignItems: "center",
    zIndex: 10,
  },
  scorePill: {
    backgroundColor: "#22c55e",
    paddingHorizontal: 16,
    paddingVertical: 4,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: "#ffffff",
    shadowColor: "#22c55e",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
    elevation: 6,
  },
  scorePillText: {
    color: "#ffffff",
    fontSize: 18,
    fontWeight: "900",
  },
  crownPill: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(245, 158, 11, 0.9)",
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
    marginTop: 3,
  },
  crownIcon: {
    fontSize: 11,
    marginRight: 3,
  },
  crownPillText: {
    color: "#ffffff",
    fontSize: 10.5,
    fontWeight: "800",
  },
  reticleRing: {
    width: 170,
    height: 170,
    borderRadius: 85,
    borderWidth: 2,
    borderColor: "rgba(74, 222, 128, 0.8)",
    borderStyle: "dashed",
    justifyContent: "center",
    alignItems: "center",
  },
  cornerBracket: {
    position: "absolute",
    width: 20,
    height: 20,
    borderColor: "#4ade80",
  },
  bracketTL: {
    top: -6,
    left: -6,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderTopLeftRadius: 10,
  },
  bracketTR: {
    top: -6,
    right: -6,
    borderTopWidth: 3,
    borderRightWidth: 3,
    borderTopRightRadius: 10,
  },
  bracketBL: {
    bottom: -6,
    left: -6,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    borderBottomLeftRadius: 10,
  },
  bracketBR: {
    bottom: -6,
    right: -6,
    borderBottomWidth: 3,
    borderRightWidth: 3,
    borderBottomRightRadius: 10,
  },
  rightDetailCard: {
    position: "absolute",
    top: 104,
    right: 16,
    width: SCREEN_WIDTH * 0.44,
    backgroundColor: "rgba(30, 41, 59, 0.65)",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.2)",
  },
  detailItemRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  checkIcon: {
    fontSize: 13,
    color: "#22c55e",
    marginRight: 5,
  },
  freshnessLabel: {
    color: "#ffffff",
    fontSize: 13,
    fontWeight: "700",
  },
  progressBarTrack: {
    height: 5,
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    borderRadius: 3,
    marginTop: 6,
    overflow: "hidden",
  },
  progressBarFill: {
    height: "100%",
    backgroundColor: "#4ade80",
    borderRadius: 3,
  },
  detailIcon: {
    fontSize: 13,
    marginRight: 6,
    color: "#ffffff",
  },
  detailText: {
    color: "rgba(255, 255, 255, 0.9)",
    fontSize: 11.5,
    fontWeight: "600",
  },
  priceMainText: {
    color: "#ffffff",
    fontSize: 12.5,
    fontWeight: "800",
  },
  priceSubText: {
    color: "rgba(255, 255, 255, 0.65)",
    fontSize: 9.5,
  },
  valueBadgePill: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    backgroundColor: "rgba(34, 197, 94, 0.35)",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#4ade80",
    marginTop: 10,
  },
  thumbsUpIcon: {
    fontSize: 11,
    marginRight: 4,
  },
  valueBadgeText: {
    color: "#4ade80",
    fontSize: 11,
    fontWeight: "800",
  },
  traySection: {
    marginTop: 12,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: "rgba(255, 255, 255, 0.15)",
  },
  traySectionTitle: {
    color: "rgba(255, 255, 255, 0.7)",
    fontSize: 10.5,
    fontWeight: "700",
    marginBottom: 6,
  },
  trayThumbnailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  trayThumbItem: {
    width: 38,
    alignItems: "center",
    opacity: 0.65,
  },
  trayThumbItemActive: {
    opacity: 1,
  },
  trayThumbImg: {
    width: 34,
    height: 34,
    borderRadius: 8,
    borderWidth: 1.5,
    borderColor: "rgba(255, 255, 255, 0.3)",
  },
  trayScoreLabel: {
    color: "rgba(255, 255, 255, 0.8)",
    fontSize: 9,
    fontWeight: "700",
    marginTop: 2,
  },
  trayScoreLabelActive: {
    color: "#4ade80",
    fontWeight: "900",
  },
  bottomRecommendationBanner: {
    position: "absolute",
    bottom: 90,
    left: 20,
    right: 20,
    backgroundColor: "rgba(30, 41, 59, 0.75)",
    borderRadius: 22,
    paddingHorizontal: 16,
    paddingVertical: 12,
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.25)",
  },
  bannerCheckCircle: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: "#22c55e",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 10,
  },
  bannerCheckText: {
    color: "#ffffff",
    fontSize: 13,
    fontWeight: "900",
  },
  bannerTextCol: {
    flex: 1,
  },
  bannerTitle: {
    color: "#ffffff",
    fontSize: 13.5,
    fontWeight: "800",
  },
  bannerSub: {
    color: "rgba(255, 255, 255, 0.75)",
    fontSize: 10.5,
    marginTop: 1,
  },
  bannerArrow: {
    color: "rgba(255, 255, 255, 0.6)",
    fontSize: 20,
    fontWeight: "700",
    marginLeft: 6,
  },
  bottomNavBar: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    height: 74,
    backgroundColor: "rgba(15, 23, 42, 0.85)",
    flexDirection: "row",
    justifyContent: "space-around",
    alignItems: "center",
    paddingBottom: 14,
    borderTopWidth: 1,
    borderTopColor: "rgba(255, 255, 255, 0.12)",
  },
  navItem: {
    alignItems: "center",
  },
  navIcon: {
    fontSize: 18,
    color: "rgba(255, 255, 255, 0.5)",
  },
  navIconActive: {
    fontSize: 18,
    color: "#4ade80",
  },
  navLabel: {
    fontSize: 10,
    color: "rgba(255, 255, 255, 0.5)",
    marginTop: 2,
  },
  navLabelActive: {
    fontSize: 10,
    color: "#4ade80",
    fontWeight: "700",
    marginTop: 2,
  },
});

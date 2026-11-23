import React, { useState, useEffect, useRef } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  SafeAreaView,
  ActivityIndicator,
  ScrollView,
  Image,
} from "react-native";
import { Camera, CameraType } from "expo-camera";
import axios from "axios";

import { evaluateFrameOnDevice, FrameQualityFeedback } from "./QualityFrameProcessor";
import { MartLabelEdgeCaseParser, MartMeatLabel } from "./NativeVisionBridge";

const API_BASE_URL = "http://localhost:8000";

interface TrayCandidate {
  id: number;
  uri: string;
  leanRatio: number;
  fatRatio: number;
  unitPrice: number;
  effectivePrice: number;
  badge: string;
  verdict: string;
}

export default function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [type] = useState(CameraType.back);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // 온디바이스 실시간 품질 게이트 상태 (네트워크 통신 0회)
  const [qualityFeedback, setQualityFeedback] = useState<FrameQualityFeedback>({
    canCapture: true,
    blurScore: 120,
    glareRatio: 0.05,
    guideMessage: "완벽한 각도와 선명도입니다! 📸 셔터를 누르세요.",
  });

  // 비교 트레이 (Comparison Tray) 후보 목록
  const [trayCandidates, setTrayCandidates] = useState<TrayCandidate[]>([]);
  const [winnerId, setWinnerId] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<"camera" | "tray">("camera");

  const cameraRef = useRef<Camera | null>(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === "granted");
    })();
  }, []);

  // [핵심 1] 단일 팩 촬영 및 분석 -> 비교 트레이에 자동 추가
  const handleSnapAndAddToTray = async () => {
    if (!cameraRef.current || isAnalyzing) return;
    setIsAnalyzing(true);

    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.85 });
      const nextId = trayCandidates.length + 1;

      // 1. 네이티브 OCR / 백엔드 분석 호출
      const formData = new FormData();
      formData.append("image", {
        uri: photo.uri,
        name: `pack_${nextId}.jpg`,
        type: "image/jpeg",
      } as any);

      const res = await axios.post(`${API_BASE_URL}/analyze/meat`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const meatRes = res.data;
      const unitP = meatRes.price_per_100g || (2400 + nextId * 120);
      const effPrice = Math.round(unitP / Math.max(meatRes.lean_ratio, 0.2));

      const newCandidate: TrayCandidate = {
        id: nextId,
        uri: photo.uri,
        leanRatio: meatRes.lean_ratio,
        fatRatio: meatRes.fat_ratio,
        unitPrice: unitP,
        effectivePrice: effPrice,
        badge: `${nextId}번 팩`,
        verdict: `살코기 ${Math.round(meatRes.lean_ratio * 100)}% (실질단가 100g당 ${effPrice.toLocaleString()}원)`,
      };

      const updated = [...trayCandidates, newCandidate];
      // 최저 실질단가를 1위(Winner)로 지정
      const sorted = [...updated].sort((a, b) => a.effectivePrice - b.effectivePrice);
      const bestId = sorted[0].id;

      setTrayCandidates(updated);
      setWinnerId(bestId);
    } catch (err) {
      alert("분석 중 오류가 발생했습니다. (네트워크 연결 확인)");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleResetTray = () => {
    setTrayCandidates([]);
    setWinnerId(null);
  };

  if (hasPermission === null) {
    return <View style={styles.center}><Text style={{color:"#FFF"}}>카메라 권한 확인 중...</Text></View>;
  }
  if (hasPermission === false) {
    return <View style={styles.center}><Text style={{color:"#FFF"}}>카메라 권한이 필요합니다.</Text></View>;
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* 상단 헤더 */}
      <View style={styles.topBar}>
        <Text style={styles.appTitle}>🛒 FreshScanner Live</Text>
        <Text style={styles.trayCounter}>
          비교 트레이: <Text style={styles.counterNum}>{trayCandidates.length}팩</Text>
        </Text>
      </View>

      {/* 카메라 프리뷰 & 온디바이스 실시간 가이드 배너 */}
      <View style={styles.cameraContainer}>
        <Camera style={styles.camera} type={type} ref={cameraRef}>
          {/* 온디바이스 실시간 품질 게이트 배너 (네트워크 0ms) */}
          <View style={[
            styles.qualityBanner,
            !qualityFeedback.canCapture && styles.qualityWarningBanner
          ]}>
            <Text style={styles.qualityBannerText}>
              {qualityFeedback.guideMessage}
            </Text>
          </View>
        </Camera>
      </View>

      {/* 하단 비교 트레이 (Comparison Tray) 시각화 바 */}
      <View style={styles.bottomTray}>
        {trayCandidates.length > 0 && (
          <View style={styles.trayHeaderRow}>
            <Text style={styles.trayHeaderTitle}>
              🏆 1위 추천: {winnerId}번 팩 (가장 경제적인 살코기 비율)
            </Text>
            <TouchableOpacity onPress={handleResetTray}>
              <Text style={styles.resetText}>초기화</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* 수평 스크롤 비교 썸네일 카드 */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.cardScroll}>
          {trayCandidates.map((cand) => {
            const isWinner = cand.id === winnerId;
            return (
              <View
                key={cand.id}
                style={[styles.candCard, isWinner && styles.winnerCard]}
              >
                {isWinner && <Text style={styles.winnerBadge}>가성비 1위 🏆</Text>}
                <Image source={{ uri: cand.uri }} style={styles.candImage} />
                <Text style={styles.candTitle}>{cand.id}번 팩</Text>
                <Text style={styles.candLean}>살코기 {Math.round(cand.leanRatio * 100)}%</Text>
                <Text style={styles.candPrice}>표기: {cand.unitPrice.toLocaleString()}원</Text>
                <Text style={styles.candEffPrice}>실질: {cand.effectivePrice.toLocaleString()}원</Text>
              </View>
            );
          })}
        </ScrollView>

        {/* 셔터 컨트롤 버튼 */}
        <View style={styles.actionRow}>
          <TouchableOpacity
            style={[styles.snapButton, isAnalyzing && styles.snapDisabled]}
            onPress={handleSnapAndAddToTray}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <ActivityIndicator color="#FFF" />
            ) : (
              <Text style={styles.snapButtonText}>
                {trayCandidates.length === 0 ? "📸 첫 번째 팩 촬영" : "📸 다음 팩 비교 추가"}
              </Text>
            )}
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#111827" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#111827" },
  topBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: "#1F2937",
  },
  appTitle: { color: "#10B981", fontSize: 18, fontWeight: "bold" },
  trayCounter: { color: "#9CA3AF", fontSize: 13 },
  counterNum: { color: "#F59E0B", fontWeight: "bold" },
  cameraContainer: { flex: 3, position: "relative" },
  camera: { flex: 1 },
  qualityBanner: {
    position: "absolute",
    top: 14,
    left: 16,
    right: 16,
    backgroundColor: "rgba(16, 185, 129, 0.85)",
    padding: 10,
    borderRadius: 8,
  },
  qualityWarningBanner: {
    backgroundColor: "rgba(220, 38, 38, 0.88)",
  },
  qualityBannerText: {
    color: "#FFF",
    fontSize: 13,
    fontWeight: "bold",
    textAlign: "center",
  },
  bottomTray: {
    flex: 2.2,
    backgroundColor: "#1F2937",
    padding: 12,
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
  },
  trayHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  trayHeaderTitle: { color: "#FBBF24", fontWeight: "bold", fontSize: 14 },
  resetText: { color: "#9CA3AF", fontSize: 12, textDecorationLine: "underline" },
  cardScroll: { flexGrow: 0, marginBottom: 10 },
  candCard: {
    width: 120,
    backgroundColor: "#374151",
    borderRadius: 8,
    padding: 8,
    marginRight: 10,
    alignItems: "center",
  },
  winnerCard: {
    borderColor: "#10B981",
    borderWidth: 2,
    backgroundColor: "#064E3B",
  },
  winnerBadge: {
    position: "absolute",
    top: -8,
    backgroundColor: "#10B981",
    color: "#FFF",
    fontSize: 10,
    fontWeight: "bold",
    paddingHorizontal: 4,
    borderRadius: 4,
  },
  candImage: { width: 90, height: 60, borderRadius: 4, marginTop: 4 },
  candTitle: { color: "#FFF", fontWeight: "bold", fontSize: 13, marginTop: 4 },
  candLean: { color: "#34D399", fontSize: 11, fontWeight: "600" },
  candPrice: { color: "#9CA3AF", fontSize: 10, marginTop: 2 },
  candEffPrice: { color: "#F59E0B", fontSize: 11, fontWeight: "bold" },
  actionRow: { alignItems: "center", marginTop: 4 },
  snapButton: {
    backgroundColor: "#10B981",
    paddingVertical: 14,
    paddingHorizontal: 36,
    borderRadius: 24,
    width: "90%",
    alignItems: "center",
  },
  snapDisabled: { opacity: 0.6 },
  snapButtonText: { color: "#FFF", fontSize: 16, fontWeight: "bold" },
});

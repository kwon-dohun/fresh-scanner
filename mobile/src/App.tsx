import React, { useState, useEffect, useRef } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  SafeAreaView,
  ActivityIndicator,
  ScrollView,
} from "react-native";
import { Camera, CameraType } from "expo-camera";
import axios from "axios";

// 백엔드 API 호스트 주소
const API_BASE_URL = "http://localhost:8000";

interface DetectedBox {
  class_name: string;
  confidence: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
  quality_score: number;
}

interface XAIExplanation {
  factor: string;
  score: number;
  description: string;
}

interface PreciseReport {
  class_name: string;
  freshness_grade: string;
  overall_score: number;
  explanations: XAIExplanation[];
  ocr_text: string;
}

export default function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [type, setType] = useState(CameraType.back);
  const [isScanning, setIsScanning] = useState(false);
  const [detectedItems, setDetectedItems] = useState<DetectedBox[]>([]);
  const [report, setReport] = useState<PreciseReport | null>(null);
  const cameraRef = useRef<Camera | null>(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === "granted");
    })();
  }, []);

  // 1단계: 실시간 AR 프레임 캡처 (모의 인터벌)
  const handleRealtimeScan = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.5 });
      const formData = new FormData();
      formData.append("image", {
        uri: photo.uri,
        name: "frame.jpg",
        type: "image/jpeg",
      } as any);

      const res = await axios.post(`${API_BASE_URL}/scan/realtime`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setDetectedItems(res.data.items || []);
    } catch (err) {
      console.log("AR Scan error", err);
    }
  };

  // 2단계: 고화질 정밀 촬영 및 XAI 리포트 호출
  const handlePreciseScan = async () => {
    if (!cameraRef.current) return;
    setIsScanning(true);
    setReport(null);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.9 });
      const formData = new FormData();
      formData.append("image", {
        uri: photo.uri,
        name: "highres.jpg",
        type: "image/jpeg",
      } as any);

      const res = await axios.post(`${API_BASE_URL}/scan/precise`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setReport(res.data);
    } catch (err) {
      alert("정밀 분석 중 오류가 발생했습니다.");
    } finally {
      setIsScanning(false);
    }
  };

  if (hasPermission === null) {
    return <View style={styles.center}><Text>카메라 권한 요청 중...</Text></View>;
  }
  if (hasPermission === false) {
    return <View style={styles.center}><Text>카메라 접근 권한이 없습니다.</Text></View>;
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* 카메라 뷰 및 1단계 실시간 AR 오버레이 */}
      <View style={styles.cameraContainer}>
        <Camera style={styles.camera} type={type} ref={cameraRef}>
          {/* 실시간 탐지 바운딩 박스 AR 오버레이 */}
          {detectedItems.map((item, idx) => (
            <View
              key={idx}
              style={[
                styles.arBox,
                {
                  left: item.bbox.x1 * 0.5,
                  top: item.bbox.y1 * 0.5,
                  width: (item.bbox.x2 - item.bbox.x1) * 0.5,
                  height: (item.bbox.y2 - item.bbox.y1) * 0.5,
                },
              ]}
            >
              <Text style={styles.arBadge}>
                {item.class_name} ({item.quality_score}점)
              </Text>
            </View>
          ))}

          {/* AR 가이드 안내 배너 */}
          <View style={styles.guideBanner}>
            <Text style={styles.guideText}>
              📸 최종 후보를 사진으로 촬영하면 미세 품질 및 가성비 정밀 분석이 실행됩니다.
            </Text>
          </View>
        </Camera>
      </View>

      {/* 하단 컨트롤 및 3단 XAI 리포트 표시 */}
      <View style={styles.bottomSheet}>
        {isScanning ? (
          <ActivityIndicator size="large" color="#10B981" />
        ) : report ? (
          <ScrollView style={styles.reportScroll}>
            <View style={styles.reportHeader}>
              <Text style={styles.reportTitle}>🛒 AI 정밀 분석 리포트</Text>
              <Text style={styles.scoreText}>종합 점수: {report.overall_score}점</Text>
              <Text style={styles.gradeBadge}>등급: {report.freshness_grade.toUpperCase()}</Text>
            </View>

            <Text style={styles.sectionHeader}>💡 판단 근거 (Why):</Text>
            {report.explanations.map((exp, i) => (
              <View key={i} style={styles.expCard}>
                <Text style={styles.expFactor}>{exp.factor} (평가점수: {exp.score})</Text>
                <Text style={styles.expDesc}>{exp.description}</Text>
              </View>
            ))}

            <TouchableOpacity style={styles.resetButton} onPress={() => setReport(null)}>
              <Text style={styles.resetButtonText}>다시 스캔하기</Text>
            </TouchableOpacity>
          </ScrollView>
        ) : (
          <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.secondaryButton} onPress={handleRealtimeScan}>
              <Text style={styles.buttonText}>1단계 AR 탐색</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.primaryButton} onPress={handlePreciseScan}>
              <Text style={styles.primaryButtonText}>2단계 정밀 촬영</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  cameraContainer: { flex: 3 },
  camera: { flex: 1, position: "relative" },
  arBox: {
    position: "absolute",
    borderColor: "#10B981",
    borderWidth: 2,
    borderRadius: 6,
    backgroundColor: "rgba(16, 185, 129, 0.15)",
  },
  arBadge: {
    backgroundColor: "#10B981",
    color: "#fff",
    fontSize: 12,
    fontWeight: "bold",
    paddingHorizontal: 6,
    paddingVertical: 2,
    alignSelf: "flex-start",
  },
  guideBanner: {
    position: "absolute",
    bottom: 12,
    left: 16,
    right: 16,
    backgroundColor: "rgba(0, 0, 0, 0.75)",
    padding: 10,
    borderRadius: 8,
  },
  guideText: { color: "#FFF", fontSize: 13, textAlign: "center" },
  bottomSheet: {
    flex: 2,
    backgroundColor: "#1F2937",
    padding: 16,
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
  },
  buttonRow: { flexDirection: "row", justifyContent: "space-around", marginTop: 20 },
  secondaryButton: {
    backgroundColor: "#374151",
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 10,
  },
  primaryButton: {
    backgroundColor: "#10B981",
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 10,
  },
  buttonText: { color: "#FFF", fontWeight: "600", fontSize: 16 },
  primaryButtonText: { color: "#FFF", fontWeight: "bold", fontSize: 16 },
  reportScroll: { flex: 1 },
  reportHeader: { marginBottom: 12 },
  reportTitle: { color: "#FFF", fontSize: 18, fontWeight: "bold" },
  scoreText: { color: "#10B981", fontSize: 24, fontWeight: "bold", marginTop: 4 },
  gradeBadge: { color: "#F59E0B", fontSize: 14, fontWeight: "600", marginTop: 2 },
  sectionHeader: { color: "#E5E7EB", fontSize: 15, fontWeight: "600", marginVertical: 8 },
  expCard: {
    backgroundColor: "#374151",
    padding: 10,
    borderRadius: 8,
    marginBottom: 8,
  },
  expFactor: { color: "#60A5FA", fontWeight: "bold", fontSize: 13 },
  expDesc: { color: "#D1D5DB", fontSize: 13, marginTop: 2 },
  resetButton: {
    backgroundColor: "#4B5563",
    padding: 12,
    borderRadius: 8,
    alignItems: "center",
    marginTop: 12,
  },
  resetButtonText: { color: "#FFF", fontWeight: "bold" },
});

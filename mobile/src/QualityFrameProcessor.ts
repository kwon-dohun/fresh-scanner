/**
 * FreshScanner 모바일 온디바이스 실시간 프레임 프로세서 (Zero-Latency Frame Processor).
 *
 * 네트워크 통신(HTTP) 없이 모바일 기기 내부(Worklet / JSI)에서 직접 동작하여,
 * 지하 마트 환경에서도 0ms 지연으로 카메라 프리뷰의 초점 흐림(Blur) 및 비닐 랩 반사광(Glare)을 실시간 차단합니다.
 */

export interface FrameQualityFeedback {
  canCapture: boolean;
  blurScore: number;
  glareRatio: number;
  guideMessage: string;
}

/**
 * 카메라 프리뷰 Y-Plane(Luma) 버퍼를 샘플링하여 온디바이스에서 즉각 품질 판정.
 * (react-native-vision-camera 프레임 프로세서 호환)
 */
export function evaluateFrameOnDevice(
  lumaData: Uint8Array | number[],
  width: number,
  height: number
): FrameQualityFeedback {
  if (!lumaData || lumaData.length === 0) {
    return {
      canCapture: false,
      blurScore: 0,
      glareRatio: 1,
      guideMessage: "카메라 프레임 준비 중...",
    };
  }

  // 1. 성능 최적화를 위한 4x4 서브샘플링 스트라이드
  const stride = 4;
  const sampleW = Math.floor(width / stride);
  const sampleH = Math.floor(height / stride);

  let saturatedCount = 0;
  let sampledTotal = 0;

  // 2. 라플라시안 2차 미분 분산 계산용 변수 (라플라시안 커널: [0, 1, 0; 1, -4, 1; 0, 1, 0])
  let laplacianSum = 0;
  let laplacianSqSum = 0;
  let laplacianCount = 0;

  for (let y = 1; y < sampleH - 1; y++) {
    const rowOffset = y * stride * width;
    const prevRowOffset = (y - 1) * stride * width;
    const nextRowOffset = (y + 1) * stride * width;

    for (let x = 1; x < sampleW - 1; x++) {
      const col = x * stride;
      const center = lumaData[rowOffset + col];

      sampledTotal++;

      // [A] 고휘도 반사광(Glare) 검출: Y채널 명도 > 238
      if (center > 238) {
        saturatedCount++;
      }

      // [B] 이웃 픽셀 라플라시안 합성곱 연산
      const up = lumaData[prevRowOffset + col];
      const down = lumaData[nextRowOffset + col];
      const left = lumaData[rowOffset + (col - stride)];
      const right = lumaData[rowOffset + (col + stride)];

      const lap = up + down + left + right - 4 * center;
      laplacianSum += lap;
      laplacianSqSum += lap * lap;
      laplacianCount++;
    }
  }

  // 3. 지표 산출
  const glareRatio = sampledTotal > 0 ? saturatedCount / sampledTotal : 0;

  const meanLap = laplacianCount > 0 ? laplacianSum / laplacianCount : 0;
  const variance = laplacianCount > 0 ? laplacianSqSum / laplacianCount - meanLap * meanLap : 0;
  const blurScore = Math.round(variance);

  // 4. 품질 임계치 검증
  // - blurScore < 85: 손떨림 또는 초점 미동기화
  // - glareRatio > 0.14: 팩 표면 랩 비닐 빛 반사 과다
  let canCapture = true;
  let guideMessage = "완벽한 각도와 선명도입니다! 📸 셔터를 누르세요.";

  if (blurScore < 85) {
    canCapture = false;
    guideMessage = "⚠️ 초점이 흐립니다. 스마트폰을 잠시 고정해 주세요.";
  } else if (glareRatio > 0.14) {
    canCapture = false;
    guideMessage = "💡 비닐 빛 반사가 심합니다. 폰 각도를 15도 정도 살짝 기울여 주세요.";
  }

  return {
    canCapture,
    blurScore,
    glareRatio: Math.round(glareRatio * 1000) / 1000,
    guideMessage,
  };
}

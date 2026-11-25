# 프레시스캐너 (가칭)

마트·시장 매대에 카메라를 비추면 **보이는 과일마다 순위 배지**를 띄우고, 배지를 누르면 **왜 그 순위인지 근거와 증거 마스크**를 보여주는 안드로이드 앱.

> **상태: v0, 감귤 1품목, 실측 검증 전.** 점수 계산은 실제 화면 픽셀로 하지만, 임계값은 통설 기반 가설이라 정확도를 주장하지 않는다. 검증 계획은 [docs/validation/실측설계.md](docs/validation/실측설계.md).

## 지금 되는 것

- CameraX 실시간 분석(프레임당 약 7~30ms, 에뮬레이터 측정)과 풀스크린 미리보기
- 색 기반 감귤 검출. 마스크의 거리 변환 봉우리를 개체 중심으로 삼아 맞닿은 과일을 나눈다
- 개체별 룰 점수: 색(균일도, 연두빛), 껍질 결(매끈함), 편평비. 어두운 반점이 3%를 넘으면 결함 Veto("주의")
- 같은 화면에 보이는 과일끼리만 순위를 매기고, 점수는 이동평균으로 안정화
- 배지 탭 → 바텀시트: 순위, 점수, 한 줄 이유, 요인 막대, 증거 마스크(하양은 판단에 쓴 곳, 빨강은 결함 의심)
- "이 장면 기록": 사진과 개체별 원시 특징을 저장해 실측 데이터로 쓴다

## 아직 아닌 것

- **꼭지** 항목은 계산하지 않는다 (룰 4항목 중 3개만 구현)
- 정육, 수박, 사과, 딸기 등 다른 품목
- 학습 모델. 지금 검출기는 색 기반이라 갈색 나무 매대나 주황 포장재에서 오검출할 수 있다
- 서버. 모든 분석이 기기 안에서 끝나고, 인터넷 권한도 없다 (시세 연동 등은 나중)
- 실제 시장 정확도. 합성 장면 벤치마크 F1 0.98은 쉬운 장면 기준이라 믿으면 안 된다

## 동작 흐름

```
CameraX 프레임 ─ Frame.kt        회전 보정 + 256px 폭으로 다운샘플
              ─ FruitDetector    HSV 마스크 → 거리 변환 시드 → 조각 병합 → 개체별 특징
              ─ Scorer           색·결·편평비 점수 + 결함 Veto
              ─ Tracker          개체 id 유지, 점수 이동평균, 화면 안 개체끼리 순위
              ─ OverlayView      링과 배지, 탭 처리 (FILL_CENTER 좌표 매핑)
              ─ 바텀시트 + Evidence  탭한 개체의 확대 사진 위에 판단 영역 마스크
```

## 빌드와 실행

필요한 것: JDK 17, Android SDK(compileSdk 37 플랫폼), `local.properties`에 `sdk.dir=C:/Users/.../Sdk` (슬래시).

```powershell
./gradlew testDebugUnitTest      # 유닛 테스트 16개와 합성 장면 벤치마크
./gradlew assembleDebug          # app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

**프로젝트 경로에 한글이 있으면 AGP가 빌드를 막는다.** 영문 경로에서 빌드한다.

벤치마크 수치를 보려면 `./gradlew testDebugUnitTest -i`로 돌려 `==` 로 시작하는 출력을 본다.

### 데모 모드 (디버그 빌드 전용)

에뮬레이터 카메라는 움직일 수 없어서, 카메라 대신 합성 감귤 사진을 같은 분석 파이프라인에 넣는 모드가 있다.

```powershell
adb shell am start -n com.freshscanner.app/.MainActivity --ez demo true
```

### 실측 기록 가져오기

시트의 "이 장면 기록"을 누르면 기기에 `.jpg`와 `.json`이 저장된다. `brix` 칸은 비어 있으니 당도계로 잰 값을 직접 채운다.

```powershell
adb pull /sdcard/Android/data/com.freshscanner.app/files/captures ./captures
```

## 폴더 구조

```
app/src/main/java/com/freshscanner/app/
  MainActivity.kt     화면 흐름, 카메라, 바텀시트, 기록 저장
  OverlayView.kt      링·배지 그리기와 탭 처리
  EvidenceView.kt     증거 사진 + 마스크 그리기
  Evidence.kt         탭한 개체의 사진과 마스크 생성
  Demo.kt             디버그 전용 데모 장면
  core/
    Frame.kt          회전 보정 다운샘플
    FruitDetector.kt  색 기반 검출 (거리 변환 시드, 병합)
    Scorer.kt         룰 점수와 Veto
    Tracker.kt        id 유지, 점수 이동평균, 순위
app/src/test/         유닛 테스트와 합성 장면 벤치마크
docs/validation/      실측 검증 설계와 기록 시트
```

## 다음 단계

1. **실측 검증.** 감귤 50개를 룰로 순위 매기고 당도계(Brix)와 대조한다. 통과 기준은 설계 문서에 미리 정해 뒀다.
2. 통과한 요인만 `(점수, 신뢰도, 증거 마스크)` 형태의 독립 요인으로 정리하고, 신뢰도 기반 가중치를 붙인다.
3. 다음 품목(사과, 삼겹살)과, 라벨 데이터가 쌓이면 학습 검출기.

## 옛 프로토타입

이 저장소의 이전 내용(Expo + FastAPI, 실시간 점수가 모의값이던 버전)은 태그 `legacy-expo-mock`에 보존돼 있다. KAMIS 시세 연동, 라벨 OCR, 점수 엔진 구조를 다시 쓸 때는 거기서 꺼낸다.

```powershell
git checkout legacy-expo-mock
```

# Night Driving Assist (야간 주행 보조 시스템)

야간 주행 시 보행자 및 차량을 실시간으로 감지하고  
위험도를 판단하여 경고를 제공하는 AI 기반 보조 시스템입니다.

---

## 📌 프로젝트 개요

- YOLOv8 기반 객체 탐지
- 야간 환경에 맞춘 밝기 보정
- 거리 기반 위험 판단 로직 적용
- 실시간 영상 처리 (카메라 / 영상 파일)

---

## 🎯 주요 기능

### 1️⃣ 객체 탐지
- 사람 (person)
- 차량 (car, truck, bus)
- 자전거 / 오토바이

---

### 2️⃣ 위험 판단

| 상태 | 조건 |
|------|------|
| 🟢 SAFE | 멀리 있음 |
| 🟠 WARNING | 가까워짐 |
| 🔴 DANGER | 매우 근접 |

👉 사람은 차량보다 **우선 위험 처리**

---

### 3️⃣ 야간 밝기 보정
- ROI(하단 영역)만 선택적으로 밝기 개선
- 감마 보정 적용
- 과도한 밝기 증가 방지

---

### 4️⃣ 프리셋 기능 (카메라 위치 대응)

| 키 | 설명 |
|----|------|
| 1 | dashboard |
| 2 | rearview |
| 3 | low_mount |
| 4 | full_view |

---

### 5️⃣ 경고 시스템
- 위험 시 경고음 발생
- 로그 파일 기록

---

## 🖥 실행 방법

### 1. 설치

```bash
pip install ultralytics opencv-python numpy

### 2. 실행
python main.py

### 3. 입력변경

# 영상 테스트
cap = cv2.VideoCapture("C:/night_drive/videos/영상파일.mp4")

# 카메라 사용
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

📂 프로젝트 구조
ai_video/
 ├ main.py              # 메인 실행 코드
 ├ camera_check.py      # 카메라 테스트 코드
 ├ README.md
 ├ .gitignore


⚙️ 사용 기술
Python
OpenCV
YOLOv8 (Ultralytics)
NumPy

import cv2  # 영상 처리 라이브러리
from ultralytics import YOLO  # YOLO 객체 탐지 모델
import datetime  # 현재 시간 표시용
import time  # FPS, 로그 시간 계산용
import numpy as np  # 감마 보정용 배열 처리
import winsound  # Windows 경고음 재생용 라이브러리


# ===== 카메라 설치 위치 프리셋 기본값 =====
CAMERA_PRESET = "dashboard"  # 시작 프리셋을 설정한다


# ===== 카메라 프리셋별 설정값 =====
PRESETS = {
    "dashboard": {  # 대시보드 위쪽 설치 기준
        "ROI_TOP_RATIO": 0.50,  # 위험 판단 영역 시작 위치
        "DANGER_Y_RATIO": 0.72,  # 차량 위험 기준 y 비율
        "WARNING_Y_RATIO": 0.60,  # 차량 주의 기준 y 비율
        "DANGER_BOX_AREA": 35000,  # 차량 위험 기준 박스 면적
        "WARNING_BOX_AREA": 18000,  # 차량 주의 기준 박스 면적
        "PERSON_DANGER_Y_RATIO": 0.65,  # 사람 위험 기준 y 비율
        "PERSON_WARNING_Y_RATIO": 0.55,  # 사람 주의 기준 y 비율
    },
    "rearview": {  # 룸미러 근처 설치 기준
        "ROI_TOP_RATIO": 0.45,  # 위험 판단 영역 시작 위치를 더 위로 잡는다
        "DANGER_Y_RATIO": 0.70,  # 차량 위험 기준 y 비율
        "WARNING_Y_RATIO": 0.58,  # 차량 주의 기준 y 비율
        "DANGER_BOX_AREA": 32000,  # 차량 위험 기준 박스 면적
        "WARNING_BOX_AREA": 16000,  # 차량 주의 기준 박스 면적
        "PERSON_DANGER_Y_RATIO": 0.62,  # 사람 위험 기준 y 비율
        "PERSON_WARNING_Y_RATIO": 0.52,  # 사람 주의 기준 y 비율
    },
    "low_mount": {  # 유리 하단 또는 낮은 위치 설치 기준
        "ROI_TOP_RATIO": 0.55,  # 위험 판단 영역 시작 위치를 더 아래로 잡는다
        "DANGER_Y_RATIO": 0.76,  # 차량 위험 기준 y 비율
        "WARNING_Y_RATIO": 0.64,  # 차량 주의 기준 y 비율
        "DANGER_BOX_AREA": 38000,  # 차량 위험 기준 박스 면적
        "WARNING_BOX_AREA": 20000,  # 차량 주의 기준 박스 면적
        "PERSON_DANGER_Y_RATIO": 0.68,  # 사람 위험 기준 y 비율
        "PERSON_WARNING_Y_RATIO": 0.58,  # 사람 주의 기준 y 비율
    },
    "full_view": {  # 전체 화면에 가깝게 넓게 보는 프리셋
        "ROI_TOP_RATIO": 0.00,  # 위험 판단 영역을 화면 맨 위부터 시작한다
        "DANGER_Y_RATIO": 0.72,  # 차량 위험 기준 y 비율
        "WARNING_Y_RATIO": 0.60,  # 차량 주의 기준 y 비율
        "DANGER_BOX_AREA": 35000,  # 차량 위험 기준 박스 면적
        "WARNING_BOX_AREA": 18000,  # 차량 주의 기준 박스 면적
        "PERSON_DANGER_Y_RATIO": 0.65,  # 사람 위험 기준 y 비율
        "PERSON_WARNING_Y_RATIO": 0.55,  # 사람 주의 기준 y 비율
    }
}


# ===== 프리셋 적용 함수 =====
def apply_preset(preset_name):  # 프리셋 이름을 받아서 설정값을 반환하는 함수
    preset = PRESETS[preset_name]  # 선택한 프리셋 딕셔너리를 가져온다
    return (
        preset["ROI_TOP_RATIO"],  # 위험 판단 영역 시작 위치를 반환한다
        preset["DANGER_Y_RATIO"],  # 차량 위험 기준 y 비율을 반환한다
        preset["WARNING_Y_RATIO"],  # 차량 주의 기준 y 비율을 반환한다
        preset["DANGER_BOX_AREA"],  # 차량 위험 기준 박스 면적을 반환한다
        preset["WARNING_BOX_AREA"],  # 차량 주의 기준 박스 면적을 반환한다
        preset["PERSON_DANGER_Y_RATIO"],  # 사람 위험 기준 y 비율을 반환한다
        preset["PERSON_WARNING_Y_RATIO"],  # 사람 주의 기준 y 비율을 반환한다
    )


# ===== 감마 보정 함수 =====
def adjust_gamma(image, gamma=1.5):  # 감마 보정 함수 정의
    invGamma = 1.0 / gamma  # 감마 역수를 계산한다
    table = [((i / 255.0) ** invGamma) * 255 for i in range(256)]  # 0~255 밝기 보정 테이블 생성
    table = np.array(table, dtype="uint8")  # numpy 배열로 변환
    return cv2.LUT(image, table)  # LUT 방식으로 감마 보정을 적용한다


# ===== 경고음 함수 =====
def play_warning_beep():  # 위험 상태일 때 경고음을 울리는 함수
    winsound.Beep(2000, 300)  # 2000Hz 주파수로 0.3초 동안 소리를 재생한다


# ===== YOLO 모델 로드 =====
model = YOLO("yolov8n.pt")  # YOLO 모델을 불러온다


# ===== 현재 프리셋 적용 =====
current_preset_name = CAMERA_PRESET  # 현재 프리셋 이름을 저장한다

(
    ROI_TOP_RATIO,  # 위험 판단 영역 시작 위치
    DANGER_Y_RATIO,  # 차량 위험 기준 y 비율
    WARNING_Y_RATIO,  # 차량 주의 기준 y 비율
    DANGER_BOX_AREA,  # 차량 위험 기준 박스 면적
    WARNING_BOX_AREA,  # 차량 주의 기준 박스 면적
    PERSON_DANGER_Y_RATIO,  # 사람 위험 기준 y 비율
    PERSON_WARNING_Y_RATIO,  # 사람 주의 기준 y 비율
) = apply_preset(current_preset_name)  # 현재 프리셋을 적용한다


# ===== 경고음 관련 설정 =====
last_beep_time = 0  # 마지막 경고음 재생 시간을 저장한다
BEEP_INTERVAL = 0.8  # 경고음 최소 간격을 설정한다


# ===== 영상 테스트 모드 =====
cap = cv2.VideoCapture("C:/night_drive/videos/006.mp4")  # 테스트용 영상 파일을 연다

# ===== 카메라 모드 =====
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 기본 카메라를 연다
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 카메라 가로 해상도를 설정한다
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 카메라 세로 해상도를 설정한다
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 카메라 버퍼를 최소화한다

if not cap.isOpened():  # 영상 또는 카메라가 정상적으로 열렸는지 확인한다
    print("영상 열기 실패")  # 실패 메시지를 출력한다
    exit()  # 프로그램을 종료한다


log_file = open("danger_log.txt", "a", encoding="utf-8")  # 위험 로그 파일을 연다
last_log_time = 0  # 마지막 로그 저장 시간을 저장한다
fail_count = 0  # 프레임 읽기 실패 횟수를 저장한다


while True:  # 프레임 반복 처리 시작
    loop_start_time = time.time()  # FPS 계산용 시작 시간을 저장한다

    ret, frame = cap.read()  # 프레임 한 장을 읽는다

    if not ret:  # 프레임을 읽지 못했는지 확인한다
        fail_count += 1  # 실패 횟수를 증가시킨다
        print(f"프레임 읽기 실패: {fail_count}회")  # 실패 횟수를 출력한다
        time.sleep(0.2)  # 잠깐 대기한다

        if fail_count >= 10:  # 실패가 10번 이상 누적되면
            print("영상 또는 카메라 연결이 불안정하여 종료합니다.")  # 종료 메시지를 출력한다
            break  # 반복문을 종료한다

        continue  # 다음 반복으로 넘어간다

    fail_count = 0  # 정상 프레임이면 실패 횟수를 초기화한다

    height, width, _ = frame.shape  # 현재 프레임의 높이와 너비를 가져온다

    # ===== 위험 판단 영역 설정 =====
    start_x = 0  # 위험 판단 영역의 왼쪽 시작 좌표를 설정한다
    end_x = width  # 위험 판단 영역의 오른쪽 끝 좌표를 설정한다
    start_y = int(height * ROI_TOP_RATIO)  # 위험 판단 시작 y좌표를 설정한다
    end_y = height  # 위험 판단 끝 y좌표를 설정한다

    # ===== 밝기 보정 영역을 하늘색 박스와 동일하게 맞춤 =====
    brightness_roi_start_y = start_y  # 밝기 보정 시작 위치를 하늘색 위험 판단 영역 시작 위치와 같게 맞춘다
    brightness_roi = frame[brightness_roi_start_y:height, 0:width]  # 하늘색 박스와 같은 영역부터 아래까지 밝기 보정을 적용한다

    mean_brightness = brightness_roi.mean()  # 밝기 보정 영역 평균 밝기를 계산한다

    if mean_brightness < 30:  # 매우 어두운 경우
        bright_roi = cv2.convertScaleAbs(brightness_roi, alpha=1.8, beta=30)  # 밝기와 대비를 강하게 높인다
        bright_roi = adjust_gamma(bright_roi, gamma=1.5)  # 감마 보정을 추가한다
        frame[brightness_roi_start_y:height, 0:width] = bright_roi  # 보정한 ROI를 원본 프레임에 다시 넣는다

    elif mean_brightness < 60:  # 일반적인 야간 정도로 어두운 경우
        bright_roi = cv2.convertScaleAbs(brightness_roi, alpha=1.5, beta=20)  # 밝기와 대비를 적당히 높인다
        bright_roi = adjust_gamma(bright_roi, gamma=1.2)  # 약한 감마 보정을 적용한다
        frame[brightness_roi_start_y:height, 0:width] = bright_roi  # 보정한 ROI를 원본 프레임에 다시 넣는다

    # ===== 낮/밤 상태 판단 =====
    if mean_brightness < 60:  # 평균 밝기가 60보다 작으면
        day_state = "NIGHT"  # 야간으로 표시한다
    else:  # 평균 밝기가 60 이상이면
        day_state = "DAY"  # 주간으로 표시한다

    danger_detected = False  # 전체 위험 상태를 저장하는 변수
    warning_detected = False  # 전체 주의 상태를 저장하는 변수
    person_priority_detected = False  # 사람 우선 위험 상태를 저장하는 변수

    results = model(frame, verbose=False)  # 현재 프레임에 대해 YOLO 탐지를 실행한다

    for result in results:  # 탐지 결과를 하나씩 확인한다
        for box in result.boxes:  # 각 바운딩 박스를 하나씩 확인한다
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표를 정수로 가져온다
            cls = int(box.cls[0])  # 클래스 번호를 가져온다
            label = model.names[cls]  # 클래스 이름을 가져온다
            conf = float(box.conf[0])  # 탐지 신뢰도를 가져온다

            if label in ["person", "car", "truck", "bus", "bicycle", "motorcycle"] and conf > 0.5:  # 사람/차량 계열만 사용한다
                cx = int((x1 + x2) / 2)  # 객체 중심 x좌표를 계산한다
                cy = int((y1 + y2) / 2)  # 객체 중심 y좌표를 계산한다

                box_width = x2 - x1  # 박스 가로 길이를 계산한다
                box_height = y2 - y1  # 박스 세로 길이를 계산한다
                box_area = box_width * box_height  # 박스 면적을 계산한다

                in_center = start_x < cx < end_x and start_y < cy < end_y  # 객체 중심이 위험 판단 영역 안에 있는지 확인한다
                center_y_ratio = cy / height  # 객체 중심이 화면 아래쪽에 얼마나 가까운지 비율로 계산한다

                if in_center:  # 객체가 위험 판단 영역 안에 있을 때만 상태를 판단한다

                    # ===== 사람 기준 =====
                    if label == "person":  # 사람이 감지되었을 때
                        if center_y_ratio > PERSON_DANGER_Y_RATIO:  # 사람이 화면 아래쪽에 가까이 있으면
                            color = (0, 0, 255)  # 빨간색을 사용한다
                            text = "DANGER"  # 위험 상태로 표시한다
                            danger_detected = True  # 전체 위험 상태를 켠다
                            person_priority_detected = True  # 사람 우선 위험 상태를 켠다

                            current_time = time.time()  # 현재 시간을 저장한다

                            if current_time - last_log_time >= 1:  # 1초마다 한 번만 로그를 저장한다
                                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간을 문자열로 만든다
                                log_file.write(f"{now} - DANGER detected: {label}\n")  # 로그 파일에 기록한다
                                log_file.flush()  # 즉시 파일에 반영한다
                                last_log_time = current_time  # 마지막 로그 시간을 갱신한다

                        elif center_y_ratio > PERSON_WARNING_Y_RATIO:  # 사람이 조금 위쪽에 있으면
                            color = (0, 165, 255)  # 주황색을 사용한다
                            text = "WARNING"  # 주의 상태로 표시한다
                            warning_detected = True  # 전체 주의 상태를 켠다

                        else:  # 사람이 멀리 있으면
                            color = (0, 255, 0)  # 초록색을 사용한다
                            text = "SAFE"  # 안전 상태로 표시한다

                    # ===== 차량 기준 =====
                    elif label in ["car", "truck", "bus", "bicycle", "motorcycle"]:  # 차량 계열 객체일 때
                        if center_y_ratio > DANGER_Y_RATIO and box_area > DANGER_BOX_AREA:  # 매우 가깝고 박스가 크면
                            color = (0, 0, 255)  # 빨간색을 사용한다
                            text = "DANGER"  # 위험 상태로 표시한다
                            danger_detected = True  # 전체 위험 상태를 켠다

                            current_time = time.time()  # 현재 시간을 저장한다

                            if current_time - last_log_time >= 1:  # 1초마다 한 번만 로그를 저장한다
                                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간을 문자열로 만든다
                                log_file.write(f"{now} - DANGER detected: {label}\n")  # 로그 파일에 기록한다
                                log_file.flush()  # 즉시 파일에 반영한다
                                last_log_time = current_time  # 마지막 로그 시간을 갱신한다

                        elif center_y_ratio > WARNING_Y_RATIO and box_area > WARNING_BOX_AREA:  # 조금 가까운 경우
                            color = (0, 165, 255)  # 주황색을 사용한다
                            text = "WARNING"  # 주의 상태로 표시한다
                            warning_detected = True  # 전체 주의 상태를 켠다

                        else:  # 차량이 멀리 있으면
                            color = (0, 255, 0)  # 초록색을 사용한다
                            text = "SAFE"  # 안전 상태로 표시한다

                    else:  # 예외 상황 처리
                        color = (0, 255, 0)  # 초록색을 사용한다
                        text = "SAFE"  # 안전 상태로 표시한다

                else:  # 위험 판단 영역 밖이면
                    color = (0, 255, 0)  # 초록색을 사용한다
                    text = "SAFE"  # 안전 상태로 표시한다

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 객체 바운딩 박스를 그린다
                cv2.putText(frame, f"{label} {text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  # 객체 이름과 상태를 표시한다

                if label == "person":  # 사람이면 추가 안내 문구를 표시한다
                    cv2.putText(frame, "PEDESTRIAN", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 사람 경고 문구를 표시한다

    # ===== 사람 우선 전체 상태 정리 =====
    if person_priority_detected:  # 사람이 위험 상태로 감지되었으면
        danger_detected = True  # 전체 상태를 무조건 위험으로 유지한다
        warning_detected = False  # 사람 위험이 있으면 주의 상태를 끈다

    # ===== 경고음 처리 =====
    current_beep_time = time.time()  # 현재 시간을 저장한다
    if danger_detected and current_beep_time - last_beep_time >= BEEP_INTERVAL:  # 위험 상태이고 경고음 간격이 지났으면
        play_warning_beep()  # 경고음을 재생한다
        last_beep_time = current_beep_time  # 마지막 경고음 시간을 갱신한다

    # ===== 위험 판단 영역 표시 =====
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 0), 2)  # 하늘색 박스로 위험 판단 영역을 표시한다

    # ===== 화면 UI 표시 =====
    cv2.putText(frame, "Night Driving Assist", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 프로그램 제목을 표시한다

    current_time_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간을 문자열로 만든다
    cv2.putText(frame, current_time_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 현재 시간을 표시한다
    cv2.putText(frame, "Log Saving: ON", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # 로그 저장 상태를 표시한다
    cv2.putText(frame, f"{day_state} ({int(mean_brightness)})", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 낮/밤 상태를 표시한다
    cv2.putText(frame, f"Preset: {current_preset_name}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)  # 현재 프리셋을 표시한다
    cv2.putText(frame, "1:dashboard  2:rearview  3:low_mount  4:full_view", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 255), 2)  # 키보드 프리셋 변경 안내를 표시한다

    fps = 1 / (time.time() - loop_start_time)  # FPS를 계산한다
    cv2.putText(frame, f"FPS: {int(fps)}", (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # FPS를 표시한다

    if person_priority_detected:  # 사람이 위험 상태면 가장 우선해서 표시한다
        cv2.putText(frame, "PEDESTRIAN DANGER !!!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)  # 사람 우선 위험 문구를 표시한다
    elif danger_detected:  # 일반 위험 상태이면
        cv2.putText(frame, "DANGER !!!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)  # 위험 문구를 표시한다
    elif warning_detected:  # 주의 상태이면
        cv2.putText(frame, "WARNING !!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)  # 주의 문구를 표시한다
    else:  # 아무 위험도 없으면
        cv2.putText(frame, "SAFE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)  # 안전 문구를 표시한다

    cv2.imshow("Night Assistant Camera", frame)  # 결과 화면을 출력한다

    # ===== 키보드 입력 처리 =====
    key = cv2.waitKey(1) & 0xFF  # 키보드 입력을 받는다

    if key == ord("1"):  # 1번 키를 누르면
        current_preset_name = "dashboard"  # dashboard 프리셋으로 변경한다
        (
            ROI_TOP_RATIO,
            DANGER_Y_RATIO,
            WARNING_Y_RATIO,
            DANGER_BOX_AREA,
            WARNING_BOX_AREA,
            PERSON_DANGER_Y_RATIO,
            PERSON_WARNING_Y_RATIO,
        ) = apply_preset(current_preset_name)  # 변경된 프리셋 값을 즉시 적용한다

    elif key == ord("2"):  # 2번 키를 누르면
        current_preset_name = "rearview"  # rearview 프리셋으로 변경한다
        (
            ROI_TOP_RATIO,
            DANGER_Y_RATIO,
            WARNING_Y_RATIO,
            DANGER_BOX_AREA,
            WARNING_BOX_AREA,
            PERSON_DANGER_Y_RATIO,
            PERSON_WARNING_Y_RATIO,
        ) = apply_preset(current_preset_name)  # 변경된 프리셋 값을 즉시 적용한다

    elif key == ord("3"):  # 3번 키를 누르면
        current_preset_name = "low_mount"  # low_mount 프리셋으로 변경한다
        (
            ROI_TOP_RATIO,
            DANGER_Y_RATIO,
            WARNING_Y_RATIO,
            DANGER_BOX_AREA,
            WARNING_BOX_AREA,
            PERSON_DANGER_Y_RATIO,
            PERSON_WARNING_Y_RATIO,
        ) = apply_preset(current_preset_name)  # 변경된 프리셋 값을 즉시 적용한다

    elif key == ord("4"):  # 4번 키를 누르면
        current_preset_name = "full_view"  # full_view 프리셋으로 변경한다
        (
            ROI_TOP_RATIO,
            DANGER_Y_RATIO,
            WARNING_Y_RATIO,
            DANGER_BOX_AREA,
            WARNING_BOX_AREA,
            PERSON_DANGER_Y_RATIO,
            PERSON_WARNING_Y_RATIO,
        ) = apply_preset(current_preset_name)  # 변경된 프리셋 값을 즉시 적용한다

    elif key == ord("q"):  # q 키를 누르면
        break  # 프로그램을 종료한다


cap.release()  # 영상 또는 카메라를 해제한다
cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫는다
log_file.close()  # 로그 파일을 닫는다
import cv2  # OpenCV 라이브러리를 불러온다
from ultralytics import YOLO  # YOLO 모델을 사용하기 위해 불러온다
import datetime  # 현재 시간을 가져오기 위한 라이브러리
import time  # 시간 차이와 대기를 처리하기 위한 라이브러리

model = YOLO("yolov8n.pt")  # YOLOv8 기본 모델을 불러온다

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows DirectShow 방식으로 0번 카메라를 연다
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 카메라 가로 해상도를 640으로 설정한다
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 카메라 세로 해상도를 480으로 설정한다
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 카메라 버퍼를 줄여서 지연과 꼬임을 줄인다

if not cap.isOpened():  # 카메라가 정상적으로 열렸는지 확인한다
    print("카메라를 열 수 없습니다.")  # 오류 메시지를 출력한다
    exit()  # 프로그램을 종료한다

log_file = open("danger_log.txt", "a", encoding="utf-8")  # 로그를 저장할 파일을 UTF-8 인코딩으로 연다
last_log_time = 0  # 마지막 로그 저장 시간을 저장한다
fail_count = 0  # 프레임 읽기 실패 횟수를 저장한다

while True:  # 계속해서 프레임을 읽기 위해 반복한다
    ret, frame = cap.read()  # 카메라에서 프레임 한 장을 읽어온다

    if not ret:  # 프레임을 읽지 못했는지 확인한다
        fail_count += 1  # 프레임 실패 횟수를 1 증가시킨다
        print(f"프레임 읽기 실패: {fail_count}회")  # 현재 실패 횟수를 출력한다
        time.sleep(0.2)  # 너무 빠르게 반복되지 않도록 잠깐 대기한다

        if fail_count >= 10:  # 실패가 너무 많이 누적되면
            print("카메라 연결이 불안정하여 프로그램을 종료합니다.")  # 종료 메시지를 출력한다
            break  # 반복문을 종료한다

        continue  # 다음 반복으로 넘어간다

    fail_count = 0  # 프레임을 정상적으로 읽었으면 실패 횟수를 다시 0으로 초기화한다

    mean_brightness = frame.mean()  # 현재 화면 밝기 평균값을 계산한다

    if mean_brightness < 80:  # 화면이 너무 어두우면
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)  # 화면을 조금 밝게 만든다
    elif mean_brightness > 180:  # 화면이 너무 밝으면
        frame = cv2.convertScaleAbs(frame, alpha=0.9, beta=-10)  # 화면을 조금 어둡게 만든다

    height, width, _ = frame.shape  # 현재 프레임의 높이와 너비를 가져온다

    start_x = int(width * 0.3)  # 중앙 위험영역의 시작 x좌표를 계산한다
    start_y = int(height * 0.3)  # 중앙 위험영역의 시작 y좌표를 계산한다
    end_x = int(width * 0.7)  # 중앙 위험영역의 끝 x좌표를 계산한다
    end_y = int(height * 0.7)  # 중앙 위험영역의 끝 y좌표를 계산한다

    danger_detected = False  # 전체 화면 기준 위험 상태를 저장하는 변수
    warning_detected = False  # 전체 화면 기준 주의 상태를 저장하는 변수

    results = model(frame, verbose=False)  # 현재 프레임을 YOLO로 분석하되 터미널 로그는 출력하지 않는다

    for result in results:  # 탐지 결과를 하나씩 확인한다
        for box in result.boxes:  # 각 탐지 박스를 하나씩 확인한다
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 박스 좌표를 정수로 가져온다
            cls = int(box.cls[0])  # 클래스 번호를 가져온다
            label = model.names[cls]  # 클래스 이름을 가져온다
            conf = float(box.conf[0])  # 탐지 정확도(신뢰도)를 가져온다

            if label in ["person", "car", "truck", "bus", "bicycle", "motorcycle"] and conf > 0.5:  # 사람/차량 관련 객체이면서 정확도가 50% 이상일 때만 사용한다
                cx = int((x1 + x2) / 2)  # 객체 중심의 x좌표를 계산한다
                cy = int((y1 + y2) / 2)  # 객체 중심의 y좌표를 계산한다

                box_width = x2 - x1  # 박스의 너비를 계산한다
                box_height = y2 - y1  # 박스의 높이를 계산한다
                box_area = box_width * box_height  # 박스의 면적을 계산한다

                in_center = start_x < cx < end_x and start_y < cy < end_y  # 객체 중심이 중앙 위험영역 안에 있는지 확인한다

                if in_center and box_area > 50000:  # 중앙에 있으면서 객체가 매우 크면 위험으로 판단한다
                    color = (0, 0, 255)  # 빨간색을 사용한다
                    text = "DANGER"  # 위험 상태 문구를 설정한다
                    danger_detected = True  # 전체 상태를 위험으로 바꾼다

                    current_time = time.time()  # 현재 시간을 초 단위로 가져온다

                    if current_time - last_log_time >= 1:  # 마지막 저장 이후 1초가 지났을 때만 로그를 저장한다
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간을 문자열로 만든다
                        log_file.write(f"{now} - DANGER detected: {label}\n")  # 위험 객체와 시간을 로그 파일에 저장한다
                        log_file.flush()  # 파일에 즉시 반영한다
                        last_log_time = current_time  # 마지막 로그 저장 시간을 현재 시간으로 바꾼다

                elif in_center and box_area > 20000:  # 중앙에 있으면서 객체가 중간 크기면 주의로 판단한다
                    color = (0, 165, 255)  # 주황색을 사용한다
                    text = "WARNING"  # 주의 상태 문구를 설정한다
                    warning_detected = True  # 전체 상태를 주의로 바꾼다

                else:  # 그 외의 경우는 안전으로 본다
                    color = (0, 255, 0)  # 초록색을 사용한다
                    text = "SAFE"  # 안전 상태 문구를 설정한다

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 객체 위치에 상태별 색상 박스를 그린다
                cv2.putText(frame, f"{label} {text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  # 객체 이름과 상태를 함께 표시한다

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # 중앙 위험영역을 파란 박스로 표시한다
    cv2.putText(frame, "Night Driving Assist", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                2)  # 화면 왼쪽 위에 프로그램 제목을 표시한다

    current_time_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 날짜와 시간을 문자열로 만든다
    cv2.putText(frame, current_time_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                2)  # 화면에 현재 시간을 표시한다
    cv2.putText(frame, "Log Saving: ON", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),
                2)  # 로그 저장 기능이 켜져 있음을 표시한다

    if danger_detected:  # 위험 객체가 하나라도 있으면
        cv2.putText(frame, "DANGER !!!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)  # 전체 상태를 위험으로 표시한다
    elif warning_detected:  # 위험은 아니지만 주의 객체가 하나라도 있으면
        cv2.putText(frame, "WARNING !!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)  # 전체 상태를 주의로 표시한다
    else:  # 위험도 주의도 없으면
        cv2.putText(frame, "SAFE", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)  # 전체 상태를 안전으로 표시한다

    cv2.imshow("Night Assistant Camera", frame)  # 결과 화면을 출력한다

    if cv2.waitKey(1) & 0xFF == ord("q"):  # q 키를 누르면 프로그램을 종료한다
        break  # 반복문을 종료한다

cap.release()  # 사용이 끝난 카메라를 해제한다
cv2.destroyAllWindows()  # OpenCV 창을 모두 닫는다
log_file.close()  # 프로그램 종료 시 로그 파일을 닫는다
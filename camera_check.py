import cv2  # OpenCV 라이브러리를 불러온다

for i in range(3):  # 0번부터 2번 카메라까지 확인한다
    cap = cv2.VideoCapture(i)  # i번 카메라를 연다
    ret, frame = cap.read()  # 프레임을 한 장 읽어본다

    if ret:  # 프레임을 정상적으로 읽었는지 확인한다
        print(f"{i}번 카메라 사용 가능")  # 사용 가능한 카메라 번호를 출력한다
    else:  # 프레임을 못 읽었으면
        print(f"{i}번 카메라 사용 불가")  # 사용할 수 없는 카메라 번호를 출력한다

    cap.release()  # 테스트한 카메라를 닫는다
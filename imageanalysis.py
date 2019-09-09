# -*- coding: utf-8 -*-
import cv2

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    COLOR_WINDOW_NAME = "eyes_number_sketch"
    GRAY_WINDOW_NAME = "face_sketch"

    #今回はPC内蔵カメラのみの使用なのでDEVICE_ID=0とする
    DEVICE_ID = 0
    human = 0

    # 分類器の指定と特徴量の取得
    #カスケード分類器の取得は各自で
    face_cascade = cv2.CascadeClassifier(r"opencv-4.1.1\data\haarcascades\haarcascade_frontalface_alt2.xml")
    eye_cascade = cv2.CascadeClassifier(r"opencv-4.1.1\data\haarcascades\haarcascade_eye.xml")

    # カメラ映像の取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(COLOR_WINDOW_NAME)
    cv2.namedWindow(GRAY_WINDOW_NAME)

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.11, minNeighbors=3, minSize=(1, 1))

        # 検出した顔を矩形で囲む
        for (x, y, w, h) in face:
            human+=1
            cv2.rectangle(img_gray, (x, y), (x+w, y+h), (255,0,0), thickness = 3)
            # 顔部分のみ（グレースケール）
            faceimg_gray = img_gray[y:y+h, x:x+w]
            # 顔部分のみ（カラースケール）　
            faceimg_color = img[y:y+h, x:x+w]
            eye = eye_cascade.detectMultiScale(faceimg_gray, scaleFactor=1.11, minNeighbors=3, minSize=(1, 1))
            for (ex,ey,ew,eh) in eye:
                # 検知した目を矩形で囲む
                cv2.rectangle(faceimg_color,(ex,ey),(ex+ew,ey+eh),(0,255,0), thickness = 1)
        # フレーム表示
        cv2.putText(img,'Number of human = ' + str(human), (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
        human = 0
        cv2.imshow(COLOR_WINDOW_NAME, c_frame)
        cv2.imshow(GRAY_WINDOW_NAME, img_gray)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレームの読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()

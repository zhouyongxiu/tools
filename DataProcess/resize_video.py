# -*- coding: utf-8 -*-

import cv2

# 获得视频的格式
videoCapture = cv2.VideoCapture('2.MP4')

# 获得码率及尺寸
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) / 2),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2))
fourcc = int(videoCapture.get(cv2.CAP_PROP_FOURCC))
# 指定写视频的格式, I420-avi, MJPG-mp4
videoWriter = cv2.VideoWriter('oto_other.mp4', fourcc, fps, size)

# 读帧
success, frame = videoCapture.read()

while success:
    # cv2.imshow("Oto Video", frame)  # 显示
    # cv2.waitKey(1000 / int(fps))  # 延迟
    frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
    # cv2.imshow("Oto Video", frame)  # 显示
    # cv2.waitKey(1000 / int(fps))  # 延迟
    videoWriter.write(frame)  # 写视频帧
    success, frame = videoCapture.read()  # 获取下一帧

videoWriter.release()
videoCapture.release()

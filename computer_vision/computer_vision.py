from skimage.metrics import structural_similarity
import numpy as np
import cv2


def get_median_frame(cap):
    frame_ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=40)
    frames = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)
    return np.median(frames, axis=0).astype(dtype=np.uint8)


def get_diff_contours(current_frame, gray_median_frame):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(gray_median_frame, current_frame_gray, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def draw_rectangle_by_min_area(area_value, contours, current_frame):
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_value:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (36, 255, 12), 2)


def main():
    cap = cv2.VideoCapture('video2.mp4')
    gray_median_frame = cv2.cvtColor(get_median_frame(cap), cv2.COLOR_BGR2GRAY)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret = True
    while ret:
        ret, frame = cap.read()
        contours = get_diff_contours(frame, gray_median_frame)
        draw_rectangle_by_min_area(500, contours, frame)
        cv2.imshow('after', frame)
        if cv2.waitKey(1) == ord('q'):
            break


main()

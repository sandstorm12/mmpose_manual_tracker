import os
import cv2
import yaml
import pickle


BUFFER_SIZE = 100


def _load_detections(configs):
    path = configs['detection']['output']
    path_resume = configs['tracking']['output']
    if configs['tracking']['resume'] and os.path.exists(path_resume):
        path = path_resume

    with open(path, 'rb') as handle:
        detections = pickle.load(handle)

    return detections


def visualize_keypoints(image, detections, selection, idx_frame):
    image = image.copy()

    detections.sort(key=lambda x: x['bbox'][0])

    for idx_person, detection in enumerate(detections):
        color = (0, 255, 0)
        if idx_person == selection:
            color = (255, 0, 0)
        
        cv2.rectangle(
            image,
            (int(detection['bbox'][0]), int(detection['bbox'][1])),
            (int(detection['bbox'][2]), int(detection['bbox'][3])),
            color, 2)

        cv2.putText(image, str(f"id:{detection['id']}"), 
            (int(detection['bbox'][0]), int(detection['bbox'][1])), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 1, 2)
        cv2.putText(image, str(f"{idx_person}"), 
            (int(detection['bbox'][0]), int(detection['bbox'][3])), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 1, 4)
        
    cv2.putText(image, str(f"{idx_frame}"), 
        (image.shape[1] - 100, image.shape[0] - 100), 
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 255, 255), 1, 4)
        
    return image


def _box_track(detections, idx_frame, idx_frame_prev, threshold):
    # Assign ID based on IOU from previous bboxes
    for idx_det_curr, detection_curr in enumerate(detections[idx_frame]):
        if detection_curr['id'] != -1:
            continue

        max_iou = 0
        max_idx_det_prev = -1
        for idx_det_prev, detection_prev in enumerate(detections[idx_frame_prev]):
            bbox_curr = detection_curr['bbox']
            bbox_prev = detection_prev['bbox']

            x_left = max(bbox_curr[0], bbox_prev[0])
            y_top = max(bbox_curr[1], bbox_prev[1])
            x_right = min(bbox_curr[2], bbox_prev[2])
            y_bottom = min(bbox_curr[3], bbox_prev[3])

            if x_right < x_left or y_bottom < y_top:
                iou = 0
            else:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                bbox_curr_area = (bbox_curr[2] - bbox_curr[0]) * (bbox_curr[3] - bbox_curr[1])
                bbox_prev_area = (bbox_prev[2] - bbox_prev[0]) * (bbox_prev[3] - bbox_prev[1])
                iou = intersection_area / float(bbox_curr_area + bbox_prev_area - intersection_area)

            if iou > max_iou:
                max_iou = iou
                max_idx_det_prev = idx_det_prev

        if max_iou > threshold:
            detections[idx_frame][idx_det_curr]['id'] = detections[idx_frame_prev][max_idx_det_prev]['id']
        else:
            detections[idx_frame][idx_det_curr]['id'] = -1



def _track(detections, configs):
    cap = cv2.VideoCapture(configs['input_video'])

    buffer = [None] * BUFFER_SIZE

    play = True
    idx_frame = 0
    idx_latest = 0
    while True:
        if idx_latest == idx_frame:
            ret, frame = cap.read()
            if not ret:
                print("End of file reached.")
                break

            buffer[idx_frame % len(buffer)] = frame
        else:
            frame = buffer[idx_frame % len(buffer)]

        _box_track(detections, idx_frame, idx_frame - 1,
                   configs['tracking']['iou_threshold'])

        selection = -1
        while True:
            frame_vis = visualize_keypoints(frame, detections[idx_frame], selection, idx_frame)

            cv2.imshow("Frame", cv2.resize(frame_vis, configs['detection']['visualize_size']))
            key = cv2.waitKey(0)
            if key == ord('s'):
                play = False
                break
            elif key == ord('q'):
                idx_frame = max(-1, idx_frame - 2, idx_latest - BUFFER_SIZE)
                break
            elif key == ord('e'):
                break
            elif key >= ord('0') and key <= ord('9'):
                if selection != -1:
                    detections[idx_frame][selection]['id'] = key - ord('0')
                    selection = -1
                else:
                    selection = key - ord('0')

        if not play:
            break
        idx_frame += 1
        idx_latest = max(idx_latest, idx_frame)

    cap.release()

    return detections


def _store_results(results, path):
    with open(path, 'wb') as handle:
        pickle.dump(results, handle)


def track(configs):
    detections = _load_detections(configs)

    detections = \
        _track(detections, configs)

    _store_results(detections, configs['tracking']['output'])


# Just for test
if __name__ == "__main__":
    import yaml

    path = '/home/hamid/Documents/phd/mmpose_manual_tracker/sandbox/config_test.yml'

    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    print(configs)

    track(configs)

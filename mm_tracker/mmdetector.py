import cv2
import yaml
import pickle

from tqdm import tqdm
from mmpose.apis import MMPoseInferencer


def _load_model(configs):
    model = MMPoseInferencer(configs['detection']['model'])

    return model


def _get_skeleton(image, model, configs):
    result_generator = model(image)
    
    detections = []
    for result in result_generator:
        poeple_keypoints = result['predictions'][0]
        for predictions in poeple_keypoints:
            bbox = predictions['bbox']
            bbox_conf = predictions['bbox_score']
            keypoint = predictions['keypoints']
            keypoint_conf = predictions['keypoint_scores']
            
            if bbox_conf > configs['detection']['threshold']:
                detections.append({
                    'bbox': bbox[0],
                    'bbox_conf': bbox_conf.item(),
                    'keypoint': keypoint,
                    'keypoint_conf': keypoint_conf,
                    'id': -1,
                })

    return detections


def visualize_keypoints(image, detections):
    for detection in detections:
        for _, point in enumerate(detection['keypoint']):
            cv2.circle(image, (int(point[0]), int(point[1])),
                    3, (0, 255, 0), -1)

        cv2.rectangle(
            image,
            (int(detection['bbox'][0]), int(detection['bbox'][1])),
            (int(detection['bbox'][2]), int(detection['bbox'][3])),
            (0, 255, 0), 2)

        cv2.putText(image, str(f"{detection['bbox_conf']:0.2f}"), 
            (int(detection['bbox'][0]), int(detection['bbox'][1])), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 1, 2)


def _detect(model, configs):
    cap = cv2.VideoCapture(configs['input_video'])

    detections = []

    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if configs['detection']['max_length'] != -1:
        video_frames = min(video_frames, configs['detection']['max_length'])

    for idx_frame in tqdm(range(video_frames)):
        ret, frame = cap.read()
        if not ret:
            print("End of file reached.")
            break

        detections_frame = _get_skeleton(frame, model, configs)

        if len(detections_frame) > 0:
            if configs['detection']['visualize']:
                visualize_keypoints(frame, detections_frame)

        if configs['detection']['visualize']:
            cv2.imshow("Frame", cv2.resize(frame, configs['detection']['visualize_size']))
            if cv2.waitKey(1) == ord('q'):
                break

        detections.append(detections_frame)

    cap.release()

    return detections


def _store_results(results, path):
    with open(path, 'wb') as handle:
        pickle.dump(results, handle)


def detect(configs):
    model = _load_model(configs)

    detections = \
        _detect(model, configs)

    _store_results(detections, configs['detection']['output'])


# Just for test
if __name__ == "__main__":
    import yaml

    path = '/home/hamid/Documents/phd/mmpose_manual_tracker/sandbox/config_test.yml'

    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    print(configs)

    detect(configs)

from time import time, sleep
import queue, threading
import pickle

import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm

from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.util import print_fps
from utils.vitpose_util import keypoints_from_heatmaps
from utils.yolov6_util import letterbox, non_max_suppression, preprocess_with_bboxes, xyxy2xywh, plot_box_and_label 
from utils.prepUtil import calculate_euclidean_distance, shift_points, get_rotation_angle, rotate_cropped_image, rotate_keypoints, convert_ratio, add_padding_cropped_img, calculate_overlap


def inference(original_imgs, yolov6_sess, vitpose_sess, cfg, smooth_net=None):
    """
    Args:
        original_imgs: (numpy.ndarray), (B, H, W, C), RGB color format
        yolov6_sess: (onnxruntime.InferenceSession)
        vitpose_sess: (onnxruntime.InferenceSession)
    Returns:
         infered_imgs: (numpy.ndarray), (B, H, W, C), BGR color format
    """

    
    if cfg.no_background:
        backgrounds = np.zeros_like(original_imgs)
    else:
        backgrounds = original_imgs

    yolov6_img_size = yolov6_sess.get_inputs()[0].shape[-2:]
    vitpose_img_size = vitpose_sess.get_inputs()[0].shape[-2:]
    
    
    # Preprocess images
    processed_imgs = []
    for img in original_imgs:
        img = letterbox(img, yolov6_img_size, auto=False)[0]
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        processed_imgs.append(img)
    
    processed_imgs = np.stack(processed_imgs, axis=0)
    
    
    # Predict bboxes
    preds = []
    input_name = yolov6_sess.get_inputs()[0].name
    for img_batch in np.array_split(processed_imgs, (len(processed_imgs)-1) // cfg.yolo_batch_size + 1):
        preds.append(yolov6_sess.run(None, {input_name: img_batch})[0])

    preds = np.concatenate(preds)
    
    # Postprocess preds
    preds = preds[..., :6] # take only human class
    bbox_groups = non_max_suppression(preds, cfg.conf_thres, cfg.iou_thres)
    
    # Preprocess images for ViTPose input
    processed_imgs = []
    xyxy_groups = []
    conf_groups = []
    sections = []
    detection_check_list = []
    for idx, (original_img, bboxes) in enumerate(zip(original_imgs, bbox_groups)):
        img_list, xyxy_list, conf_list = preprocess_with_bboxes(original_img, bboxes, yolov6_img_size, vitpose_img_size, cfg)
    
        if len(img_list) != 0:
            processed_imgs.append(np.stack(img_list))
            xyxy_groups.append(np.stack(xyxy_list))
            conf_groups.append(conf_list)
            detection_check_list.append(True)
        else:
            detection_check_list.append(False)
        
        sections.append(len(img_list))

    if sum(sections) == 0: # nothing detected
        return backgrounds[..., ::-1].copy(), ([], [])
    
    processed_imgs = np.concatenate(processed_imgs)
    processed_imgs = processed_imgs.transpose(0, 3, 1, 2).astype('float32')
    sections = np.cumsum(sections)
    
    
    # Predict keypoints
    heatmaps = []
    num_batch = (len(processed_imgs)-1) // cfg.pose_batch_size + 1
    input_name = vitpose_sess.get_inputs()[0].name
    for img_batch in np.array_split(processed_imgs, num_batch):
        heatmaps.append(vitpose_sess.run(None, {input_name: img_batch})[0])

    heatmaps = np.concatenate(heatmaps)
    
    # Postprocess heatmaps
    xywh_groups = xyxy2xywh(np.concatenate(xyxy_groups))
    center_xy = xywh_groups[:, [0,1]]
    scale_hw = xywh_groups[:, [2, 3]]

    keypoints, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=center_xy, scale=scale_hw, use_udp=True)
    keypoints = np.concatenate([keypoints[:, :, ::-1], prob], axis=2)
    keypoint_groups = np.split(keypoints, sections)
    
    # Visualization
    pid = 1 # dummy
    infered_imgs = backgrounds[..., ::-1].copy()

    iteration = zip(keypoint_groups, xyxy_groups, conf_groups, backgrounds[..., ::-1])
    for idx, (keypoints, xyxy_list, conf_list, bg_img) in enumerate(iteration):
        if len(keypoints) == 0:
            continue

        img = bg_img.copy()

        if not cfg.no_bbox:
            for xyxy, conf in zip(xyxy_list, conf_list):
                lw = int(np.ceil((xyxy[2]+xyxy[3]-xyxy[0]-xyxy[1]) * 5 / 3000))
                plot_box_and_label(img, lw, xyxy, 'person '+ f'{conf*100:0.0f}%', color=(40,150,30))

        if not cfg.no_skeleton:
            for points, xyxy in zip(keypoints, xyxy_list):
                xywh = xyxy2xywh(xyxy) if cfg.dynamic_drawing else None
                img = draw_points_and_skeleton(img, points, joints_dict()['coco']['skeleton'], person_index=pid,
                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                            points_palette_samples=10, confidence_threshold=cfg.key_conf_thres, xywh=xywh)
                
        infered_imgs[idx] = img

    if cfg.result_scale:
        new_imgs = []
        for img in infered_imgs:
            size = (int(img.shape[1] * cfg.result_scale), int(img.shape[0] * cfg.result_scale))
            img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)
            new_imgs.append(img)
        infered_imgs = np.stack(new_imgs)

    return infered_imgs, (bbox_groups, keypoint_groups)

def extract_keypoints_2(folder_path, yolov6_sess, vitpose_sess, cfg):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"{folder_path} does not exist")
    
    # create a csv file
    csv_data = {
        'img_name':[],
        'img_shape':[],
        'labels': [],
        'bbox':[],
        'keypoints':[]
    }

    target_class_count_label = 0
    target_class_count_csv = 0
    
    for img_name in tqdm(os.listdir(folder_path), desc="Processing images"):
        img_path = os.path.join(folder_path, img_name)
        img_origin = cv2.imread(img_path)
        img_copy = img_origin.copy()
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

        labels = []
        if label_path:
            with open(label_path, 'r') as f:
                for line in f: 
                    class_id, x, y, w, h = map(float, line.strip().split())
                    if class_id in cfg.target_class:
                        xmin, ymin = int((x - w / 2) * img_origin.shape[1]), int((y - h / 2) * img_origin.shape[0])
                        xmax, ymax = int((x + w / 2) * img_origin.shape[1]), int((y + h / 2) * img_origin.shape[0])
                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                        target_class_count_label += 1
                        labels.append([int(class_id), xmin, ymin, xmax, ymax])

        img_origin = img_origin[..., ::-1] # BGR to RGB
        img_origin = np.expand_dims(img_origin, axis=0)
        _, pred = inference(img_origin, yolov6_sess, vitpose_sess, cfg)
        if len(pred[0]) == 0:
            continue
        bboxs = pred[0][0]
        keypoints = pred[1][0]

        for index, (bbox, keypoint) in enumerate(zip(bboxs, keypoints)):
            class_id = 0
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            temp_bbox = [xmin, ymin, xmax, ymax]

            if len(labels) != 0:
                for label in labels:
                    overlap_percentage = calculate_overlap(label[1:], temp_bbox)
                    euclid_distances = calculate_euclidean_distance(temp_bbox, label[1:])
                    # print(f'overlap_percentage: {overlap_percentage}, euclid_distances: {euclid_distances}')
                    # print('-------')
                    if overlap_percentage >= cfg.matched_overlap_thresh and euclid_distances <= cfg.euclid_thresh:
                        target_class_count_csv += 1
                        class_id = 1
                        break
                    else:
                        if overlap_percentage >= 0.9 or euclid_distances <= 10:
                            target_class_count_csv += 1
                            class_id = 1
                            break

            cropped_img = img_copy[ymin:ymax, xmin:xmax]
            scaled_bbox, scale_ratio, padding_ratio = letterbox(im=cropped_img, new_shape=(64,64), scaleup=True, stride=64)

            for i, pt in enumerate(keypoint):
                if pt[2] <= 0.5:
                    keypoint[i] = [0, 0, pt[2]]
                    continue

                new_pt_x, new_pt_y = shift_points([xmin, ymin, xmax, ymax, pt[1], pt[0], scale_ratio, padding_ratio])
                keypoint[i] = [new_pt_x, new_pt_y, pt[2]]

            keypoint = keypoint[:, :2]

            if cfg.normalize_kps:
                angle, _ = get_rotation_angle(keypoint)
                if angle is not None:
                    cropped_padding, padded_keypoint = add_padding_cropped_img(scaled_bbox, keypoint)
                    rotated_image, M = rotate_cropped_image(cropped_padding, angle)
                    rotated_keypoints = rotate_keypoints(padded_keypoint, M)
                    norm_keypoint = convert_ratio(rotated_image, rotated_keypoints)
                else:
                    norm_keypoint = convert_ratio(scaled_bbox, keypoint)
            else:
                norm_keypoint = keypoint

            csv_data['img_name'].append(img_name)
            csv_data['img_shape'].append((yolov6_sess.get_inputs()[0].shape[-2], yolov6_sess.get_inputs()[0].shape[-1]))
            csv_data['labels'].append(class_id)
            csv_data['bbox'].append(temp_bbox)
            csv_data['keypoints'].append(norm_keypoint)
            
    df = pd.DataFrame(csv_data)
    save_name = folder_path.replace("images", "") + 'labels.csv'
    df.to_csv(save_name, index=False)
    print(f"Extracted keypoints are saved to {save_name}")
    print(f"Target class count in labels: {target_class_count_label}")
    print(f"Target class count in csv: {target_class_count_csv}")
    print(f"Successfully extract {round(target_class_count_csv/target_class_count_label*100, 2)}% of target class in labels")

def inference_image(img_path, yolov6_sess, vitpose_sess, cfg):
    img_origin = cv2.imread(img_path)
    img_origin = img_origin[..., ::-1] # BGR to RGB
    img_origin = np.expand_dims(img_origin, axis=0)
    img, pred = inference(img_origin, yolov6_sess, vitpose_sess, cfg)

    print('-'*10 + "\nPress 'Q' key on OpenCV window if you want to close")
    cv2.imshow("OpenCV", img[0])

    if cfg.save:
        save_name = img_path.replace(".jpg", "_result.jpg")
        cv2.imwrite(save_name, img[0])
    if cfg.save_prediction:
        preds = {'bbox':[], 'pose':[]}
        preds['bbox'].extend(pred[0])
        preds['pose'].extend(pred[1])
        save_name = img_path.replace(".jpg", "_prediction.pkl")
        with open(save_name, 'wb') as f:
            pickle.dump(preds, f)

    cv2.waitKey(0)




def inference_video(vid_path, yolov6_sess, vitpose_sess, cfg, smooth_net=None):
    video = cv2.VideoCapture(vid_path)
    frames = []
    preds = {'bbox':[], 'pose':[]}

    if cfg.save:
        out_name = '.'.join(vid_path.split('.')[:-1]) + '_result.mp4'
        out_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if cfg.set_fps is not None:
            out_fps = cfg.set_fps
        else:
            out_fps = video.get(cv2.CAP_PROP_FPS)
        out_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(out_name, out_fourcc, out_fps, out_size)

    print('-'*10 + "\nPress 'Q' key on OpenCV window if you want to close")
    tic = time()
    while(True):
        ret, frame = video.read()

        if ret:
            # if not cfg.save:
            #     frame = cv2.resize(frame, (1280, 720))
            frames.append(frame)
            if len(frames) < cfg.yolo_batch_size:
                continue

            frames = np.stack(frames)
            frames = frames[..., ::-1] # BGR to RGB
            results, pred = inference(frames, yolov6_sess, vitpose_sess, cfg)

            toc = time()
            fps = 1/(toc - tic)
            tic = time()

            print_fps(fps*cfg.yolo_batch_size)
            
            cv2.imshow('OpenCV', results[-1])

            if cfg.save:
                for res in results:
                    out.write(res)
            if cfg.save_prediction:
                preds['bbox'].extend(pred[0])
                preds['pose'].extend(pred[1])

            frames = []

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    
    if cfg.save_prediction:
        save_name = '.'.join(vid_path.split('.')[:-1]) + '_prediction.pkl'
        with open(save_name, 'wb') as f:
            pickle.dump(preds, f)

    video.release()
    if cfg.save: out.release()
    cv2.destroyAllWindows()




def inference_webcam(webcam, yolov6_sess, vitpose_sess, cfg, smooth_net=None):
    event = threading.Event()

    # bufferless VideoCapture
    cap = AsyncVideoCapture(webcam, event)
    preds = {'bbox':[], 'pose':[]}

    if cfg.save:
        frame_queue = queue.Queue(1)

        out_name = 'webcam_result.mp4'
        out_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = cfg.set_fps
        out_size = (int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        out = AsyncVideoWriter(out_name, out_fourcc, out_fps, out_size, cap, frame_queue, event)

    print('-'*10 + "\nPress 'Q' key on OpenCV window if you want to close")
    tic = time()
    while not cap.is_dead:
        frame = cap.read()

        frame = frame[..., ::-1] # BGR to RGB
        frame = np.expand_dims(frame, axis=0)
        frame, pred = inference(frame, yolov6_sess, vitpose_sess, cfg)

        toc = time()
        fps = 1/(toc - tic)
        tic = time()
        
        print_fps(fps)

        cv2.imshow("OpenCV", frame[0])

        if cfg.save:
            try:
                frame_queue.put_nowait(frame[0])
            except queue.Full:
                pass

        if cfg.save_prediction:
                preds['bbox'].extend(pred[0])
                preds['pose'].extend(pred[1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.event.set()
            break

    if cfg.save_prediction:
        save_name = 'webcam_prediction.pkl'
        with open(save_name, 'wb') as f:
            pickle.dump(preds, f)
    
    time_out = 10.0
    tic = time()
    while (not cap.is_dead) or (cfg.save and not out.is_dead):
        toc = time()
        if toc-tic > time_out:
            break

    cv2.destroyAllWindows()



class AsyncVideoCapture:
    def __init__(self, webcam, event):
        self.cap = cv2.VideoCapture(webcam)
        if self.cap.isOpened():
            self.event = event
            self.q = queue.Queue()
            t = threading.Thread(target=self._reader)
            t.daemon = True
            t.start()
            self.is_dead = False
        else:
            self.cap.release()
            self.is_dead = True

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()

            if (not ret) or self.event.is_set():
                break

            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass

            self.q.put(frame)

        self.cap.release()
        self.is_dead = True

    def read(self):
        return self.q.get()
    

class AsyncVideoWriter:
    def __init__(self, out_name, out_fourcc, out_fps, out_size, cap, frame_queue, event):
        self.out = cv2.VideoWriter(out_name, out_fourcc, out_fps, out_size)
        if self.out.isOpened():
            self.cap = cap
            self.event = event
            self.last_frame = np.zeros((out_size[1], out_size[0],3), np.uint8)
            self.frame_queue = frame_queue
            t = threading.Thread(target=self._writer)
            t.daemon = True
            t.start()
            self.period = 1/out_fps
            self.is_dead = False
        else:
            self.out.release()
            self.is_dead = True

    def _writer(self):
        diff = 0
        tic = time()
        while True:
            if self.event.is_set():
                break

            try:
                self.last_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            
            self.out.write(self.last_frame)

            # match writing speed to the desire fps
            elapsed_time =  time() - tic
            tic = time()
            diff += self.period - elapsed_time
            if diff > 0:
                sleep(diff)
            
        self.out.release()
        self.is_dead = True
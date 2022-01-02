import argparse
import math
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, get_corners
from utils.torch_utils import select_device, load_classifier, time_synchronized

import pyautogui

state = None


def detect(save_img=False):
    state = None
    VERDICT = ''
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        prev = 0
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # my changes
            # list to store all the fingers
            fingers = []



            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                detections = {'LEFT': [], 'RIGHT': [], 'ENGINE': [], 'BRAKE': []}

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        c1, c2 = get_corners(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        detections[names[int(cls)]].append([(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, xyxy])

                        # c1, c2 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        # print("plot log", c1, c2)
                        # fingers.append([(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2])

                # filter the values
                # for left
                lefts = list(detections['LEFT'])
                lefts.sort()

                if len(lefts):
                    detections['LEFT'] = lefts[0]

                # for rights
                rights = list(detections['RIGHT'])
                rights.sort(reverse=True)

                if len(rights):
                    detections['RIGHT'] = rights[0]

                if len(detections['ENGINE']):
                    detections['ENGINE'] = detections['ENGINE'][0]

                if len(detections['BRAKE']):
                    detections['BRAKE'] = detections['BRAKE'][0]

                # labelling

                for key in detections:
                    if len(detections[key]):
                        # print(key, detections[key])
                        plot_one_box(detections[key][2], im0, label=key, color=colors[int(cls)], line_thickness=1)

                # DECISION MAKING
                NEW_VERDICT = ''
                if len(detections['LEFT']) and len(detections['RIGHT']):
                    if state is None:
                        # calibration
                        # # print("calibrating")
                        # print(detections)
                        origin = math.atan((detections['LEFT'][1] - detections['RIGHT'][1])/ (detections['LEFT'][0] - detections['RIGHT'][0]))
                        state = "CALIBRATED"
                        origin = math.degrees(origin)
                        print("CALIBRATION COMPLETE")
                        condition = True

                    else:


                        current = math.atan((detections['LEFT'][1] - detections['RIGHT'][1])/ (detections['LEFT'][0] - detections['RIGHT'][0]))
                        current = math.degrees(current)
                        print("CURRENT", current, "ORIGIN", origin)

                        if abs(current - origin) < 10:
                            print('FRONT', end=" ")
                            NEW_VERDICT += 'FRONT'
                            pass
                            # pyautogui.press('LEFT')
                        else:
                            if current > origin:
                                # pyautogui.press('left')
                                if len(VERDICT):
                                    NEW_VERDICT += '_LEFT'
                                print('LEFT', end=" ")
                            else:
                                if len(VERDICT):
                                    NEW_VERDICT += '_RIGHT'
                                # pyautogui.press('right')
                                print('RIGHT', end=" ")

                if len(detections['ENGINE']):
                    if len(VERDICT):
                        NEW_VERDICT += '_ENGINE'
                    # pyautogui.press('up')
                    print('ENGINE', end=" ")

                if len(detections['BRAKE']):
                    if len(VERDICT):
                        NEW_VERDICT += '_DOWN'
                    # pyautogui.press('down')
                    print('BRAKE', end=" ")

                print("")
                print("VERDICT", NEW_VERDICT)
                if VERDICT != NEW_VERDICT or 1:

                    VERDICT = NEW_VERDICT
                    verd = VERDICT.split('_')
                    if 'LEFT' in verd:
                        pyautogui.keyDown('left')
                        time.sleep(0.1)
                        pyautogui.keyUp('left')
                    else:
                        pyautogui.keyUp('left')

                    if 'RIGHT' in verd:
                        pyautogui.keyDown('right')
                        time.sleep(0.1)
                        pyautogui.keyUp('right')
                    else:
                        pyautogui.keyUp('right')

                    if 'ENGINE' in verd:
                        pyautogui.keyDown('up')
                    else:
                        pyautogui.keyUp('up')

                    if 'BRAKE' in verd:
                        pyautogui.keyDown('down')
                    else:
                        pyautogui.keyUp('down')


            # # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            # print("total no of FIGNERS", len(fingers), ":", fingers)
            #
            #
            # if prev != len(fingers):
            #     # now move the mouse
            #     if len(fingers) == 1:
            #         # pyautogui.moveTo(fingers[0][0] * 4, fingers[0][1] * 4)
            #         pyautogui.press('space')
            #
            #     elif len(fingers) == 2:
            #         # pyautogui.click()
            #         pyautogui.press('right')
            #
            #     elif len(fingers) == 3:
            #         pyautogui.press('left')
            #
            #     elif len(fingers) == 4:
            #         exit()
            #     prev = len(fingers)

            # Stream results
            if view_img:
                # my code
                # resizing
                # define the screen resulation
                screen_res = 1920, 1080
                scale_width = screen_res[0] / img.shape[1]
                scale_height = screen_res[1] / img.shape[0]
                scale = min(scale_width, scale_height)
                # resized window width and height
                window_width = int(img.shape[1] * scale)
                window_height = int(img.shape[0] * scale)
                # cv2.WINDOW_NORMAL makes the output window resizealbe
                cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
                # resize the window according to the screen resolution
                width, height = pyautogui.size()
                width = int(width / 4)
                height = int(height / 4)
                cv2.resizeWindow('Resized Window', (width, height))

                cv2.imshow('Resized Window', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    # print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

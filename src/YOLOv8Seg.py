import cv2
import numpy as np
import onnxruntime as ort
import os

from onnx_pipeline import Colors
import evaluation_metrics as eval

class YOLOv8Seg:
    def __init__(self, onnx_model):
        # Load the ONNX model
        self.session = ort.InferenceSession(onnx_model)
        self.ndtype = np.float32 if self.session.get_inputs()[0].type == 'tensor(float)' else np.float16
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]
        self.color_palette = Colors()

    # Perform inference and return boxes, segments, and masks
    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})
        #print(len(preds))  # Debugging: print number of predictions
        boxes, segments, masks = self.postprocess(preds, im0=im0, ratio=ratio, pad_w=pad_w, pad_h=pad_h,
                                                  conf_threshold=conf_threshold, iou_threshold=iou_threshold, nm=nm)
        return boxes, segments, masks

    # Preprocess the input image (resize and normalize)
    def preprocess(self, img):
        shape = img.shape[:2]
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = np.ascontiguousarray(np.einsum('HWC->CHW', img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    # Postprocess the model predictions: filter boxes, generate masks and segments
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        x, protos = preds[0], preds[1]
        print(x.shape, protos.shape)  # Debugging: print shapes of predictions
        x = np.einsum('bcn->bnc', x)
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]
        if len(x) > 0:
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks
        else:
            return [], [], []

    # Process masks based on bounding boxes and resize them to fit the original image
    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)
        masks = np.einsum('HWN -> NHW', masks)
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    # Scale the masks to match the size of the original image
    def scale_mask(self, masks, im0_shape, ratio_pad=None):
        im1_shape = masks.shape[:2]
        if ratio_pad is None:
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2
        else:
            pad = ratio_pad[1]
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    # Crop the mask to match the bounding boxes
    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    # Convert the masks to contour segments
    def masks2segments(self, masks):
        segments = []
        for mask in masks.astype(np.uint8):
            c = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))
            segments.append(c.astype('float32'))
        return segments

    # Draw bounding boxes and segmentation masks on the image
    def draw_and_visualize(self, im, bboxes, segments, vis=True, save=False):
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          self.color_palette(int(cls_), bgr=True), 1, cv2.LINE_AA)
            cv2.putText(im, f'Class {cls_}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_palette(int(cls_), bgr=True), 2, cv2.LINE_AA)
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
        return im

    def get_pred_and_true(self, image, txt_file_path, og_shape=(1024, 1280)):
        """Get predicted and true masks."""
        boxes, segments, masks = self(image)
        y_pred = (prediction[0].masks.data.squeeze(0) * 255).numpy().astype(np.uint8)
        print('get_pred_and_true y_pred, prediction.mask shape: ', y_pred.shape, prediction[0].masks.data.shape)
        mask_image = eval.process_yolo_txt_file(txt_file_path, og_shape)
        y_true = cv2.resize(mask_image * 255, (y_pred.shape[1], y_pred.shape[0]))

        return y_true, y_pred

    # Evaluate model
    def evaluate(self, images_dir, txt_dir, log_dir='./', metrics='all'):
        """
        Evaluates the model on specified metrics

        Args:
            images_dir: Path to the folder containing input images.
            txt_dir: Path to the folder containing ground truth masks.
            pred_dir: Path to the folder where predictions should be saved.
            metrics: Metrics to be calculated. Values can be one of 'dice', 'hausd' or 'all'.
        
        Returns:

        """

        # Validate metrics and set flags
        get_dice = False
        get_hausd = False

        if metrics == 'all':
            get_dice = True
            get_hausd = True
        elif metrics == 'dice':
            get_dice = True
        elif metrics == 'hausd':
            get_hausd = True
        else:
            # Throw error
            raise ValueError("Please enter a valid metric string from: 'dice', 'hausd' or 'all'.")

        log_file_path = os.path.join(log_dir, "metrics_log.txt") 

        # Open the log file for writing metrics
        with open(log_file_path, 'w') as log_file:
            log_file.write("Image Name,Dice Coefficient,Hausdorff Distance\n")

            # Process each image in the specified directory
            for image_name in os.listdir(images_dir):
                if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
                    image_path = os.path.join(images_dir, image_name)
                    txt_file_path = os.path.join(txt_dir, os.path.splitext(image_name)[0] + '.txt')
                    
                    if not os.path.exists(txt_file_path):
                        print(f"Warning: No corresponding TXT file found for {image_name}. Skipping.")
                        continue
                    
                    img = cv2.imread(image_path)
                    y_true, y_pred = self.get_pred_and_true(img, txt_file_path)
                    print('Shapes of img, y_true, y_pred', img.shape, y_true.shape, y_pred.shape)
                    # Calculate metrics
                    if get_dice:
                        dice_score = eval.dice(y_pred, y_true)
                    if get_hausd:
                        hausdorff_dist = eval.hausdorff_distance_mask(y_true, y_pred)

                    # Log metrics to file
                    log_file.write(f"{image_name},{dice_score:.4f},{hausdorff_dist:.4f}\n")
                    print(f"Processed {image_name}: Dice={dice_score:.4f}, Hausdorff={hausdorff_dist:.4f}")

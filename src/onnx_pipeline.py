import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse

# Color class for visualization (hex colors converted to RGB)
class Colors:
    """
    A utility class to handle color palettes for visualization.

    Attributes:
        palette (list): List of colors in RGB format.
        n (int): Number of colors in the palette.
    """
    def __init__(self):
        """
        Initializes the Colors class with a predefined color palette in hex format.
        The hex values are converted to RGB.
        """
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """
        Returns the RGB or BGR color from the palette.

        Args:
            i (int): Index of the color in the palette.
            bgr (bool): If True, returns the color in BGR format.

        Returns:
            tuple: RGB or BGR color.
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """
        Converts a hex color code to RGB.

        Args:
            h (str): Hex color code.

        Returns:
            tuple: RGB color as a tuple.
        """
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# YOLOv8 Segmentation class that handles inference and visualization
class YOLOv8Seg:
    """
    A class to handle inference and visualization for YOLOv8 segmentation models using ONNX runtime.

    Attributes:
        session (onnxruntime.InferenceSession): ONNX runtime session for the loaded model.
        ndtype (numpy.dtype): Data type of the model input.
        model_height (int): Height of the model input.
        model_width (int): Width of the model input.
        color_palette (Colors): Instance of the Colors class for visualization.
    """
    def __init__(self, onnx_model, num_threads=3, num_streams=1):
        """
        Initializes the YOLOv8Seg model by loading the ONNX model and setting the input dimensions.

        Args:
            onnx_model (str): Path to the ONNX model file.
        """
        session_options = ort.SessionOptions()
        session_options.enable_profiling = False 
        session_options.intra_op_num_threads = num_threads   # Number of threads within a single operation
        session_options.inter_op_num_threads = num_streams   # Number of threads across multiple operations
        self.session = ort.InferenceSession(onnx_model, sess_options=session_options)
        self.ndtype = np.float32 if self.session.get_inputs()[0].type == 'tensor(float)' else np.float16
        self.model_height, self.model_width = [x.shape for x in self.session.get_inputs()][0][-2:]
        self.color_palette = Colors()

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        Performs inference on an input image and returns the predicted bounding boxes, segments, and masks.

        Args:
            im0 (numpy.ndarray): Input image.
            conf_threshold (float): Confidence threshold for filtering boxes.
            iou_threshold (float): IoU threshold for non-maximum suppression.
            nm (int): Number of mask channels in the model output.

        Returns:
            tuple: (boxes, segments, masks) where:
                - boxes: Array of bounding box coordinates, confidence, and class predictions.
                - segments: List of segmentation contour segments.
                - masks: Boolean array representing the masks for the detected objects.
        """
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)
        preds = self.session.run(None, {self.session.get_inputs()[0].name: im})
        print(len(preds))  # Debugging: print number of predictions
        boxes, segments, masks = self.postprocess(preds, im0=im0, ratio=ratio, pad_w=pad_w, pad_h=pad_h,
                                                  conf_threshold=conf_threshold, iou_threshold=iou_threshold, nm=nm)
        return boxes, segments, masks

    def preprocess(self, img):
        """
        Preprocesses the input image by resizing and padding it to match the model input size.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            tuple: (processed_image, ratio, padding) where:
                - processed_image: Image resized and padded to fit the model input.
                - ratio: Scaling ratio applied to the image.
                - padding: Padding applied to the image (pad_w, pad_h).
        """
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

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Postprocesses the model predictions by filtering boxes and generating masks and segments.

        Args:
            preds (list): Model predictions.
            im0 (numpy.ndarray): Original input image before preprocessing.
            ratio (tuple): Ratio used for scaling the image.
            pad_w (float): Width of the padding applied during preprocessing.
            pad_h (float): Height of the padding applied during preprocessing.
            conf_threshold (float): Confidence threshold for filtering boxes.
            iou_threshold (float): IoU threshold for non-maximum suppression.
            nm (int): Number of mask channels in the model output.

        Returns:
            tuple: (boxes, segments, masks) where:
                - boxes: Array of bounding box coordinates, confidence, and class predictions.
                - segments: List of segmentation contour segments.
                - masks: Boolean array representing the masks for the detected objects.
        """
        x, protos = preds[0], preds[1]
        print(x.shape, protos.shape)
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

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Processes the mask predictions by resizing and applying the bounding boxes to crop the masks.

        Args:
            protos (numpy.ndarray): Prototype masks from the model output.
            masks_in (numpy.ndarray): Predicted mask coefficients.
            bboxes (numpy.ndarray): Bounding box coordinates.
            im0_shape (tuple): Shape of the original input image.

        Returns:
            numpy.ndarray: Resized and cropped masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)
        masks = np.einsum('HWN -> NHW', masks)
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    def scale_mask(self, masks, im0_shape, ratio_pad=None):
        """
        Scales the masks to match the original image size.

        Args:
            masks (numpy.ndarray): Mask predictions.
            im0_shape (tuple): Shape of the original input image.
            ratio_pad (tuple, optional): Padding ratios applied during preprocessing.

        Returns:
            numpy.ndarray: Resized masks.
        """
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

    def crop_mask(self, masks, boxes):
        """
        Crops the masks to match the bounding box regions.

        Args:
            masks (numpy.ndarray): Mask predictions.
            boxes (numpy.ndarray): Bounding box coordinates.

        Returns:
            numpy.ndarray: Cropped masks.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def masks2segments(self, masks):
        """
        Converts the binary masks into contour segments for visualization.

        Args:
            masks (numpy.ndarray): Binary masks for detected objects.

        Returns:
            list: List of contour segments for each mask.
        """
        segments = []
        for mask in masks.astype(np.uint8):
            c = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))
            segments.append(c.astype('float32'))
        return segments

    def draw_and_visualize(self, im, bboxes, segments, vis=True, save=False):
        """
        Draws bounding boxes and segmentation masks on the input image and visualizes or saves the output.

        Args:
            im (numpy.ndarray): Original input image.
            bboxes (numpy.ndarray): Detected bounding boxes.
            segments (list): Segmentation contour segments.
            vis (bool): If True, displays the image.
            save (bool): If True, saves the output image.

        Returns:
            numpy.ndarray: Image with drawn bounding boxes and masks.
        """
        im_canvas = im.copy()
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)
            cv2.fillPoly(im_canvas, np.int32([segment]), self.color_palette(int(cls_), bgr=True))
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          self.color_palette(int(cls_), bgr=True), 1, cv2.LINE_AA)
            cv2.putText(im, f'Class {cls_}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_palette(int(cls_), bgr=True), 2, cv2.LINE_AA)
            print(int(box[0]), int(box[1]), int(box[2]), int(box[3]), conf, cls_)
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
        return im

if __name__ == '__main__':
    # Argument parsing for command-line options
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation ONNX Inference')
    parser.add_argument('--model', type=str, default='best.onnx', help='Path to the ONNX model file')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image file')
    parser.add_argument('--output', type=str, default='output_segmented_image.jpg', help='Path to save the output image')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use for inference')
    parser.add_argument('--num_streams', type=int, default=1, help='Number of streams to use for inference')
    args = parser.parse_args()

    start_time = time.time()

    # Initialize the model
    model = YOLOv8Seg(args.model, num_threads=args.num_threads, num_streams=args.num_streams)

    # Load input image
    img = cv2.imread(args.input)

    # Run inference
    boxes, segments, _ = model(img, conf_threshold=0.4, iou_threshold=0.45)

    # Draw and visualize the result
    if len(boxes) > 0:
        output_image = model.draw_and_visualize(img, boxes, segments, vis=False, save=True)
    else:
        output_image = img

    # Save and display result
    cv2.imwrite(args.output, output_image)

    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    print(f"Output image saved to: {args.output}")

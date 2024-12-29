# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ROOT, ops
import copy
import torch


class DetectionPredictor(BasePredictor): 
    def postprocess_original(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred,
                                   appearance_features=None, appearance_feature_map=None))
        return results
    
    def postprocess(self, preds, img, orig_imgs, appearance_feature_layer=None, return_feature_map=False):
        """Post-processes predictions and returns a list of Results objects."""
        if appearance_feature_layer is None:
            return self.postprocess_original(preds, img, orig_imgs)

        # appearance_feature_layer can be layer1 ... layer21
        features_dict = copy.deepcopy(preds[2])
        
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )
        
        preds_map = copy.deepcopy(preds)
        
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            features, feature_map = self.extract_appearance_features(features_dict, preds_map,
                                            appearance_feature_layer, img)
            if not return_feature_map:
                feature_map = None
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred
                                   ,appearance_features=features, appearance_feature_map=feature_map))
        return results

   
    def extract_appearance_features(self, feature_map, preds, appearance_feature_layer, img):
        # (48, 368, 640)
        feature_map = feature_map[appearance_feature_layer][0, :, :, :]
        reshaped_feature_map = feature_map.permute(1, 2, 0)  # (368, 640, 48)

        feature_dim = reshaped_feature_map.shape[-1]  # cmap

        preds[0][:, :4] = ops.scale_boxes(
            img.shape[2:], preds[0][:, :4], reshaped_feature_map.shape)
        boxes = preds[0][:, :4].long().cpu().numpy()
        features_normalized = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box

            # (48, height, width)
            extracted_feature = feature_map[:, y_min:y_max, x_min:x_max]

            if 0 not in extracted_feature.shape:
                feature_mean = torch.mean(
                    extracted_feature, dim=(1, 2))  # (48,)
                normalized_feature = feature_mean / \
                    feature_mean.norm(p=2, dim=0, keepdim=True)
            else:
                normalized_feature = torch.ones(
                    feature_dim, dtype=torch.float32, device=reshaped_feature_map.device)

            features_normalized.append(normalized_feature)

        features = torch.stack(
            features_normalized, dim=0) if features_normalized else torch.tensor([])

        return features, feature_map


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()

import wandb

from torchvision import transforms
from pytorch_lightning.callbacks import Callback
 
class LogPredictionsCallback(Callback):

    def __init__(self, wandb_logger, class_names):
        super().__init__()
        self.threshold = 0.75
        self._wandb_logger = wandb_logger
        self.class_labels = {i: x for i, x in enumerate(class_names)}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            n = 5
            img_list, targets = batch
            img_list = img_list[:n]
            targets = targets[:n]

            columns = ['ID', 'Image']
            data = []

            filtered_output = self._filter_model_output(outputs, self.threshold)

            for index, (image, prediction) in enumerate(zip(img_list, filtered_output[:n])):
                boxes = prediction.get('boxes')
                scores = prediction.get('scores')
                labels = prediction.get('labels').detach().cpu().numpy()

                image = transforms.ToPILImage()(image.detach().cpu())

                wboxes = self._get_wandb_bboxes(boxes, scores, labels)

                img = wandb.Image(
                    image,
                    boxes=wboxes,
                )

                data.append([index, img])

            if len(data) > 0:
                self._wandb_logger.log_table(key='Sample Val Pred', columns=columns, data=data)


    def _filter_model_output(self, outputs, score_threshold):
        filtred_output = list()
        for image in outputs:
            filtred_image = dict()
            for key in image.keys():
                filtred_image[key] = image[key][image['scores'] >= score_threshold]
                filtred_output.append(filtred_image)
        return filtred_output


    def _get_wandb_bboxes(self, bboxes, scores, labels):
        wandb_boxes = {}

        box_data = []
        for bbox, score, label in zip(bboxes, scores, labels):
            if not isinstance(label, int):
                label = int(label)

            if len(bbox) == 5:
                confidence = float(bbox[4])
                class_name = self.class_labels[label]
                box_caption = f'{class_name} {confidence:.2f}'
            else:
                box_caption = str(self.class_labels[label])

            position = dict(
                minX=int(bbox[0]),
                minY=int(bbox[1]),
                maxX=int(bbox[2]),
                maxY=int(bbox[3]))

            box_data.append({
                'position': position,
                'class_id': label,
                "scores" : {
                    "acc" : float(score)
                },
                'box_caption': box_caption,
                'domain': 'pixel'
            })

        wandb_bbox_dict = {
            'box_data': box_data,
            'class_labels': self.class_labels
        }

        wandb_boxes['predictions'] = wandb_bbox_dict

        return wandb_boxes
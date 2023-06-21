import os
import logging
import boto3
import io
import json


from mmdet.apis import init_detector, inference_detector
from torch import hub

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class yolov5(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(yolov5, self).__init__(**kwargs)

        # Model
        # or yolov5n - yolov5x6, custom
        self.model = hub.load("ultralytics/yolov5", "yolov5s")

    def _get_image_url(self, task):
        image_url = task['data'].get(
            self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3', endpoint_url=self.endpoint_url)
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(
                    f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]

        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)

        model_results = inference_detector(self.model, image_path)
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        for bboxes, label in zip(model_results, self.model.CLASSES):
            output_label = self.label_map.get(label, label)

            if output_label not in self.labels_in_config:
                print(output_label + ' label not found in project config.')
                continue
            for bbox in bboxes:
                bbox = list(bbox)
                if not bbox:
                    continue
                score = float(bbox[-1])
                if score < self.score_thresh:
                    continue
                x, y, xmax, ymax = bbox[:4]
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': x / img_width * 100,
                        'y': y / img_height * 100,
                        'width': (xmax - x) / img_width * 100,
                        'height': (ymax - y) / img_height * 100
                    },
                    'score': score
                })
                all_scores.append(score)
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        return [{
            'result': results,
            'score': avg_score
        }]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data

import os
import logging
import boto3
import io
import json

from torch import hub

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class YOLOv5(LabelStudioMLBase):

    def __init__(self, image_dir=None, labels_file=None, score_threshold=0.0,  **kwargs):
        super(YOLOv5, self).__init__(**kwargs)

        self.labels_file = labels_file
        self.endpoint_url = kwargs.get('endpoint_url')
        self.score_thresh = score_threshold

        # default Label Studio image upload folder
        # get data_dir from environment variable or from settings
        upload_dir = os.path.join(os.environ.get('DATA_DIR'), 'media', 'upload') or os.path.join(
            get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for label_name, label_attrs in self.labels_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

        # Model
        # or yolov5n - yolov5x6, custom
        self.model = hub.load("ultralytics/yolov5", "custom", path="best.pt")
        # self.model.cuda() # use GPU

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

        model_results = self.model(image_path, size=1280)
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)

        df = model_results.pandas().xyxy[0]
        for row in range(len(df)):
            try:
                xmin = df.at[row, 'xmin']
                ymin = df.at[row, 'ymin']
                xmax = df.at[row, 'xmax']
                ymax = df.at[row, 'ymax']
                confidence = df.at[row, 'confidence']
                class_name = df.at[row, 'name']
                class_id = df.at[row, 'class']

                if confidence < 0.5:
                    continue

                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',

                    'value': {
                        'rectanglelabels': [class_name],
                        'x': xmin / img_width * 100,
                        'y': ymin / img_height * 100,
                        'width': (xmax - xmin) / img_width * 100,
                        'height': (ymax - ymin) / img_height * 100
                    },
                    'score': confidence
                })
                all_scores.append(confidence)
            except Exception as e:
                print(e)
                continue
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        return [{
            'result': results,
            'score': avg_score
        }]

        #     # Compare this snippet from label_studio_ml\examples\mmdetection copy\mmdetection.py:

        #     if output_label not in self.labels_in_config:
        #         print(output_label + ' label not found in project config.')
        #         continue
        #     for bbox in bboxes:
        #         bbox = list(bbox)
        #         if not bbox:
        #             continue
        #         score = float(bbox[-1])
        #         if score < self.score_thresh:
        #             continue
        #         x, y, xmax, ymax = bbox[:4]
        #         results.append({
        #             'from_name': self.from_name,
        #             'to_name': self.to_name,
        #             'type': 'rectanglelabels',
        #             'value': {
        #                 'rectanglelabels': [output_label],
        #                 'x': x / img_width * 100,
        #                 'y': y / img_height * 100,
        #                 'width': (xmax - x) / img_width * 100,
        #                 'height': (ymax - y) / img_height * 100
        #             },
        #             'score': score
        #         })
        #         all_scores.append(score)
        # avg_score = sum(all_scores) / max(len(all_scores), 1)
        # return [{
        #     'result': results,
        #     'score': avg_score
        # }]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data

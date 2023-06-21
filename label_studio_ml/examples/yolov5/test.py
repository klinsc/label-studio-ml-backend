from torch import hub
from label_studio_ml.utils import get_image_size

# write a test function


def test_yolov5():
    model = hub.load("ultralytics/yolov5", "custom", path="best.pt")
    imageURL = "atc-AngThong3-m-h.jpg"

    results = model(imageURL, size=1280)

    df = results.pandas().xyxy[0]

    print(df)

    results = []
    all_scores = []
    img_width, img_height = get_image_size(imageURL)

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
                # 'from_name': self.from_name,
                # 'to_name': self.to_name,
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


if __name__ == "__main__":
    test_yolov5()

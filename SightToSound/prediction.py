
import sys
import json
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2


# 'content' is base-64-encoded image data.
def get_prediction(file_path):

    project_id = 'handsignreader'
    model_id = 'ICN4276277797351063552'

    with open(file_path, 'rb') as ff:
        content = ff.read()

    prediction_client = automl_v1beta1.PredictionServiceClient()

    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'image': {'image_bytes': content }}
    params = {}
    request = str(prediction_client.predict(name, payload, params))
    temp = request.split(" ")
    return temp[-1][1:2]  # waits till request is returned

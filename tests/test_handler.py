import pytest
from catflow_service_inference.worker import inference_handler, load_models
from catflow_service_inference.embedding import ImageFeatureExtractor
from catflow_service_inference.model import Model
from moto import mock_s3
from pathlib import Path
import boto3
import io
import os
from PIL import Image

S3_ENDPOINT_URL = "http://localhost:5002"
AWS_BUCKET_NAME = "catflow-test"
AWS_ACCESS_KEY_ID = "catflow-video-test-key"
AWS_SECRET_ACCESS_KEY = "catflow-video-secret-key"

FIXTURE_DIR = Path(__file__).parent.resolve() / "test_files"


class AsyncS3Wrapper:
    """Just fake it so I can still mock with moto

    The alternative is setting up a mock server and connecting to it, as in
    catflow-worker's tests"""

    def __init__(self):
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        self.client = session.client("s3", region_name="us-east-1")

    async def download_fileobj(self, Bucket=None, Key=None, Fileobj=None):
        buf = io.BytesIO()
        self.client.download_fileobj(Bucket, Key, buf)

        # Ensure we're reading from the start of the file
        buf.seek(0)
        data = buf.read()

        await Fileobj.write(data)

        return Fileobj


@pytest.fixture(scope="session")
def s3_client():
    # Get data file
    image = "tests/test_files/car.png"

    # Set up model
    model_name = "tests/test_files/yolov5n"  # load_model adds the .pt
    os.environ["CATFLOW_MODEL_NAME"] = model_name
    os.environ["CATFLOW_MODEL_THRESHOLD"] = "0.2"
    load_models()

    with mock_s3():
        s3 = AsyncS3Wrapper()

        # Push it to mock S3 so our worker can retrieve it
        s3.client.create_bucket(Bucket=AWS_BUCKET_NAME)
        with open(image, "rb") as f:
            s3.client.upload_fileobj(f, AWS_BUCKET_NAME, "test1.png")
        with open(image, "rb") as f:
            s3.client.upload_fileobj(f, AWS_BUCKET_NAME, "test2.png")

        yield s3


@pytest.mark.datafiles(
    FIXTURE_DIR / "car.png",
    FIXTURE_DIR / "yolov5n.pt",
)
def test_infer(datafiles):
    """Verify that infer produces the correct output"""
    img_path = datafiles / "car.png"
    assert img_path.is_file()

    model_name = str(datafiles / "yolov5n")  # load_model adds the .pt
    model = Model(model_name, 0.2)
    assert model.model_name == os.path.basename(model_name)

    pil_image = Image.open(img_path)
    predictions = model.predict(pil_image)
    assert len(predictions) == 1

    prediction = predictions[0]
    assert len(prediction) == 6  # x,y,w,h,conf,label
    assert prediction[5] == "car"


def is_valid_embedding(vector):
    if len(vector) != 512:
        return False
    for element in vector:
        if not isinstance(element, float):
            return False

    return True


@pytest.mark.datafiles(
    FIXTURE_DIR / "car.png",
)
def test_embed(datafiles):
    # Get data file
    image = str(next(datafiles.iterdir()))
    pil_image = Image.open(image)

    feature_extractor = ImageFeatureExtractor()
    vector = feature_extractor.get_vector(pil_image)
    assert is_valid_embedding(vector)


@pytest.mark.asyncio
async def test_worker2(s3_client, datafiles):
    # Test worker's behavior in the 'detect' pipeline
    status, responses = await inference_handler(
        ["test1.png", "test2.png"], "detect.rawframes", s3_client, AWS_BUCKET_NAME
    )

    # responses = [ ("response.key", [
    #   ("test1.png", modelname, [[x,y,w,h,c,l]]),
    #   ("test2.png", modelname, [[x,y,w,h,c,l]]),
    #   ] ) ]

    assert status is True
    assert len(responses) == 1
    routing_key, annotations = responses[0]
    assert routing_key == "detect.annotations"
    assert len(annotations) == 2

    assert annotations[0][0] == "test1.png"
    assert annotations[1][0] == "test2.png"

    for annotation in annotations:
        key, model_name, predictions = annotation
        assert len(predictions) == 1
        prediction = predictions[0]
        assert len(prediction) == 6  # x,y,w,h,conf,label
        assert prediction[5] == "car"
        assert model_name == "yolov5n"


@pytest.mark.asyncio
async def test_worker3(s3_client, datafiles):
    # Test worker's behavior in the 'ingest' pipeline (generate embeddings)
    status, responses = await inference_handler(
        ["test1.png"], "ingest.rawframes", s3_client, AWS_BUCKET_NAME
    )

    assert status is True
    assert len(responses) == 1
    routing_key, embeddings = responses[0]
    assert routing_key == "ingest.embeddings"
    assert len(embeddings) == 1

    key, vector = embeddings[0]
    assert key == "test1.png"
    assert is_valid_embedding(vector)


@pytest.mark.asyncio
async def test_worker1(s3_client, datafiles):
    # Test worker's behavior in the 'ingest' pipeline (generate annotations)
    status, responses = await inference_handler(
        ["test1.png"], "ingest.filteredframes", s3_client, AWS_BUCKET_NAME
    )

    assert status is True
    assert len(responses) == 1
    routing_key, annotations = responses[0]
    assert routing_key == "ingest.annotations"
    assert len(annotations) == 1

    key, model_name, predictions = annotations[0]
    assert key == "test1.png"
    assert len(predictions) == 1
    prediction = predictions[0]
    assert len(prediction) == 6  # x,y,w,h,conf,label
    assert prediction[5] == "car"
    assert model_name == "yolov5n"

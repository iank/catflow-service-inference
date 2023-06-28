import pytest
from catflow_service_inference.worker import inference_handler, load_models
from catflow_service_inference.embedding import ImageFeatureExtractor
from catflow_service_inference.model import Model
from catflow_worker.types import (
    RawFrame,
    RawFrameSchema,
    AnnotatedFrameSchema,
    EmbeddedFrameSchema,
    Prediction,
    VideoFile,
)
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
    assert isinstance(prediction, Prediction)
    assert prediction.label == "car"


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
async def test_worker_detect(s3_client, datafiles):
    # Test worker's behavior in the 'detect' pipeline

    # Expected input: list of RawFrame
    video = VideoFile(key="test.mp4")
    frames = [
        RawFrame(key="test1.png", source=video),
        RawFrame(key="test2.png", source=video),
    ]
    frames_msg = RawFrameSchema(many=True).dump(frames)

    # Expected output: list of AnnotatedFrame
    output_schema = AnnotatedFrameSchema(many=True)

    status, responses = await inference_handler(
        frames_msg, "detect.rawframes", s3_client, AWS_BUCKET_NAME
    )

    assert status is True
    assert len(responses) == 1

    routing_key, annotations_msg = responses[0]
    annotations = output_schema.load(annotations_msg)

    assert routing_key == "detect.annotatedframes"
    assert len(annotations) == 2

    assert annotations[0].key == "test1.png"
    assert annotations[1].key == "test2.png"

    for annotation in annotations:
        assert annotation.source.key == "test.mp4"
        assert annotation.model_name == "yolov5n"

        predictions = annotation.predictions
        assert len(predictions) == 1
        assert predictions[0].label == "car"


@pytest.mark.asyncio
async def test_worker_ingest(s3_client, datafiles):
    # Test worker's behavior in the 'ingest' pipeline (generate embeddings)

    # Expected input: list of RawFrame (only ever one at a time in this pipeline)
    video = VideoFile(key="test.mp4")
    frames = [
        RawFrame(key="test1.png", source=video),
    ]
    frames_msg = RawFrameSchema(many=True).dump(frames)

    # Expected output: list of EmbeddedFrame
    output_schema = EmbeddedFrameSchema(many=True)

    status, responses = await inference_handler(
        frames_msg, "ingest.rawframes", s3_client, AWS_BUCKET_NAME
    )

    assert status is True
    assert len(responses) == 1

    routing_key, embeddings_msg = responses[0]
    embeddings = output_schema.load(embeddings_msg)

    assert routing_key == "filter.embeddings"
    assert len(embeddings) == 1

    embeddedframe = embeddings[0]
    assert embeddedframe.key == "test1.png"
    assert embeddedframe.source.key == "test.mp4"
    assert is_valid_embedding(embeddedframe.embedding.vector)


@pytest.mark.asyncio
async def test_worker_filter(s3_client, datafiles):
    # Test worker's behavior in the 'ingest' pipeline (generate annotations)

    # Expected input: list of RawFrame (only ever one at a time in this pipeline)
    video = VideoFile(key="test.mp4")
    frames = [
        RawFrame(key="test1.png", source=video),
    ]
    frames_msg = RawFrameSchema(many=True).dump(frames)

    # Expected output: list of AnnotatedFrame
    output_schema = AnnotatedFrameSchema(many=True)

    status, responses = await inference_handler(
        frames_msg, "filter.rawframes", s3_client, AWS_BUCKET_NAME
    )

    assert status is True
    assert len(responses) == 1

    routing_key, annotations_msg = responses[0]
    annotations = output_schema.load(annotations_msg)

    assert routing_key == "ingest.annotatedframes"
    assert len(annotations) == 1

    annotation = annotations[0]
    assert annotation.key == "test1.png"
    assert annotation.source.key == "test.mp4"
    assert annotation.model_name == "yolov5n"

    predictions = annotation.predictions
    assert len(predictions) == 1
    assert predictions[0].label == "car"

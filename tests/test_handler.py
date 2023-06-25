import pytest
from catflow_service_inference.worker import inference_handler
from moto import mock_s3
from pathlib import Path
import boto3
import io

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

    async def download_fileobj(self, bucket, bey, f):
        buffer = io.BytesIO()
        self.client.download_fileobj(bucket, bey, buffer)

        # Ensure we're reading from the start of the file
        buffer.seek(0)
        data = buffer.read()

        await f.write(data)

        return f

    async def upload_fileobj(self, f, bucket, key):
        return self.client.upload_fileobj(f, bucket, key)


@pytest.fixture
def s3_client():
    with mock_s3():
        s3 = AsyncS3Wrapper()
        yield s3


@pytest.mark.datafiles(
    FIXTURE_DIR / "car.png",
)
def test_infer(datafiles):
    # Get data file
    # image = str(next(datafiles.iterdir()))

    # TODO
    pass


@pytest.mark.datafiles(
    FIXTURE_DIR / "car.png",
)
def test_embed(datafiles):
    # Get data file
    # image = str(next(datafiles.iterdir()))

    # TODO
    pass


@pytest.mark.datafiles(
    FIXTURE_DIR / "car.png",
)
@pytest.mark.asyncio
async def test_worker(s3_client, datafiles):
    # Get data file
    image = str(next(datafiles.iterdir()))

    # Push it to mock S3 so our worker can retrieve it
    s3_client.client.create_bucket(Bucket=AWS_BUCKET_NAME)
    with open(image, "rb") as f:
        await s3_client.upload_fileobj(f, AWS_BUCKET_NAME, "test1.png")

    # Test worker's behavior in the 'detect' pipeline
    status, responses = await inference_handler(
        ["test1.png"], "detect.rawframes", s3_client, AWS_BUCKET_NAME
    )

    # TODO

    # Test worker's behavior in the 'ingest' pipeline (generate embeddings)
    status, responses = await inference_handler(
        ["test1.png"], "ingest.rawframes", s3_client, AWS_BUCKET_NAME
    )

    # TODO

    # Test worker's behavior in the 'ingest' pipeline (generate annotations)
    status, responses = await inference_handler(
        ["test1.png"], "ingest.filteredframes", s3_client, AWS_BUCKET_NAME
    )

    # TODO

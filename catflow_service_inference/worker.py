from typing import Any, List, Tuple
import signal
import asyncio
from catflow_worker import Worker
from catflow_worker.types import (
    RawFrameSchema,
    AnnotatedFrameSchema,
    AnnotatedFrame,
    EmbeddedFrameSchema,
    EmbeddedFrame,
    Embedding,
)
from .embedding import ImageFeatureExtractor
from .model import Model
from PIL import Image
import io
import os
import aiofiles

import logging

FEAT = None
MODEL = None


def load_models():
    global MODEL
    global FEAT
    model_name = os.environ["CATFLOW_MODEL_NAME"]
    threshold = float(os.environ["CATFLOW_MODEL_THRESHOLD"])

    logging.info("Loading model {model_name} (threshold={threshold})")
    MODEL = Model(model_name, threshold)

    logging.info("Loading embedding model")
    FEAT = ImageFeatureExtractor()


async def inference_handler(
    msg: Any, key: str, s3: Any, bucket: str
) -> Tuple[bool, List[Tuple[str, Any]]]:
    """Run inference on the given frames

    ingest.rawframes: Generate embeddings
    filter.rawframes: Generate annotations
    detect.rawframes: Generate annotations"""
    global MODEL
    global FEAT
    logging.info(f"[*] Message received ({key})")

    # Routing
    pipeline, datatype = key.split(".")
    assert datatype == "rawframes"

    if pipeline == "ingest":
        pipeline_out = "filter"
        datatype_out = "embeddings"
    elif pipeline == "filter":
        pipeline_out = "ingest"
        datatype_out = "annotatedframes"
    elif pipeline == "detect":
        pipeline_out = "detect"
        datatype_out = "annotatedframes"
    else:
        raise ValueError(f"Unexpected message from {key}")

    logging.debug(f"Will generate {datatype_out} to {pipeline_out}")

    # Load message
    msg_objs = RawFrameSchema(many=True).load(msg)

    # Download from S3 and open PIL image
    responseobjects = []
    for rawframe in msg_objs:
        logging.debug(f"Downloading {rawframe.key} from {bucket}")

        buf = io.BytesIO()
        async with aiofiles.tempfile.NamedTemporaryFile("wb+") as f:
            await s3.download_fileobj(bucket, rawframe.key, f)
            await f.seek(0)
            buf.write(await f.read())

        buf.seek(0)
        image = Image.open(buf)

        if datatype_out == "annotatedframes":
            predictions = MODEL.predict(image)
            annotatedframe = AnnotatedFrame(
                key=rawframe.key,
                source=rawframe.source,
                model_name=MODEL.model_name,
                predictions=predictions,
            )

            responseobjects.append(annotatedframe)
        elif datatype_out == "embeddings":
            vector = FEAT.get_vector(image)
            embeddedframe = EmbeddedFrame(
                key=rawframe.key,
                source=rawframe.source,
                embedding=Embedding(vector=vector),
            )

            responseobjects.append(embeddedframe)

    # Dump response
    if datatype_out == "annotatedframes":
        schema_out = AnnotatedFrameSchema(many=True)
    elif datatype_out == "embeddings":
        schema_out = EmbeddedFrameSchema(many=True)

    responseobjects_msg = schema_out.dump(responseobjects)

    # Always do one message- videosplit service handled batching
    logging.info(
        f"[-] {len(responseobjects)} objects -> {pipeline_out}.{datatype_out} (1 msg)"
    )
    return True, [(f"{pipeline_out}.{datatype_out}", responseobjects_msg)]


async def shutdown(worker, task):
    await worker.shutdown()
    task.cancel()
    try:
        await task
    except asyncio.exceptions.CancelledError:
        pass


async def startup(queue: str, topic_key: str):
    worker = await Worker.create(inference_handler, queue, topic_key)
    task = asyncio.create_task(worker.work())

    def handle_sigint(sig, frame):
        print("^ SIGINT received, shutting down...")
        asyncio.create_task(shutdown(worker, task))

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        if not await task:
            print("[!] Exited with error")
            return False
    except asyncio.exceptions.CancelledError:
        return True


def main() -> bool:
    topic_key = "*.rawframes"
    queue_name = "catflow-service-inference"
    logging.basicConfig(level=logging.INFO)

    load_models()

    return asyncio.run(startup(queue_name, topic_key))

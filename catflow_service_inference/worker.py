from typing import Any, List, Tuple
import signal
import asyncio
from catflow_worker import Worker
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
) -> Tuple[bool, List[Tuple[str, Tuple[str, List[float]]]]]:
    """Run inference on the given frames

    ingest.rawframes: Generate embeddings
    ingest.filteredframes: Generate annotations

    detect.rawframes: Generate annotations"""
    global MODEL
    global FEAT
    logging.info(f"[*] Message received ({key})")

    pipeline, _ = key.split(".")
    if pipeline == "filter":
        responsekey = "ingest.annotatedframes"
        action = "annotate"
        logging.debug(f"Will generate annotations to {responsekey}")
    elif pipeline == "detect":
        responsekey = "detect.annotatedframes"
        action = "annotate"
        logging.debug(f"Will generate annotations to {responsekey}")
    elif pipeline == "ingest":
        responsekey = "filter.embeddings"
        action = "embed"
        logging.debug(f"Will generate embeddings to {responsekey}")
    else:
        raise ValueError(f"Unexpected message from {key}")

    responseobjects = []

    # Download from S3 and open PIL image
    for s3key in msg:
        logging.debug(f"Downloading {s3key} from {bucket}")

        buf = io.BytesIO()
        async with aiofiles.tempfile.NamedTemporaryFile("wb+") as f:
            await s3.download_fileobj(bucket, s3key, f)
            await f.seek(0)
            buf.write(await f.read())

        buf.seek(0)
        image = Image.open(buf)

        if action == "annotate":
            predictions = MODEL.predict(image)
            annotation = (s3key, MODEL.model_name, predictions)
            responseobjects.append(annotation)
        elif action == "embed":
            embedding = FEAT.get_vector(image)
            responseobjects.append((s3key, embedding))

    logging.info(
        f"[-] {action}: {len(responseobjects)} objects -> {responsekey} (1 msg)"
    )
    return True, [(responsekey, responseobjects)]


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

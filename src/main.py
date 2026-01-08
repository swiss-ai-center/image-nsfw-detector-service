import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
import os
import io
import json
from PIL import Image
import numpy as np
import tensorflow as tf

# Some constants for this service
SUB_CAT_NAMES = ['nsfw_cartoon', 'nsfw_nudity', 'nsfw_porn', 'nsfw_suggestive',
                 'safe_cartoon', 'safe_general', 'safe_person']
CAT_NAMES = ['nsfw', 'safe']
IMG_SIZE = 224
CHANNELS = 3
N_CLASSES = len(SUB_CAT_NAMES)
WEIGHT_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'model/TL_MNV2_finetune_224_B32_AD1E10-5_NSFW-V2.1_DA2.hdf5')

settings = get_settings()


class MyService(Service):
    """
    Not Safe For Work (NSFW) image classification service.
    Caution: the current version of the service is able to detect nudity, sexual and hentai content.
    It is not able to detect profanity and violence for now.
    """

    # Any additional fields must be excluded for Pydantic to work
    _base_model: object
    _nsfw_model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="NSFW Image Detection",
            slug="nsfw-image-detection",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="image", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.APPLICATION_JSON]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_RECOGNITION,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/image-nsfw-detector/",
        )
        self._logger = get_logger(settings)

        # Load the base model
        self._logger.info("Loading the base model...")
        self._base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))

        self._logger.info("Base model loaded. Recreating structure of model before loading fine-tuned weights...")

        # Create model using functional API or by adding base_model layers directly
        self._nsfw_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
            self._base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(N_CLASSES),
            tf.keras.layers.Activation('softmax')
        ], name='MNV2')

        # Build the model before loading weights
        self._nsfw_model.build((None, IMG_SIZE, IMG_SIZE, CHANNELS))

        self._logger.info('Loading weights from file: {}'.format(WEIGHT_FILE))
        self._nsfw_model.load_weights(WEIGHT_FILE)
        self._logger.info('Weights loaded.')

    def build_score_dict(self, scores, class_names):
        """
        Build a dictionary of scores from a numpy array of float (scores) and a list of class names.
        :param scores: the numpy array of scores
        :param class_names: the list of class names to be associated to the scores
        :return: a dictionary of scores in the form of {'nsfw_cartoon': 0.1, 'nsfw_nudity': 0.6, ...}
        """
        score_dict = {}
        for i, score in enumerate(scores):
            score_dict[class_names[i]] = str(score)
        return score_dict

    def predict_from_image(self, image_tensor):
        """
        Compute the predicted classes from an image tensor by calling the model.predict()
        on that tensor. The method decides on the winning main-category by summing the
        scores on the range of sub-category scores. Then it takes the arg max to elect the
        winner of the categories and sub-categories.
        :param image_tensor: the tensor image from which to predict
        :return: a tuple with the winner category, the winner sub-category, the list of
        category scores and the list of sub-category scores
        """
        image_tensor = np.array([image_tensor])
        self._logger.info("Image tensor shape: {}".format(image_tensor.shape))
        pred_sub_cat = self._nsfw_model.predict(image_tensor, verbose=0)
        self._logger.info("Prediction shape: {}".format(pred_sub_cat.shape))
        self._logger.info("Prediction: {}".format(pred_sub_cat))
        pred_cat = np.zeros((1, 2))
        pred_cat[:, 0] = np.sum(pred_sub_cat[:, :4], axis=1)  # do the sum of nsfw sub-categories to compute nsfw pred
        pred_cat[:, 1] = np.sum(pred_sub_cat[:, 4:], axis=1)  # same thing for safe
        # in the end, the pred_cat is a similar output tensor as pred_sub_cat but on 2 main categories nsfw and safe
        # let's use the first prediction for now (disregarding the fliped image)
        scores_dict_sub_cat = self.build_score_dict(pred_sub_cat[0], SUB_CAT_NAMES)
        self._logger.info("Scores sub-cat: {}".format(scores_dict_sub_cat))
        scores_dict_cat = self.build_score_dict(pred_cat[0], CAT_NAMES)
        self._logger.info("Scores cat: {}".format(scores_dict_cat))
        winner_sub_cat = pred_sub_cat.argmax(axis=1)[0]
        winner_cat = pred_cat.argmax(axis=1)[0]
        # get the prediction as category and subcategory
        prediction_subcategory = SUB_CAT_NAMES[winner_sub_cat]
        prediction_category = CAT_NAMES[winner_cat]
        return prediction_category, prediction_subcategory, scores_dict_cat, scores_dict_sub_cat

    def process(self, data):
        # NOTE that the data is a dictionary with the keys being the field names set in the data_in_fields
        raw = data["image"].data
        buff = io.BytesIO(raw)
        image = Image.open(buff)
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        image_tensor = np.array(image)
        self._logger.info("Image shape: {}".format(image_tensor.shape))
        image_tensor = tf.keras.applications.mobilenet.preprocess_input(image_tensor)
        self._logger.info("Image shape after preprocessing: {}".format(image_tensor.shape))
        prediction_category, prediction_subcategory, scores_dict_cat, scores_dict_sub_cat = \
            self.predict_from_image(image_tensor)

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(
                data=json.dumps({'prediction_category': prediction_category,
                                 'prediction_subcategory': prediction_subcategory,
                                 'scores_dict_cat': scores_dict_cat,
                                 'scores_dict_sub_cat': scores_dict_sub_cat}),
                type=FieldDescriptionType.APPLICATION_JSON
            )
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(f"Aborting service announcement after "
                                       f"{settings.engine_announce_retries} retries")

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """
Detects between two main categories : 'nsfw' and 'safe', and detects the following sub-categories:
'nsfw_cartoon', 'nsfw_nudity', 'nsfw_porn', 'nsfw_suggestive', 'safe_cartoon', 'safe_general', 'safe_person'
"""
api_summary = """
This service detects nudity, sexual and hentai content in images, or if the image is 'safe for work'.
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="NSFW Image Detection API.",
    description=api_description,
    version="1.0.0",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=['Service'])
app.include_router(tasks_router, tags=['Tasks'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)

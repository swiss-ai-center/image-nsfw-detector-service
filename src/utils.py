import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI

from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.service import TasksService
from common_code.service.models import Service


# Import the service defined in the other file
from my_service import MyService

settings = get_settings()
# Global variable
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

    async def run_heartbeat(my_service: Service, interval: int = 30):
        # Get interval from settings or default to 30 seconds
        interval = getattr(settings, 'heartbeat_interval', interval)

        while True:
            # Wait for the defined interval before sending the next ping
            await asyncio.sleep(interval)

            for engine_url in settings.engine_urls:
                try:
                    await service_service.heartbeat(engine_url, my_service)
                except Exception as e:
                    logger.warning(f"Failed to send heartbeat to {engine_url}: {e}")

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    # Start the heartbeat task in the background
    heartbeat_task = asyncio.create_task(run_heartbeat(my_service, 20))
    yield
    heartbeat_task.cancel()
    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)

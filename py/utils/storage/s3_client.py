from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except ImportError as exc:  # pragma: no cover
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore
    _BOTO_IMPORT_ERROR = exc
else:
    _BOTO_IMPORT_ERROR = None

from py.config import settings
from py.utils.logging import get_logger

logger = get_logger(__name__, category="video")


class S3StorageNotConfigured(RuntimeError):
    """Raised when S3 storage is requested but not configured."""


class S3UploadError(RuntimeError):
    """Raised when an S3 upload fails."""


@dataclass
class FrameUploadResult:
    bucket: str
    key: str
    url: str


def _normalise_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ""
    prefix = prefix.strip("/")
    return f"{prefix}/" if prefix else ""


class FrameStorageClient:
    """Helper for persisting frames to S3 and generating presigned URLs."""

    def __init__(self) -> None:
        if not settings.video_frame_bucket:
            raise S3StorageNotConfigured(
                "VIDEO_FRAME_BUCKET must be set to enable S3 frame storage."
            )

        if boto3 is None:
            raise RuntimeError(
                "boto3 is required for S3 frame storage. Install the optional dependency."
            ) from _BOTO_IMPORT_ERROR

        session_kwargs = {}
        if settings.aws_profile:
            session_kwargs["profile_name"] = settings.aws_profile
        if settings.aws_region:
            session_kwargs["region_name"] = settings.aws_region

        session = boto3.session.Session(**session_kwargs)
        self._client = session.client("s3")
        self._bucket = settings.video_frame_bucket
        self._prefix = _normalise_prefix(settings.video_frame_prefix)
        self._url_ttl = max(int(settings.video_frame_url_ttl_seconds or 600), 60)

    @property
    def bucket(self) -> str:
        return self._bucket

    def build_object_key(
        self, *, channel_id: str, captured_at: datetime, source_path: str
    ) -> str:
        timestamp = captured_at.strftime("%Y/%m/%d/%H%M%S%f")
        filename = Path(source_path).name
        return f"{self._prefix}{channel_id}/{timestamp}_{filename}"

    def upload_frame(
        self,
        *,
        image_path: str,
        channel_id: str,
        captured_at: datetime,
        content_type: Optional[str] = None,
    ) -> FrameUploadResult:
        object_key = self.build_object_key(
            channel_id=channel_id, captured_at=captured_at, source_path=image_path
        )
        content_type = (
            content_type or mimetypes.guess_type(image_path)[0] or "image/jpeg"
        )

        extra_args = {"ContentType": content_type}

        try:
            self._client.upload_file(
                image_path,
                self._bucket,
                object_key,
                ExtraArgs=extra_args,
            )
        except (ClientError, BotoCoreError) as exc:
            logger.error(
                "Failed to upload frame to S3 (%s/%s): %s",
                self._bucket,
                object_key,
                exc,
            )
            raise S3UploadError(str(exc)) from exc

        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": object_key},
                ExpiresIn=self._url_ttl,
            )
        except (ClientError, BotoCoreError) as exc:
            logger.error(
                "Failed to generate presigned URL for %s/%s: %s",
                self._bucket,
                object_key,
                exc,
            )
            raise S3UploadError(str(exc)) from exc

        logger.debug("Uploaded frame to s3://%s/%s", self._bucket, object_key)
        return FrameUploadResult(bucket=self._bucket, key=object_key, url=url)


_frame_storage_client: Optional[FrameStorageClient] = None


def is_enabled() -> bool:
    return bool(settings.video_frame_bucket)


def get_frame_storage_client() -> FrameStorageClient:
    global _frame_storage_client
    if _frame_storage_client is None:
        _frame_storage_client = FrameStorageClient()
    return _frame_storage_client


def upload_frame_to_s3(
    *,
    image_path: str,
    channel_id: str,
    captured_at: datetime,
) -> FrameUploadResult:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Frame not found: {image_path}")

    client = get_frame_storage_client()
    return client.upload_frame(
        image_path=image_path,
        channel_id=channel_id,
        captured_at=captured_at,
    )

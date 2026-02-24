"""
MinerU API Client
Implements the official MinerU API for document parsing with support for
single/batch file processing and result retrieval.
"""

import requests
import time
import hashlib
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ModelVersion(Enum):
    """Supported model versions"""
    PIPELINE = "pipeline"
    VLM = "vlm"
    MINERU_HTML = "MinerU-HTML"


class TaskState(Enum):
    """Task processing states"""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CONVERTING = "converting"
    WAITING_FILE = "waiting-file"


@dataclass
class ExtractProgress:
    """Extraction progress information"""
    extracted_pages: int
    total_pages: int
    start_time: str


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    state: TaskState
    data_id: Optional[str] = None
    full_zip_url: Optional[str] = None
    err_msg: Optional[str] = None
    extract_progress: Optional[ExtractProgress] = None
    file_name: Optional[str] = None


class MinerUClient:
    """
    Client for MinerU document parsing API.

    Supports:
    - Single file parsing via URL
    - Batch file parsing via URL
    - Batch file upload
    - Task result polling
    - Bidirectional coordinate mapping
    """

    def __init__(self, api_token: str, base_url: str = "https://mineru.net/api/v4"):
        """
        Initialize the MinerU client.

        Args:
            api_token: Your MinerU API token (get from https://mineru.net/apiManage/token)
            base_url: Base URL for the API (default: https://mineru.net/api/v4)
        """
        self.api_token = api_token
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        })

    def create_task_from_url(
        self,
        url: str,
        model_version: ModelVersion = ModelVersion.PIPELINE,
        is_ocr: bool = False,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "ch",
        data_id: Optional[str] = None,
        callback: Optional[str] = None,
        seed: Optional[str] = None,
        extra_formats: Optional[List[str]] = None,
        page_ranges: Optional[str] = None
    ) -> str:
        """
        Create a parsing task from a file URL.

        Args:
            url: URL of the file to parse (supports pdf, doc, ppt, images, html)
            model_version: Model version to use (pipeline, vlm, or MinerU-HTML)
            is_ocr: Enable OCR for pipeline model (default: False)
            enable_formula: Enable formula recognition (default: True)
            enable_table: Enable table recognition (default: True)
            language: Document language code (default: 'ch')
            data_id: Custom data ID for tracking
            callback: Callback URL for notification when task completes
            seed: Random string for callback signature verification
            extra_formats: Additional export formats (docx, html, latex)
            page_ranges: Page range to parse (e.g., "1-10", "2,4-6")

        Returns:
            task_id: ID of the created task

        Raises:
            ValueError: If the API request fails
        """
        endpoint = f"{self.base_url}/extract/task"
        data = {
            "url": url,
            "model_version": model_version.value,
            "is_ocr": is_ocr,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language
        }

        # Add optional parameters
        if data_id:
            data["data_id"] = data_id
        if callback:
            if not seed:
                raise ValueError("seed is required when using callback")
            data["callback"] = callback
            data["seed"] = seed
        if extra_formats:
            data["extra_formats"] = extra_formats
        if page_ranges:
            data["page_ranges"] = page_ranges

        response = self.session.post(endpoint, json=data)
        result = response.json()

        if result.get("code") != 0:
            raise ValueError(f"API Error: {result.get('msg', 'Unknown error')}")

        return result["data"]["task_id"]

    def get_task_result(self, task_id: str) -> TaskResult:
        """
        Get the result of a parsing task.

        Args:
            task_id: ID of the task to query

        Returns:
            TaskResult: Task result information

        Raises:
            ValueError: If the API request fails
        """
        endpoint = f"{self.base_url}/extract/task/{task_id}"
        response = self.session.get(endpoint)
        result = response.json()

        if result.get("code") != 0:
            raise ValueError(f"API Error: {result.get('msg', 'Unknown error')}")

        data = result["data"]

        # Parse extract progress if present
        extract_progress = None
        if "extract_progress" in data:
            ep = data["extract_progress"]
            extract_progress = ExtractProgress(
                extracted_pages=ep["extracted_pages"],
                total_pages=ep["total_pages"],
                start_time=ep["start_time"]
            )

        return TaskResult(
            task_id=data["task_id"],
            state=TaskState(data["state"]),
            data_id=data.get("data_id"),
            full_zip_url=data.get("full_zip_url"),
            err_msg=data.get("err_msg"),
            extract_progress=extract_progress
        )

    def wait_for_completion(
        self,
        task_id: str,
        check_interval: int = 5,
        timeout: int = 600
    ) -> TaskResult:
        """
        Wait for a task to complete.

        Args:
            task_id: ID of the task to wait for
            check_interval: Seconds between status checks (default: 5)
            timeout: Maximum seconds to wait (default: 600)

        Returns:
            TaskResult: Final task result

        Raises:
            TimeoutError: If task doesn't complete within timeout
        """
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

            result = self.get_task_result(task_id)

            if result.state == TaskState.DONE:
                return result
            elif result.state == TaskState.FAILED:
                return result
            elif result.state in [TaskState.PENDING, TaskState.RUNNING, TaskState.CONVERTING]:
                if result.extract_progress:
                    print(f"Task {task_id}: {result.state.value} - "
                          f"{result.extract_progress.extracted_pages}/"
                          f"{result.extract_progress.total_pages} pages")
                time.sleep(check_interval)
            else:
                time.sleep(check_interval)

    def create_batch_task_from_urls(
        self,
        files: List[Dict[str, str]],
        model_version: ModelVersion = ModelVersion.PIPELINE,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "ch",
        callback: Optional[str] = None,
        seed: Optional[str] = None,
        extra_formats: Optional[List[str]] = None
    ) -> str:
        """
        Create batch parsing tasks from file URLs.

        Args:
            files: List of file dicts with 'url' and optional 'data_id', 'page_ranges', 'is_ocr'
            model_version: Model version to use
            enable_formula: Enable formula recognition
            enable_table: Enable table recognition
            language: Document language code
            callback: Callback URL for notification
            seed: Random string for callback signature
            extra_formats: Additional export formats

        Returns:
            batch_id: ID of the batch task
        """
        endpoint = f"{self.base_url}/extract/task/batch"
        data = {
            "files": files,
            "model_version": model_version.value,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language
        }

        if callback:
            if not seed:
                raise ValueError("seed is required when using callback")
            data["callback"] = callback
            data["seed"] = seed
        if extra_formats:
            data["extra_formats"] = extra_formats

        response = self.session.post(endpoint, json=data)
        result = response.json()

        if result.get("code") != 0:
            raise ValueError(f"API Error: {result.get('msg', 'Unknown error')}")

        return result["data"]["batch_id"]

    def get_batch_results(self, batch_id: str) -> Dict[str, Any]:
        """
        Get results of a batch task.

        Args:
            batch_id: ID of the batch task

        Returns:
            Dict with batch results containing extract_result list
        """
        endpoint = f"{self.base_url}/extract-results/batch/{batch_id}"
        response = self.session.get(endpoint)
        result = response.json()

        if result.get("code") != 0:
            raise ValueError(f"API Error: {result.get('msg', 'Unknown error')}")

        return result["data"]

    def request_batch_file_upload(
        self,
        files: List[Dict[str, str]],
        model_version: ModelVersion = ModelVersion.PIPELINE,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "ch",
        callback: Optional[str] = None,
        seed: Optional[str] = None,
        extra_formats: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Request upload URLs for batch file upload.

        Args:
            files: List of file dicts with 'name' and optional 'data_id', 'page_ranges', 'is_ocr'
            model_version: Model version to use
            enable_formula: Enable formula recognition
            enable_table: Enable table recognition
            language: Document language code
            callback: Callback URL for notification
            seed: Random string for callback signature
            extra_formats: Additional export formats

        Returns:
            Dict with batch_id and file_urls list
        """
        endpoint = f"{self.base_url}/file-urls/batch"
        data = {
            "files": files,
            "model_version": model_version.value,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language
        }

        if callback:
            if not seed:
                raise ValueError("seed is required when using callback")
            data["callback"] = callback
            data["seed"] = seed
        if extra_formats:
            data["extra_formats"] = extra_formats

        response = self.session.post(endpoint, json=data)
        result = response.json()

        if result.get("code") != 0:
            raise ValueError(f"API Error: {result.get('msg', 'Unknown error')}")

        return result["data"]

    def upload_file(self, upload_url: str, file_path: str) -> bool:
        """
        Upload a file to the provided upload URL.

        Args:
            upload_url: Presigned upload URL
            file_path: Local path to the file

        Returns:
            bool: True if upload successful
        """
        with open(file_path, 'rb') as f:
            # Don't set Content-Type for upload
            headers = {k: v for k, v in self.session.headers.items() if k.lower() != 'content-type'}
            response = requests.put(upload_url, data=f, headers=headers)
            return response.status_code == 200

    def verify_callback_checksum(self, uid: str, seed: str, content: str, received_checksum: str) -> bool:
        """
        Verify callback signature for security.

        Args:
            uid: User ID from personal center
            seed: Random string used when creating task
            content: JSON content string from callback
            received_checksum: Checksum received in callback

        Returns:
            bool: True if checksum is valid
        """
        message = f"{uid}{seed}{content}"
        calculated_checksum = hashlib.sha256(message.encode()).hexdigest()
        return calculated_checksum == received_checksum


if __name__ == "__main__":
    # Example usage
    import os

    # Get API token from environment variable
    api_token = os.getenv("MINERU_API_TOKEN")
    if not api_token:
        print("Please set MINERU_API_TOKEN environment variable")
        exit(1)

    # Initialize client
    client = MinerUClient(api_token)

    # Example 1: Parse from URL
    print("Creating task from URL...")
    task_id = client.create_task_from_url(
        url="https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
        model_version=ModelVersion.VLM
    )
    print(f"Task created: {task_id}")

    # Wait for completion
    print("Waiting for completion...")
    result = client.wait_for_completion(task_id)
    print(f"Task state: {result.state.value}")
    if result.full_zip_url:
        print(f"Download URL: {result.full_zip_url}")

    # Example 2: Upload local file
    # print("\nRequesting upload URL...")
    # upload_data = client.request_batch_file_upload(
    #     files=[{"name": "demo.pdf", "data_id": "local-file-1"}],
    #     model_version=ModelVersion.VLM
    # )
    # print(f"Batch ID: {upload_data['batch_id']}")
    # print(f"Upload URL: {upload_data['file_urls'][0]}")
    #
    # # Upload the file
    # success = client.upload_file(upload_data['file_urls'][0], "demo.pdf")
    # print(f"Upload {'successful' if success else 'failed'}")

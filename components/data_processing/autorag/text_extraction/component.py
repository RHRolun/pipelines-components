from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=["docling[ort]"],
)
def text_extraction(
    sampled_documents_descriptor: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Output[dsl.Artifact],
):
    """Text Extraction component.

    Reads the sampled_documents_descriptor YAML (from documents_sampling), fetches
    the listed documents from S3, and extracts text using the docling library.

    Args:
        sampled_documents_descriptor: Input artifact containing
            sampled_documents_descriptor.yaml with bucket, prefix, and documents list.
        extracted_text: Output artifact where the extracted text content will be stored.
    """
    import json
    import logging
    import os
    import sys
    import tempfile
    import time
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from pathlib import Path

    import boto3
    import yaml
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    SAMPLED_DOCUMENTS_DESCRIPTOR_FILENAME = "sampled_documents_descriptor.yaml"
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".html", ".txt"}
    DOWNLOAD_MAX_WORKERS = 8

    logger = logging.getLogger("Text Extraction component logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

    descriptor_root = Path(sampled_documents_descriptor.path)
    if descriptor_root.is_dir():
        descriptor_path = descriptor_root / SAMPLED_DOCUMENTS_DESCRIPTOR_FILENAME
    else:
        descriptor_path = descriptor_root

    if not descriptor_path.exists():
        raise FileNotFoundError(f"Descriptor not found: {descriptor_path}")

    with open(descriptor_path) as f:
        descriptor = yaml.safe_load(f)

    bucket = descriptor["bucket"]
    documents = descriptor["documents"]

    s3_creds = {k: os.environ.get(k) for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT"]}
    for k, v in s3_creds.items():
        if v is None:
            raise ValueError(f"{k} environment variable not set. Check if kubernetes secret was configured properly.")

    s3_creds["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION", "")

    session = boto3.session.Session(
        aws_access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
        region_name=s3_creds.get("AWS_DEFAULT_REGION"),
    )
    s3_client = session.client(
        service_name="s3",
        endpoint_url=s3_creds["AWS_S3_ENDPOINT"],
    )

    def download_document(doc: dict) -> bool:
        key = doc["key"]
        local_name = doc.get("output_basename") or key
        local_path = download_path / local_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Downloading %s", key)
            s3_client.download_file(bucket, key, str(local_path))
            return True
        except Exception as e:
            logger.error("Failed to fetch %s: %s", key, e)
            raise

    def process_document(file_path_str: str, output_dir_str: str) -> bool:
        try:
            path = Path(file_path_str)
            out_dir = Path(output_dir_str)
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = False
            pipeline_options.accelerator_options = AcceleratorOptions(device="cpu", num_threads=1)
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
            result = converter.convert(path)
            markdown_content = result.document.export_to_markdown()
            output_file = out_dir / f"{path.name}.md"
            output_file.write_text(markdown_content, encoding="utf-8")
            return True
        except Exception as e:
            logger.error("Failed to process %s: %s", file_path_str, e)
            return False

    EXTRACTION_METRICS_FILENAME = "extraction_metrics.json"

    with tempfile.TemporaryDirectory() as download_dir:
        download_path = Path(download_dir)
        output_dir = Path(extracted_text.path)
        output_dir.mkdir(parents=True, exist_ok=True)

        download_workers = min(DOWNLOAD_MAX_WORKERS, len(documents)) if documents else 1
        download_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            list(executor.map(download_document, documents))
        download_seconds = time.perf_counter() - download_start
        logger.info("Documents download completed in %.2f seconds (%d documents)", download_seconds, len(documents))

        files_to_process = [
            f for f in download_path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        # Phase 2: text extraction (timed)
        logger.info("Starting text extraction for %d documents.", len(files_to_process))
        process_workers = min(os.cpu_count() or 1, len(files_to_process)) if files_to_process else 1
        worker_fn = partial(process_document, output_dir_str=str(output_dir))
        extraction_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=process_workers) as executor:
            results = list(executor.map(worker_fn, [str(f) for f in files_to_process]))
        extraction_seconds = time.perf_counter() - extraction_start
        logger.info("Text extraction completed in %.2f seconds", extraction_seconds)

    processed_count = sum(1 for r in results if r)
    error_count = len(results) - processed_count

    metrics = {
        "download_seconds": round(download_seconds, 3),
        "extraction_seconds": round(extraction_seconds, 3),
        "total_seconds": round(download_seconds + extraction_seconds, 3),
        "document_count": len(documents),
        "processed_count": processed_count,
        "error_count": error_count,
    }
    metrics_path = output_dir / EXTRACTION_METRICS_FILENAME
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s: %s", metrics_path, metrics)

    summary = f"Text extraction completed. Total processed: {processed_count}, Errors: {error_count}."
    logger.info(summary)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        text_extraction,
        package_path=__file__.replace(".py", "_component.yaml"),
    )

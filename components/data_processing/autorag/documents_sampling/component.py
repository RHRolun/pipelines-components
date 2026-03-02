from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=["boto3"],
)
def documents_sampling(
    input_data_bucket_name: str,
    input_data_path: str,
    test_data: dsl.Input[dsl.Artifact] = None,
    sampling_config: dict = None,
    sampled_documents: dsl.Output[dsl.Artifact] = None,
):
    """Documents sampling component.

    Lists available documents list from S3, applies sampling, and writes a YAML manifest
    (sampled_documents_descriptor.yaml) with metadata. Does not download document contents.

    Args:
        input_data_bucket_name: S3 (or compatible) bucket containing input data.
        input_data_path: Path to folder with input documents within the bucket.
        test_data: Optional input artifact containing test data for sampling.
        sampling_config: Optional sampling configuration. May include: max_size_gigabytes (int, default 1);
            target_count (int) for benchmarking: sample until this many documents (duplicates allowed);
            target_size_bytes (int) for benchmarking: sample until total size reaches this (duplicates allowed).
        sampled_documents: Output artifact containing the sampled documents descriptor yaml file.

    Environment variables (required when run with pipeline secret injection):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION.
    """
    import json
    import logging
    import os
    import sys
    from itertools import cycle
    from pathlib import Path

    import boto3
    import yaml

    SAMPLED_DOCUMENTS_DESCRIPTOR_FILENAME = "sampled_documents_descriptor.yaml"
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".html", ".txt"}

    if sampling_config is None:
        sampling_config = {}
    MAX_SIZE_BYTES = 1024**3 * int(sampling_config.get("max_size_gigabytes", 1))

    logger = logging.getLogger("Document Loader component logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    def get_test_data_docs_names() -> list[str]:
        if test_data is None:
            return []
        with open(test_data.path, "r") as f:
            benchmark = json.load(f)

        docs_names = []
        for question in benchmark:
            docs_names.extend(question["correct_answer_document_ids"])

        return docs_names

    def build_and_write_descriptor():
        """Validate S3 credentials, list objects, sample, and write YAML descriptor."""
        s3_creds = {
            k: os.environ.get(k)
            for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT", "AWS_DEFAULT_REGION"]
        }
        for k, v in s3_creds.items():
            if v is None:
                raise ValueError(
                    "%s environment variable not set. Check if kubernetes secret was configured properly" % k
                )

        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_creds["AWS_S3_ENDPOINT"],
            region_name=s3_creds["AWS_DEFAULT_REGION"],
            aws_access_key_id=s3_creds["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=s3_creds["AWS_SECRET_ACCESS_KEY"],
        )

        contents = s3_client.list_objects_v2(
            Bucket=input_data_bucket_name,
            Prefix=input_data_path,
        ).get("Contents", [])

        supported_files = [c for c in contents if c["Key"].endswith(tuple(SUPPORTED_EXTENSIONS))]
        if not supported_files:
            raise Exception("No supported documents found.")

        test_data_docs_names = get_test_data_docs_names()
        supported_files.sort(key=lambda c: c["Key"] not in test_data_docs_names)

        target_count = sampling_config.get("target_count")
        target_size_bytes = sampling_config.get("target_size_bytes")

        if target_count is not None or target_size_bytes is not None:
            selected = []
            total_size = 0
            target_count_val = target_count if target_count is not None else float("inf")
            target_size_val = target_size_bytes if target_size_bytes is not None else float("inf")
            cycled = cycle(supported_files)
            idx = 0
            consecutive_skips = 0
            while consecutive_skips < len(supported_files):
                if len(selected) >= target_count_val or total_size >= target_size_val:
                    break
                file_info = next(cycled)
                if total_size + file_info["Size"] > MAX_SIZE_BYTES:
                    consecutive_skips += 1
                    continue
                consecutive_skips = 0
                selected.append((file_info, idx))
                total_size += file_info["Size"]
                idx += 1

            documents = []
            for file_info, idx in selected:
                key = file_info["Key"]
                size_bytes = file_info["Size"]
                stem = Path(key).stem
                suffix = Path(key).suffix
                output_basename = f"{stem}_{idx}{suffix}"
                documents.append(
                    {
                        "key": key,
                        "size_bytes": size_bytes,
                        "output_basename": output_basename,
                    }
                )
        else:
            total_size = 0
            selected = []
            for file in supported_files:
                if total_size + file["Size"] > MAX_SIZE_BYTES:
                    continue
                selected.append(file)
                total_size += file["Size"]

            documents = []
            for file_info in selected:
                key = file_info["Key"]
                size_bytes = file_info["Size"]
                documents.append(
                    {
                        "key": key,
                        "size_bytes": size_bytes,
                    }
                )

        descriptor = {
            "bucket": input_data_bucket_name,
            "prefix": input_data_path,
            "documents": documents,
            "total_size_bytes": total_size,
            "count": len(documents),
        }

        logger.info("Sampled documents descriptor content %s", descriptor)

        os.makedirs(sampled_documents.path, exist_ok=True)
        descriptor_path = os.path.join(sampled_documents.path, SAMPLED_DOCUMENTS_DESCRIPTOR_FILENAME)
        with open(descriptor_path, "w") as f:
            yaml.safe_dump(descriptor, f, default_flow_style=False, sort_keys=False)

        logger.info("Sampled documents descriptor written to %s", descriptor_path)

    build_and_write_descriptor()


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_sampling,
        package_path=__file__.replace(".py", "_component.yaml"),
    )

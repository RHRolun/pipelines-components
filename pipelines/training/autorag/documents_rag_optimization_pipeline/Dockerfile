FROM quay.io/aipcc/base-images/cpu:3.3

# Install all packages required for the AutoRAG components
RUN pip install --extra-index-url https://pypi.org/simple --no-cache-dir \
    "ai4rag@https://github.com/IBM/ai4rag/archive/refs/heads/dev.zip" \
    "PyYAML==6.0.3" \
    "langchain-text-splitters==1.1.1" \
    "docling[ort]==2.76.0" \
    "docling-ibm-models" \
    "openpyxl==3.1.5" \
    "xlrd==2.0.1" \
    "odfpy==1.4.1" \
    "huggingface_hub==0.36.0" \
    "numpy==2.4.2" \
    "pandas==2.2.3"
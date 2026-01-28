ARG BUILD_FROM
FROM $BUILD_FROM

# Install runtime packages
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    py3-scikit-learn \
    tesseract-ocr \
    tesseract-ocr-data-eng \
    nginx \
    libstdc++ \
    libgcc \
    libgomp \
    ffmpeg \
    ffmpeg-libs

# Install Python packages (using pre-built wheels where possible)
RUN pip3 install --no-cache-dir --break-system-packages \
    flask \
    flask-cors \
    pytesseract \
    opencv-python-headless \
    numpy \
    requests \
    Pillow

# Install PyTorch CPU-only version
RUN pip3 install --no-cache-dir --break-system-packages \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install transformers
RUN pip3 install --no-cache-dir --break-system-packages \
    transformers

# Install EasyOCR (uses PyTorch, already installed)
RUN pip3 install --no-cache-dir --break-system-packages \
    easyocr

# Install PaddleOCR with CPU-only PaddlePaddle
RUN pip3 install --no-cache-dir --break-system-packages \
    paddlepaddle \
    paddleocr

# Copy application files
COPY rootfs /

# Set working directory
WORKDIR /app

# Copy application
COPY app /app

# Make scripts executable
RUN chmod a+x /etc/services.d/camera-extractor/run \
    && chmod a+x /etc/services.d/camera-extractor/finish \
    && chmod a+x /etc/services.d/nginx/run

# Build arguments
ARG BUILD_ARCH
ARG BUILD_DATE
ARG BUILD_DESCRIPTION
ARG BUILD_NAME
ARG BUILD_REF
ARG BUILD_REPOSITORY
ARG BUILD_VERSION

# Labels
LABEL \
    io.hass.name="${BUILD_NAME}" \
    io.hass.description="${BUILD_DESCRIPTION}" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION} \
    maintainer="Muhammet Turşak <tursoft@gmail.com>" \
    org.opencontainers.image.title="${BUILD_NAME}" \
    org.opencontainers.image.description="${BUILD_DESCRIPTION}" \
    org.opencontainers.image.vendor="Tursoft" \
    org.opencontainers.image.authors="Muhammet Turşak <tursoft@gmail.com>" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.url="https://tursoft.net" \
    org.opencontainers.image.source="https://github.com/${BUILD_REPOSITORY}" \
    org.opencontainers.image.documentation="https://github.com/${BUILD_REPOSITORY}/blob/main/README.md" \
    org.opencontainers.image.created=${BUILD_DATE} \
    org.opencontainers.image.revision=${BUILD_REF} \
    org.opencontainers.image.version=${BUILD_VERSION}

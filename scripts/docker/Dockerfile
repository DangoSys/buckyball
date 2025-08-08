FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    nodejs \
    npm \
    openjdk-11-jdk \
    scala \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /buckyball

# Copy project files
COPY . /buckyball/

# Install Python dependencies (if requirements.txt exists)
# RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi

# Install Node.js dependencies (if package.json exists)
# RUN if [ -f package.json ]; then npm install; fi

# Create a non-root user
RUN useradd -m -u 1000 bb && \
    chown -R bb:bb /buckyball
USER bb

# Expose common development ports
EXPOSE 3000 8080 8000

# Set default command
CMD ["/bin/bash"] 

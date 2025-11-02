# Buckyball Docker Environment

This directory contains Docker configurations for running the Buckyball project.

## File Description

- `Dockerfile`: Main Docker image build file
- `docker-compose.yml`: Docker Compose configuration file for managing containers
- `.dockerignore`: Specifies which files should not be copied into the Docker container
- `README.md`: This documentation file

## Environment Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start

### 1. Build Image

```bash
# Execute from project root directory
docker build -f docker/Dockerfile -t buckyball-dev .
```

### 2. Start Environment Using Docker Compose

```bash
# Execute from docker directory
cd docker
docker-compose up -d
```

### 3. Enter Container

```bash
# Enter running container
docker exec -it buckyball-dev bash
```

## Detailed Usage Instructions

### Using Docker Compose

1. **Start Environment**:
```bash
cd docker
docker-compose up -d
```

2. **Check Container Status**:
```bash
docker-compose ps
```

3. **Enter Container**:
```bash
docker-compose exec buckyball bash
```

4. **Stop Environment**:
```bash
docker-compose down
```

5. **Rebuild and Start**:
```bash
docker-compose up --build -d
```

## Common Issues

1. When executing `docker-compose up -d`, the following error occurs:
```
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.45/containers/json?all=1&filters=%7B%22label%22%3A%7B%22com.docker.compose.config-hash%22%3Atrue%2C%22com.docker.compose.project%3Ddocker%22%3Atrue%7D%7D": dial unix /var/run/docker.sock: connect: permission denied
```
Execute the following command and logout and login again:
```
sudo usermod -aG docker $USER
```

# Buckyball Docker 环境

这个目录包含了用于运行Buckyball项目的Docker配置。

## 文件说明

- `Dockerfile`: 主要的Docker镜像构建文件
- `docker-compose.yml`: Docker Compose配置文件，用于管理容器
- `.dockerignore`: 指定哪些文件不需要复制到Docker容器中
- `README.md`: 本说明文件

## 环境要求

- Docker Engine 20.10+
- Docker Compose 2.0+

## 快速开始

### 1. 构建镜像

```bash
# 在项目根目录下执行
docker build -f docker/Dockerfile -t buckyball-dev .
```

### 2. 使用Docker Compose启动环境

```bash
# 在docker目录下执行
cd docker
docker-compose up -d
```

### 3. 进入容器

```bash
# 进入运行中的容器
docker exec -it buckyball-dev bash
```

## 详细使用说明

### 使用Docker Compose

1. **启动环境**：
```bash
cd docker
docker-compose up -d
```

2. **查看容器状态**：
```bash
docker-compose ps
```

3. **进入容器**：
```bash
docker-compose exec buckyball bash
```

4. **停止环境**：
```bash
docker-compose down
```

5. **重新构建并启动**：
```bash
docker-compose up --build -d
```

## 常见问题
1. 执行 `docker-compose up -d` 报错如下
```
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.45/containers/json?all=1&filters=%7B%22label%22%3A%7B%22com.docker.compose.config-hash%22%3Atrue%2C%22com.docker.compose.project%3Ddocker%22%3Atrue%7D%7D": dial unix /var/run/docker.sock: connect: permission denied
```
执行以下命令, 并退出终端重新登录
```
sudo usermod -aG docker $USER
```

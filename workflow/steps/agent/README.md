
1. API 配置
```shell
# in .env
API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
BASE_URL=https://api.deepseek.com/v
```

```shell
bbdev start

curl -X POST http://localhost:5000/deepseek/chat -H "Content-Type: application/json" -d '{"message": "你好，你是谁", "model": "deepseek-chat"}'
```
# 智能文档问答 Agent

基于 LangGraph + RAG 的文档问答系统，支持扫描版 PDF 解析、向量检索、多节点 Agent 自检与拒答。

## 技术栈

- Python 3.10+
- FastAPI — Web API 框架
- LangGraph + LangChain — Agent 框架
- docling — PDF 解析（支持扫描件 OCR）
- ChromaDB — 向量数据库
- BAAI/bge-small-zh-v1.5 模型
- qwen-plus — 大语言模型

## 环境准备

### 1. 创建 conda 环境

```bash
conda env create -f environment.yml
```

### 2. 激活环境

```bash

conda activate rag-agent
```

### 3. 配置环境变量

```bash
copy .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```bash
DASHSCOPE_API_KEY=your-dashscope-key
```

### 4. 创建数据目录

```bash
mkdir -p data/pdfs data/json data/logs
```

### 清数据：
Remove-Item -Recurse -Force ".\data\chroma_db" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force ".\data\json\*.json" -ErrorAction SilentlyContinue

## 启动服务

### 开发模式（热重载）

```bash
python app/main.py
```

或：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 生产模式

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

## API 使用

### 上传文档

```bash
curl.exe -X POST "http://localhost:8000/documents/upload" -F "file=@GBT 1568-2008 键 技术条件.pdf"
```

### 查询文档列表

```bash
curl "http://localhost:8000/documents"
```

### 提问

```bash
curl.exe --% -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"question\": \"键的表面粗糙度要求是什么？\"}"
```


curl.exe --% -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"question\": \"键的抗拉强度要求是多少？\"}"


curl.exe --% -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"question\": \"键表面不允许有哪些缺陷？\"}"


curl.exe --% -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"question\": \"键的检查项目和合格质量水平是什么？？\"}"



### 健康检查

```bash
curl "http://localhost:8000/health"
```

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_agent.py -v
```

## 项目结构

```
app/
  api/          — API 路由层
  core/         — 业务核心（Agent、检索、自检、文档服务）
  parser/       — PDF 解析与分块
  models/       — 数据模型
  config/       — 配置管理
  utils/        — 工具函数
  main.py       — 应用入口
tests/          — 测试套件
data/           — 数据存储（PDF、JSON、向量库、日志）
```

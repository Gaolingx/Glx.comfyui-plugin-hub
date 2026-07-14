# Multi-source Artist Grabber

以 Danbooru 的画师实体作为“身份锚点”，从其名称、别名和主页 URL 中确认同一画师，再下载 X、Danbooru、Pixiv、Gelbooru、Safebooru、Konachan、yande.re 等来源的作品，并为每张图片生成同名 `.txt` 标签。

## 推荐工作流：Danbooru 画师 → X / Pixiv / Booru 混合任务

1. 首次使用建议双击 `先运行这个.bat`：它会创建项目独立的 `.venv`，安装抓图、浏览器登录和本地 ONNX 打标所需的全部 Python 依赖；缺少 Python 或 Chrome 时会尝试通过 winget 自动安装。完成后双击 `start.bat`，打开 `http://127.0.0.1:8710`。
2. 在“混合来源”中勾选 X、Pixiv、Danbooru 或其他需要的来源，并选择当前要编辑的来源。
3. 输入 Danbooru 画师名、别名，或已知的 X 主页 URL，点击“模糊搜索”。
4. 从候选中确认画师。程序会从 Danbooru URL 自动映射 canonical tag、X handle/user ID 和 Pixiv 数字 user ID；没有登记时才需要手填。
5. X / Pixiv 推荐使用“专用登录窗口”：
   - 首次登录或切换账号时临时显示独立 Chrome；
   - 检查成功后关闭登录窗口；抓取时按需短暂启动 headless 浏览器，读取会话后自动退出；
   - 登录由 Chrome 加密持久化到 `%LOCALAPPDATA%\DanbooruGrabber\*BrowserProfile`，窗口或服务重启后会自动恢复；
   - X 与 Pixiv Profile 完全隔离，也不会读取日常 Chrome Profile。
   - 手填 `auth_token/ct0`、cookies.txt、浏览器 Profile、Pixiv access token/PHPSESSID 仍保留在折叠的兼容设置中。
6. 选择标签来源：
   - 仅来源标签；
   - 用户自己的 OpenAI-compatible 视觉模型 Key/Endpoint；
   - 本地 WD14 风格 ONNX 模型与 `selected_tags.csv`。
7. 开始下载。所有来源保存在 `downloads/<Danbooru画师名>/`，文件名带来源前缀；相同文件按 SHA-256 跨来源去重并合并 `.txt` 标签。单个来源失败不会中断其他来源。

这种 URL 精确关联优先于“仅凭同名自动猜测”。没有登记 X URL 时仍可手动填写账号，但应自行确认身份。

## 已启用来源

| 来源 | 状态 | 认证 |
| --- | --- | --- |
| X / Twitter | gallery-dl + 实测 | 专用持久登录（推荐）；兼容 `auth_token/ct0`、cookies.txt |
| Danbooru | 稳定公开 API | 用户名/API Key 可选 |
| Openverse | 开放许可聚合 API | 匿名，可筛商业使用/允许修改 |
| Gelbooru | DAPI | User ID + API Key |
| Safebooru | DAPI | 匿名 |
| Konachan / yande.re | Moebooru API | 匿名 |
| Pixiv | 网页/API + 实测 | 专用持久登录（推荐）；兼容 access token / PHPSESSID；公开作品可匿名 |

争议或高风险来源不会因为目录中存在适配器就自动启用；实际白名单在 `sources/__init__.py`。

## 打标后端

### OpenAI-compatible 视觉模型

填写 Base URL、API Key（本地无鉴权服务可留空）和支持图片输入的模型名。可接云端服务，也可接 Ollama、llama.cpp 等提供 OpenAI-compatible 接口的本地服务。提示词留空时使用内置 Danbooru / WD14 风格预设；程序优先请求严格 JSON schema，并对不支持 `response_format` 的兼容服务自动降级。

远程模式会把图片发送到用户填写的 Endpoint。程序不会把 LLM Key 写入浏览器 `localStorage`。

### 本地 ONNX

支持常见 WD14 风格模型：

```powershell
python -m pip install onnxruntime Pillow numpy
```

也可以直接运行根目录的 `先运行这个.bat`，一次安装核心抓图依赖与上述 ONNX 依赖。`start.bat` 会优先使用项目内 `.venv`，不会依赖接收者机器上的全局 Python 包。

界面中填写：

- `.onnx` 模型路径；
- 对应的 `selected_tags.csv`；
- 标签阈值，默认 `0.35`。

## 凭据安全

- 服务只监听 `127.0.0.1`。
- X/Pixiv Cookie、站点 API Key、LLM Key 不写入 `localStorage`。
- 专用 X/Pixiv 登录由 Chrome 自身加密保存 Profile；程序状态文件只保存本机端口、进程与运行模式，不保存 Cookie 明文。
- gallery-dl/网页请求所需的临时 cookies.txt 会在每次调用结束后覆盖删除。
- `auth_token` 基本等同 X 登录会话。不要发给他人，不要贴进日志、截图或问题报告。
- X 的实际 Cookie 名是 `auth_token`，不是 Twitter API Key；界面兼容旧称 `auth_key`，但文档统一使用正确名称。
- 输入框兼容纯值、`auth_token=...` / `ct0=...`，以及误粘贴的完整 Cookie Header。

## 开发与测试

```powershell
python -m compileall -q app.py http_util.py sources tagging
python -m pip install -r requirements.txt
python -m unittest discover -s tagging -p "test_*.py" -q
python -m unittest discover -s tests -p "test_*.py" -q
python app.py --no-browser
```

## 合规说明

API 可访问不等于获得训练、再分发或商业使用许可。Booru、X、Pixiv 上的作品通常仍归原作者所有；请遵守来源站点条款、速率限制与作品版权。

用户提到的 `Twitter-Insight-LLM` 仓库没有许可证，且抓取代码与模型接口已经老旧，因此本项目没有复制或打包其源码，而是按当前需求独立实现“X 媒体来源 + 画师身份映射 + 可插拔打标器”。

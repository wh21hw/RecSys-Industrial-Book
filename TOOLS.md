# TOOLS.md - Local Notes

## Skills 使用经验

### 学术调研（web-search + web-reader）
- 多轮搜索策略：先 broad survey → 再具体论文 → 再工业应用，分批并行调用节省时间
- web_reader 读取 arxiv HTML 版本比 PDF 更可靠（PDF 可能解析失败）
- AMiner free tier 需要 API key，未配置时回退到 web-search
- 大规模调研（30+ 论文）会消耗大量工具调用，注意上下文管理

### 文件输出
- 用户要求 Markdown 格式交付，保存到 `download/` 目录
- 学术类文档需要结构化编排：问题 → 方法 → 架构 → 局限
- 用户对文档质量要求高，需要完整覆盖而非概要

### 工具调用上限
- 单次会话工具调用有上限，超长执行链（如 50+ web_reader）可能被截断
- 策略：优先搜索收集信息，再集中写入文件，避免搜索和写入交替导致中断
- 本次调研（Scaling Law 综述）执行了约 30 次搜索 + 15 次论文阅读，在上下文管理上表现良好
- 论文阅读优先选 arxiv HTML 版本（page_reader），提取纯文本后分析，比 PDF 更可靠
- 搜索策略：先核心关键词（scaling law + recsys）→ 再细分方向（one-epoch / depth / embedding / industrial）→ 再具体论文精读
- 并行搜索（一次 3 个 query）可大幅节省时间

### PDF OCR（扫描件处理）
- 扫描件 PDF（纯图片，无可提取文字）需要 OCR：先用 PyMuPDF `page.get_pixmap(dpi=200)` 提取每页为 PNG，再用 VLM CLI 逐页识别
- VLM CLI 用法：`z-ai vision -p "提取文字" -i "./page.png" -o output.json`，结果从 JSON 的 `choices[0].message.content` 读取
- 批量页面可循环调用，注意每页约 10-15 秒，8 页约 2 分钟
- PyMuPDF 检测方法：`page.get_text()` 为空 + `page.get_images()` 有图 → 扫描件

- 工作目录：`/home/z/my-project/`
- 下载目录：`/home/z/my-project/download/`
- 时区：Asia/Shanghai

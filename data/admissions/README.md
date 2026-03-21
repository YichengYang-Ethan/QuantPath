# MFE 申请数据集

## 文件说明

| 文件 | 说明 |
|------|------|
| `template.csv` | 字段定义模板，不含数据 |
| `sample.csv` | 初始手工录入样本（10位申请人，31条记录）|
| `collected.csv` | 自动收集的完整数据集（由工具生成）|
| `.seen_urls.txt` | 已爬取URL记录（防止重复）|

## 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 申请人ID（同一申请人多个项目共用同一ID）|
| `gender` | M/F/unknown | 性别 |
| `bg_type` | string | 学校背景：985 / 211 / 双非一本 / 海本(Top10/30/50) / IIT |
| `nationality` | string | 中国大陆 / 美籍 / 印度 / 港澳台 / 其他 |
| `gpa` | float | GPA数值 |
| `gpa_scale` | int | GPA满分：4 / 100 / 10 |
| `gre` | int | GRE总分（Quant+Verbal，如331）或仅Quant（如170）|
| `toefl` | int | TOEFL iBT分数 |
| `major` | string | 主专业 |
| `intern_desc` | string | 实习经历简述 |
| `has_paper` | 是/否/不明 | 是否有发表论文 |
| `has_research` | 是/否/不明 | 是否有学术研究经历 |
| `courses_note` | string | 特殊课程备注（随机微积分/实分析/C++等）|
| `program` | string | 项目ID（对应 data/programs/ 目录）|
| `result` | string | accepted / rejected / waitlisted / interview / pending |
| `season` | string | 申请季，如 26Fall / 25Fall |
| `source` | string | 数据来源：小红书 / chasedream / quantnet / gradcafe 等 |

## 数据收集工具

```bash
# 合并 sample.csv 到 collected.csv
python tools/parse_admissions.py --merge-sample

# 解析小红书/chasedream 帖子文本
python tools/parse_admissions.py --input post.txt --season 26Fall --source 小红书

# 批量解析目录下所有 .txt 文件
python tools/parse_admissions.py --dir posts/ --season 26Fall --source 小红书

# 自动爬取 QuantNet（需要 ANTHROPIC_API_KEY）
python tools/collect_data.py --source quantnet --pages 5

# 验证数据质量
python tools/parse_admissions.py --validate
```

## 数据来源

- **小红书**：搜索"MFE offer 2026"、"MFE申请结果"，大量中国申请者分享完整profile+结果
- **chasedream**：一亩三分地类似，MFE专版有历年汇总帖
- **offershow**：结构化offer分享平台
- **QuantNet**：英文MFE社区，profile evaluation帖子含结果更新
- **GradCafe**：financial engineering搜索结果（数据较少）

## 隐私说明

所有数据来自用户主动公开分享的论坛/社交媒体内容。
不收集任何个人身份信息（姓名、邮箱等）。

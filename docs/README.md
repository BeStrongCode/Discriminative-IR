# 论文 Project Page

纯静态页面，无需安装依赖或构建。入口为 `index.html`，直接用浏览器打开即可。

## 填写内容

| 内容 | 修改位置 |
| --- | --- |
| 网页标题、SEO 描述、分享信息 | `index.html` 的 `<head>` |
| 论文标题、会议、作者、单位 | `index.html` 的 `.hero` |
| Paper / arXiv / Code 链接 | `assets/script.js` 的 `PROJECT_LINKS`，填入后自动启用 |
| 恢复输入 / 输出图与图注 | `assets/script.js` 的 `RESTORATION_TASKS` |
| 摘要、贡献点 | `index.html` 的 `#abstract` |
| 方法图 | 替换 `assets/method-placeholder.svg`，或修改 `#method` 下图片的 `src` 和 `alt` |
| 实验表格、指标、数据集 | `index.html` 的 `#results` |
| BibTeX | `index.html` 的 `#bibtex`，复制按钮自动读取 |
| 致谢 / 联系方式 | `index.html` 的 `<footer>` |
| 配色、字体、页面宽度 | `assets/styles.css` 开头的 `:root` |

HTML 和 JavaScript 中也有中文 `TODO` / 配置注释。页面正文以英文占位，便于直接替换为论文内容。

## 替换对比图

将图片放进 `docs/assets/`，在 `assets/script.js` 中配置，例如：

```js
denoising: {
  label: "Denoising",
  before: "assets/denoising-input.png",
  after: "assets/denoising-ours.png",
  caption: "[Dataset / noise level / example description]",
  alt: "[Describe the image content]",
},
```

每对图片请使用相同尺寸和构图；页面以居中裁切方式适配桌面和移动端。`before` 和 `after` 同时填写且加载成功后，示意退化效果会自动关闭，显示真实结果。未配置或加载失败时保留带说明的示意图。点击任务或用左右方向键切换，拖动滑块或用方向键调整分界线。

默认的 SVG、噪声、雨、雾、模糊和暗光效果仅用于排版预览，不代表论文实验结果。实验表格没有虚构数值。

## 本地预览与发布

从仓库根目录执行：

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

浏览器访问 `http://127.0.0.1:8000`。按 `Ctrl+C` 停止预览。

如使用 GitHub Pages，在仓库设置中选择需要发布的分支和 `/docs` 目录，无需额外构建。发布前替换方括号占位内容、示意图和 BibTeX，检查所有链接与元信息。

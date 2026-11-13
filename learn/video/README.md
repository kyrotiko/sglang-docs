# V4 → V4.1 视频的制作源文件

视频复用上一级 `sglang-deepseek-v4.1-viz.html` 的章节、节点与数据流，通过 Chrome 的 SVG 时间轴逐帧渲染，再用 FFmpeg 编码为 H.264 / AAC MP4。它不是录屏实时操作，也不需要外部语音服务。

- `dsv41-video-script.json`：21 个片段的画面标题、讲解要点与旁白文本。
- `narrate-dsv41.ps1`：用 Windows 本地 `System.Speech` 中文语音生成分段 WAV。
- `build-dsv41-video.cjs`：生成旁白、渲染 1080p 动画、编码并写出播放器与字幕。

## 运行条件

Windows、Node.js 22 或更新版本、Chrome、FFmpeg（带 libx264 / AAC）以及本地中文语音。本次使用 `Microsoft Huihui Desktop`；可在脚本 JSON 中调整语音与语速。构建脚本不自动下载或安装工具。

从任意工作目录调用，所有路径显式指定：

```bash
node /path/to/learn/video/build-dsv41-video.cjs \
  --ffmpeg C:/path/to/ffmpeg.exe \
  --chrome 'C:/Program Files/Google/Chrome/Application/chrome.exe' \
  --work C:/path/to/video-work \
  --preview
```

先检查工作目录里的 `preview.png` 与 `preview.mp4`；去掉 `--preview` 后构建完整视频。已有 MP4 时默认拒绝覆盖；检查目标后可用 `--overwrite` 重建。

工作目录保存旁白 WAV、循环动画片段与编码中间件。相同旁白内容可复用已生成 WAV；画面会重新渲染。中间帧、Chrome profile 和编码器均不打包进学习目录。

## 输出

全部位于上一级 `learn/`：

- `sglang-deepseek-v4.1.mp4`：1920×1080、24 fps，H.264 视频、AAC 中文旁白，内含章节 metadata。
- `sglang-deepseek-v4.1-video.html`：离线播放器，原生播放控件、倍速与章节跳转。
- `sglang-deepseek-v4.1-video-poster.jpg`：播放封面。
- `sglang-deepseek-v4.1-video.srt` / `.vtt`：旁白逐片段字幕。
- `sglang-deepseek-v4.1-video-chapters.json`：章节起点与总时长。

画面中的文字是讲解要点；旁白字幕单独提供完整口播文本。流动速度与片段时长用于教学，不能作为算子延迟或性能收益的依据。动画内容的来源与设备边界见配套 Markdown 笔记。

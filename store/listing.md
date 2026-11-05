# Chrome 网上应用店上架材料

结构沿用鲸鱼音效台上架时的准备清单。开发者账号的注册、付款、登录和最终提交，都由账号持有人在 [开发者控制台](https://chrome.google.com/webstore/devconsole/) 完成。

## 商品详情

**名称**（清单里的 `name`，上限 75 字符）

FrameBoost 网页视频补帧

**简介**（清单里的 `description`，上限 132 字符）

用本机显卡运行 AI 模型，把网页视频实时补到 60 / 120 fps。全部在本机运行，不上传任何数据。

**详细说明**

FrameBoost 让网页视频看起来更流畅：它在两帧真实画面之间实时插入新的画面，把 24、30 fps 的视频补到 60、120 fps，或者你屏幕的刷新率上限。

主要特点：
· AI 补帧：内置 RIFE 4.9 神经网络，通过 WebGPU 在你自己的显卡上运行，模型尺寸根据显卡余量自动调整。
· 自动生效：播放视频时自动找到画面，不需要额外操作，支持大多数视频网站和本地视频文件。
· 音画同步：AI 模式下画面延后 0.3 秒，声音同步延后，看电影没有音画差。
· 稳定的节奏：输出帧率取屏幕刷新率的整除值，避免忽快忽慢。
· 显卡不够时自动降级：被游戏占满显卡时退回轻量的光流补帧，稍后自动重试。
· 快捷键 Alt+Shift+F 随时开关，方便对比效果。

隐私：画面只在本机内存和显卡中处理，扩展不向开发者或任何第三方发送请求，没有统计和广告。

使用要求：Chrome 或 Edge 121 及以上；AI 模式需要支持 WebGPU 的独立显卡，否则使用光流模式。

已知限制：
· Netflix、Disney+、Prime Video 等受 DRM 保护的视频浏览器不允许读取画面，无法补帧。
· AI 模式下暂停、调音量会有 0.3 秒滞后。
· 快速运动、车轮辐条这类细而快的物体仍可能有拖影或重影。
· 帧率不会超过屏幕刷新率。

开源地址：https://github.com/whaleinblack/frameboost

**类别**：娱乐

**语言**：中文（简体）。扩展界面目前只有中文；英文说明见下，可在控制台另加英文商品详情。

### 英文商品详情（可选）

**Summary**: Smooth web video to 60 / 120 fps with an AI model running on your own GPU. Nothing leaves your computer.

**Description**: FrameBoost inserts new frames between the real ones while a video plays, turning 24 or 30 fps video into 60, 120 fps or your display's refresh rate. It runs the RIFE 4.9 network on your own GPU through WebGPU, picks a model size your GPU can sustain, delays the sound to stay in sync, and falls back to lightweight optical flow when a game takes the GPU. Frames are processed in memory on your computer; the extension sends nothing to the developer or anyone else. DRM-protected video (Netflix, Disney+, Prime Video) cannot be read by extensions and is left alone. The interface is in Chinese. Source: https://github.com/whaleinblack/frameboost

## 隐私权规范

**单一用途**

提高网页视频的显示帧率：在视频播放时实时生成中间帧，让画面更流畅。

**权限理由**

| 权限 | 理由 |
| --- | --- |
| 主机权限（所有网站）和所有网站上的内容脚本 | 视频可能出现在任何网站上。扩展需要在页面中找到正在播放的视频元素，读取它的画面，在视频上方叠加补帧后的画布。画面只在本机处理，不发送到任何地方。 |
| scripting | 用户打开工具栏弹窗时，读取当前标签页中视频的补帧状态（帧率、分辨率、是否受 DRM 保护）；以及用户点击“重新加载视频”按钮时，在该标签页执行重新加载。 |
| storage | 保存用户的设置（开关、目标帧率、画质、是否显示帧率），以及这块显卡能实时运行的模型尺寸。 |
| declarativeNetRequestWithHostAccess | 仅在用户对跨域视频手动点击“重新加载视频以启用补帧”后使用：为该标签页的媒体响应添加 Access-Control-Allow-Origin 响应头，使浏览器允许扩展读取画面。规则只作用于该标签页，标签页关闭时删除。 |

**远程代码**：否。所有 JavaScript、WGSL 着色器和模型权重都打包在扩展内。

**数据使用**：所有数据类别都不勾选（不收集任何用户数据）。三项承诺全部勾选：不出售给第三方、不用于与单一用途无关的目的、不用于判断信用或放贷。

**隐私政策网址**：https://github.com/whaleinblack/frameboost/blob/main/PRIVACY.md （推送到 GitHub 后生效）

## 图片素材

| 素材 | 规格 | 状态 |
| --- | --- | --- |
| 商店图标 | 128×128 PNG | 已有 `icons/icon128.png` |
| 屏幕截图 | 1280×800 或 640×400，1–5 张 | 待制作 |
| 小型宣传图块 | 440×280 | 待制作 |
| 顶部宣传图块 | 1400×560 | 可选 |

截图不用有版权的视频画面，计划用 Big Buck Bunny（CC BY 3.0，Blender Foundation），并在说明里注明出处。计划内容：

1. 视频页面上显示“24 → 120 fps”的实时帧率，旁边是工具栏弹窗。
2. 两张真实画面之间插入的中间帧排成一行，展示补帧在做什么。
3. 弹窗设置的特写：目标帧率和画质。
4. “全部在本机运行”的说明卡片。

## 审核测试说明

在任意不受 DRM 保护的视频页面播放视频（例如 YouTube 上的 Big Buck Bunny），点工具栏图标，弹窗会显示“24 → 120 fps”这类实时帧率。审核机器没有支持 WebGPU 的显卡时，扩展会自动使用光流模式，功能同样可用。按 Alt+Shift+F 可开关补帧对比效果。不需要账号或测试凭据。

## 提交前检查

- [ ] 账号持有人注册 Chrome Web Store 开发者账号（一次性付费），完成联系邮箱验证。
- [ ] 推送代码到 GitHub，确认隐私政策网址能公开打开。
- [ ] 运行 `npm test`，确认测试通过。
- [ ] 运行 `npm run pack` 生成 `frameboost-1.0.0.zip` 并上传。
- [ ] 制作并上传截图和宣传图块。
- [ ] 在控制台填写上面的商品详情、隐私权规范和测试说明后提交审核。

请求所有网站的访问权限，审核通常会更深入，等待时间可能比一般扩展长。同一个 zip 也可以免费提交到 Microsoft Edge 加载项商店。

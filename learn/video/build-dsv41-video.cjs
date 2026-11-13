'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');
const { once } = require('events');

function option(name, fallback) {
  const i = process.argv.indexOf(name);
  return i < 0 ? fallback : process.argv[i + 1];
}
const preview = process.argv.includes('--preview');
const overwrite = process.argv.includes('--overwrite');
const learn = path.resolve(__dirname, '..');
const planPath = path.join(__dirname, 'dsv41-video-script.json');
const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
const ffmpeg = option('--ffmpeg', process.env.FFMPEG_PATH);
const chromePath = option('--chrome', 'C:/Program Files/Google/Chrome/Application/chrome.exe');
const work = option('--work', fs.mkdtempSync(path.join(os.tmpdir(), 'sglang-v41-movie-')));
const output = path.join(learn, 'sglang-deepseek-v4.1.mp4');
const stem = 'sglang-deepseek-v4.1-video';
const fps = 24;
const loopFrames = 48;
if (!ffmpeg || !fs.existsSync(ffmpeg)) throw new Error('Pass --ffmpeg with the encoder executable path.');
if (!fs.existsSync(chromePath)) throw new Error('Pass --chrome with a Chrome executable path.');
if (!preview && fs.existsSync(output) && !overwrite) throw new Error('Output exists. Inspect it before using --overwrite.');
fs.mkdirSync(work, { recursive: true });

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
function execute(binary, args) {
  return new Promise((resolve, reject) => {
    const process = spawn(binary, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let log = '';
    process.stdout.on('data', chunk => { log = (log + chunk).slice(-16000); });
    process.stderr.on('data', chunk => { log = (log + chunk).slice(-16000); });
    process.on('error', reject);
    process.on('close', code => code === 0 ? resolve(log) : reject(new Error(`${path.basename(binary)} exited ${code}\n${log}`)));
  });
}
function wavSeconds(file) {
  const bytes = fs.readFileSync(file);
  let byteRate = 0, size = 0;
  if (bytes.toString('ascii', 0, 4) !== 'RIFF') throw new Error('Not a RIFF wave: ' + file);
  for (let offset = 12; offset + 8 <= bytes.length;) {
    const id = bytes.toString('ascii', offset, offset + 4);
    const length = bytes.readUInt32LE(offset + 4);
    if (id === 'fmt ') byteRate = bytes.readUInt32LE(offset + 16);
    if (id === 'data') size += length;
    offset += 8 + length + length % 2;
  }
  if (!byteRate || !size) throw new Error('Missing wave data: ' + file);
  return size / byteRate;
}
function timestamp(seconds, srt = false) {
  const n = Math.round(seconds * 1000);
  const h = Math.floor(n / 3600000), m = Math.floor(n / 60000) % 60;
  const s = Math.floor(n / 1000) % 60, ms = n % 1000;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}${srt ? ',' : '.'}${String(ms).padStart(3, '0')}`;
}
function clock(seconds) { return Math.floor(seconds / 60) + ':' + String(Math.floor(seconds % 60)).padStart(2, '0'); }

let socket, browser, seq = 0;
const calls = new Map();
const browserErrors = [];
function cdp(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++seq;
    const timeout = setTimeout(() => { calls.delete(id); reject(new Error('Timeout: ' + method)); }, 30000);
    calls.set(id, { resolve, reject, timeout });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const reply = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (reply.exceptionDetails) throw new Error(JSON.stringify(reply.exceptionDetails));
  return reply.result.value;
}
async function openBrowser() {
  const profile = fs.mkdtempSync(path.join(work, 'chrome-profile-'));
  browser = spawn(chromePath, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank'], { stdio: 'ignore' });
  let port;
  for (let i = 0; i < 150; i++) {
    const file = path.join(profile, 'DevToolsActivePort');
    if (fs.existsSync(file)) { port = Number(fs.readFileSync(file, 'utf8').split('\n')[0]); break; }
    await delay(100);
  }
  if (!port) throw new Error('Chrome startup failed.');
  const pages = await (await fetch('http://127.0.0.1:' + port + '/json')).json();
  socket = new WebSocket(pages.find(p => p.type === 'page').webSocketDebuggerUrl);
  await once(socket, 'open');
  socket.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.id && calls.has(msg.id)) {
      const entry = calls.get(msg.id); calls.delete(msg.id); clearTimeout(entry.timeout);
      if (msg.error) entry.reject(new Error(JSON.stringify(msg.error))); else entry.resolve(msg.result);
    }
    if (msg.method === 'Runtime.exceptionThrown') browserErrors.push(msg.params.exceptionDetails.text);
  };
  await cdp('Runtime.enable');
  await cdp('Page.enable');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
  await cdp('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] });
  await cdp('Page.navigate', { url: 'file:///' + path.join(learn, 'sglang-deepseek-v4.1-viz.html').replace(/\\/g, '/') });
  await delay(400);
  await evaluate(`(() => {
    pause();
    document.documentElement.dataset.theme = 'dark';
    document.body.classList.remove('texture', 'reduced');
    const style = document.createElement('style');
    style.textContent = ${JSON.stringify(`
      * { transition:none!important; scroll-behavior:auto!important }
      body { width:1920px; height:1080px; overflow:hidden; background:#0d0d0d }
      .shell { display:block; min-height:0 }
      .sidebar,.topbar,.hero,.legend,.lesson-head,.controls,.step-strip,.inspector,.lab,.subgrid,.quiz,.chapter-footer,.footnote,.skip { display:none!important }
      .main { max-width:none; width:1920px; margin:0; padding:30px 58px 0 }
      #movieHeader { height:204px; padding-top:5px }
      .movie-brand { display:flex; justify-content:space-between; color:var(--secondary); font-size:17px; letter-spacing:1px }
      #movieSection { color:var(--secondary); font-size:19px; margin-top:24px; letter-spacing:1px }
      #movieTitle { font-size:43px; font-weight:650; line-height:1.4; margin-top:5px; letter-spacing:-.6px }
      .movie-chapters { display:flex; gap:11px; margin-top:14px; font-size:16px; color:var(--muted) }
      .movie-chapters span { padding:2px 13px 3px 0; border-right:1px solid var(--line) }
      .movie-chapters span:last-child { border:0 }
      .movie-chapters .current { color:var(--ink); font-weight:650 }
      .player { border:0; box-shadow:none; background:none; border-radius:0; overflow:visible }
      .diagram-grid { height:562px; gap:22px }
      .diagram-panel,.diagram-panel.v41 { border:1px solid var(--line); border-radius:15px; background:var(--surface); overflow:hidden }
      .panel-title { height:62px; padding:14px 23px 0 }
      .panel-title strong { font-size:23px }
      .panel-title span:last-child { font-size:16px }
      .svg-wrap { height:427px; min-height:0; padding:0 12px }
      .svg-wrap svg { width:100%; height:427px; max-height:none }
      .node .node-title { font-size:15px }
      .node .node-sub { font-size:11.5px }
      .wire { stroke-width:1.6 }
      .wire.lit { stroke-width:2.3 }
      .edge-caption { font-size:11px!important }
      .panel-bottom { padding:0 23px 17px; min-height:0; height:68px; font-size:16px; line-height:1.55 }
      .narration { margin-top:23px; border:1px solid var(--line); border-radius:13px; background:var(--surface); height:199px; padding:22px 27px; grid-template-columns:47px 1fr; gap:17px }
      .big-step { font-size:29px; padding-top:0 }
      .narration h3 { font-size:22px; margin-bottom:9px; font-weight:650; color:var(--secondary) }
      .narration p { max-width:none; font-size:27px; line-height:1.6; color:var(--ink) }
      .narration .takeaway { font-size:19px; line-height:1.5; margin-top:10px; color:var(--secondary) }
      #movieFooter { position:fixed; bottom:20px; left:61px; right:61px; display:flex; justify-content:space-between; font-size:15px; color:var(--muted) }
      #movieBar { position:fixed; bottom:0; left:0; height:4px; background:var(--orange) }
    `)};
    document.head.append(style);
    const header = document.createElement('div'); header.id = 'movieHeader';
    header.innerHTML = '<div class="movie-brand"><span>SGLang / LEARN · DEEPSEEK V4 → V4.1</span><span id="movieCount"></span></div><div id="movieSection"></div><h1 id="movieTitle"></h1><div class="movie-chapters" id="movieChapters"></div>';
    document.querySelector('.main').prepend(header);
    const footer = document.createElement('div'); footer.id='movieFooter';
    footer.innerHTML = '<span>源码快照 4cafabd1a3 · 节点与速度为教学示意</span><span>中文合成旁白 · V4 指原版基线，0731 / 0813 等变体另行区分</span>';
    document.body.append(footer);
    const bar=document.createElement('div'); bar.id='movieBar'; document.body.append(bar);
    window.renderMovieScene = (scene,index,count) => {
      state.chapter=scene.chapter; state.step=scene.step; state.playing=true; state.table=false; state.speed=1;
      renderLesson(); pause();
      document.getElementById('movieCount').textContent = '片段 '+String(index+1).padStart(2,'0')+' / '+count;
      document.getElementById('movieSection').textContent = scene.section;
      document.getElementById('movieTitle').textContent = scene.title;
      const names=['全景','压缩共享','检索','mHC','Engram','DSpark','图执行','平台适配'];
      const nav=document.getElementById('movieChapters'); nav.replaceChildren(...names.map((name,i)=>{const x=document.createElement('span');x.textContent=String(i+1).padStart(2,'0')+' '+name;if(i===scene.chapter)x.className='current';return x;}));
      document.getElementById('stepNum').textContent=String(index+1).padStart(2,'0');
      document.getElementById('stepTitle').textContent='画面要点';
      document.getElementById('stepText').textContent=scene.caption;
      document.getElementById('movieBar').style.width=((index+1)/count*100)+'%';
      document.querySelectorAll('animateMotion').forEach(m=>m.setAttribute('dur','2s'));
      document.querySelectorAll('.svg-wrap svg').forEach(s=>{s.pauseAnimations();s.setCurrentTime(0);});
      window.scrollTo(0,0);
    };
    return true;
  })()`);
}
async function renderScene(scene, index) {
  await evaluate(`renderMovieScene(${JSON.stringify(scene)},${index},${plan.scenes.length})`);
  const bounds = await evaluate(`({overflow:document.documentElement.scrollWidth>1920,captionBottom:document.querySelector('.narration').getBoundingClientRect().bottom,diagramBottom:document.querySelector('.diagram-grid').getBoundingClientRect().bottom})`);
  if (bounds.overflow || bounds.captionBottom > 1027 || bounds.diagramBottom > 820) throw new Error('Video layout overflow: ' + JSON.stringify(bounds));
}
async function frameAt(time, format = 'png') {
  await evaluate(`document.querySelectorAll('.svg-wrap svg').forEach(s=>s.setCurrentTime(${time}))`);
  const image = await cdp('Page.captureScreenshot', { format, ...(format === 'jpeg' ? { quality: 90 } : {}), captureBeyondViewport: false });
  return Buffer.from(image.data, 'base64');
}
async function animationLoop(index) {
  const dest = path.join(work, `loop-${String(index).padStart(2, '0')}.mp4`);
  const encoder = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', String(fps), '-i', 'pipe:0', '-frames:v', String(loopFrames), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-threads', '4', '-pix_fmt', 'yuv420p', dest], { stdio: ['pipe', 'ignore', 'pipe'] });
  let log = '';
  encoder.stderr.on('data', b => { log = (log + b).slice(-12000); });
  const closed = new Promise((resolve, reject) => {
    encoder.on('error', reject);
    encoder.on('close', code => code === 0 ? resolve() : reject(new Error('Frame encoding failed: ' + log)));
  });
  encoder.stdin.on('error', () => {});
  try {
    for (let f = 0; f < loopFrames; f++) {
      const image = await frameAt(f / fps);
      if (!encoder.stdin.write(image)) await once(encoder.stdin, 'drain');
    }
    encoder.stdin.end(); await closed;
  } catch (error) { encoder.kill(); await closed.catch(() => {}); throw error; }
  return dest;
}
async function compose(scene, index, loop) {
  const dest = path.join(work, `scene-${String(index).padStart(2, '0')}.mkv`);
  const fadeStart = (scene.duration - 0.25).toFixed(4);
  const log = await execute(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-stream_loop', '-1', '-i', loop, '-i', scene.wave,
    '-filter_complex', `[0:v]fade=t=in:st=0:d=0.22,fade=t=out:st=${fadeStart}:d=0.25[v];[1:a]adelay=350,apad,afade=t=out:st=${fadeStart}:d=0.25[a]`,
    '-map', '[v]', '-map', '[a]', '-t', scene.duration.toFixed(6), '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-threads', '4', '-pix_fmt', 'yuv420p', '-r', String(fps), '-c:a', 'pcm_s16le', '-ar', '24000', '-ac', '1', dest]);
  return dest;
}
function chapterList(scenes) {
  const list=[];
  scenes.forEach(s => { if(!list.length || list[list.length-1].label !== s.section) list.push({label:s.section,start:s.start}); });
  return list;
}
function playerHtml(scenes, duration) {
  const chapters = chapterList(scenes);
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4 → V4.1 · 视频讲解</title>
<style>:root{color-scheme:dark;--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--secondary:#c3c2b7;--line:#383835}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1280px;padding:30px 24px 40px;margin:auto}header{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:20px}h1{font-size:29px;line-height:1.4;margin:5px 0;letter-spacing:-.6px}p{margin:5px 0;color:var(--secondary);font-size:13px}.eyebrow{font:11px ui-monospace,Consolas,monospace;letter-spacing:2px;color:var(--secondary)}video{display:block;width:100%;aspect-ratio:16/9;background:#000;border:1px solid var(--line);border-radius:12px}button,select{font:inherit;background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:8px 12px;cursor:pointer}button:hover{background:#272726}button:focus-visible,a:focus-visible,select:focus-visible{outline:2px solid #86b6ef;outline-offset:3px}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:15px;flex-wrap:wrap;margin:16px 0 22px;font-size:13px}.toolbar label{display:flex;align-items:center;gap:8px}.chapters{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.chapters button{text-align:left;min-height:64px;font-size:13px}.chapters button[aria-current=true]{border-color:#d95926}.chapters span{display:block;color:var(--secondary);font:11px ui-monospace,Consolas,monospace;margin-bottom:3px}a{color:var(--ink);text-decoration-color:#68665f;text-underline-offset:4px}footer{border-top:1px solid var(--line);padding-top:17px;margin-top:26px;font-size:12px;color:var(--secondary)}.links{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0}.status{min-height:24px;font-size:12px}h2{font-size:16px;margin:0 0 12px}@media(max-width:700px){main{padding:20px 12px}h1{font-size:23px}.chapters{grid-template-columns:repeat(2,1fr)}header{display:block}.toolbar{gap:9px}video{border-radius:8px}}</style></head>
<body><main><header><div><div class="eyebrow">SGLang / LEARN · VIDEO</div><h1>DeepSeek V4 → V4.1 视频讲解</h1><p>中文旁白 · 画面要点字幕 · ${clock(duration)} · 1920 × 1080 · MP4</p></div><p>直接播放，或点击下方章节跳转。</p></header>
<video id="movie" controls playsinline preload="metadata" poster="${stem}-poster.jpg"><source src="sglang-deepseek-v4.1.mp4" type="video/mp4">当前浏览器无法播放。<a href="sglang-deepseek-v4.1.mp4">直接打开 MP4</a></video>
<div class="toolbar"><label>播放速度 <select id="rate"><option value="0.8">0.8×</option><option value="1" selected>1×</option><option value="1.25">1.25×</option><option value="1.5">1.5×</option><option value="2">2×</option></select></label><button id="restart">从头播放</button><span id="status" class="status" aria-live="polite">就绪 · 点击视频播放按钮开始</span></div>
<h2>章节导航</h2><nav class="chapters" id="chapters" aria-label="视频章节"></nav><div class="links"><a href="sglang-deepseek-v4.1.mp4" download>下载 MP4</a><a href="${stem}.srt" download>下载旁白字幕 SRT</a><a href="sglang-deepseek-v4.1-viz.html">打开交互动画版 ↗</a><a href="sglang-deepseek-v4.1.md">文字笔记 ↗</a></div>
<footer>基于本地代码快照 ${plan.sourceRevision}；旁白由本机中文合成语音生成。节点、时间节奏为教学示意，不代表性能测量。V4 对照原版架构，0731 / 0813 等变体需分别核对。<p>播放页、MP4 和封面位于同一目录，均可离线使用。</p></footer></main>
<script>'use strict';const chapters=${JSON.stringify(chapters)};const movie=document.getElementById('movie'),status=document.getElementById('status'),nav=document.getElementById('chapters');
function clock(t){return Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0')}
chapters.forEach((c,i)=>{const b=document.createElement('button'),time=document.createElement('span');time.textContent=clock(c.start);b.append(time,document.createTextNode(c.label));b.addEventListener('click',()=>{movie.currentTime=c.start;movie.play().catch(()=>{status.textContent='请点击视频中的播放按钮';});});nav.append(b);});
movie.addEventListener('timeupdate',()=>{let current=0;chapters.forEach((c,i)=>{if(movie.currentTime>=c.start-.03)current=i});Array.from(nav.children).forEach((b,i)=>b.setAttribute('aria-current',String(i===current)));status.textContent=clock(movie.currentTime)+' / '+clock(movie.duration||${duration})+' · '+chapters[current].label;});movie.addEventListener('error',()=>{status.textContent='浏览器无法播放，请直接打开下方 MP4 文件。';});document.getElementById('rate').addEventListener('change',e=>{movie.playbackRate=Number(e.target.value)});document.getElementById('restart').addEventListener('click',()=>{movie.currentTime=0;movie.play().catch(()=>{status.textContent='请点击视频中的播放按钮';});});
</script></body></html>`;
}
async function run() {
  console.log('Work directory:', work);
  const voiceDirectory = path.join(work, 'voices');
  const voiceHash = crypto.createHash('sha256').update(JSON.stringify({voice:plan.voice,rate:plan.voiceRate,text:plan.scenes.map(s=>s.say)})).digest('hex');
  const hashFile = path.join(work,'voice-hash.txt');
  if (!fs.existsSync(hashFile) || fs.readFileSync(hashFile,'utf8')!==voiceHash || plan.scenes.some((_,i)=>!fs.existsSync(path.join(voiceDirectory,`voice-${String(i).padStart(2,'0')}.wav`)))) {
    const speechLog=await execute('C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', path.join(__dirname, 'narrate-dsv41.ps1'), '-Plan', planPath, '-OutputDirectory', voiceDirectory]);
    console.log(speechLog.trim()); fs.writeFileSync(hashFile,voiceHash);
  }
  let total=0;
  const scenes=plan.scenes.map((scene,i)=>{
    const wave=path.join(voiceDirectory,`voice-${String(i).padStart(2,'0')}.wav`);
    const duration=Math.ceil((wavSeconds(wave)+1.15)*fps)/fps;
    const result={...scene,wave,duration,start:total}; total+=duration; return result;
  });
  fs.writeFileSync(path.join(work,'timeline.json'),JSON.stringify({duration:total,scenes},null,2));
  console.log('Narration timeline:',total.toFixed(2),'seconds;',scenes.length,'scenes');
  await openBrowser();
  const clips=[];
  const count=preview?1:scenes.length;
  for(let i=0;i<count;i++) {
    const scene=scenes[i];
    await renderScene(scene,i);
    if(i===0){fs.writeFileSync(path.join(work,'preview.png'),await frameAt(.6));fs.writeFileSync(path.join(work,'poster.jpg'),await frameAt(.6,'jpeg'));}
    if(i===13||i===19)fs.writeFileSync(path.join(work,`check-scene-${i}.png`),await frameAt(.6));
    const loop=await animationLoop(i);
    clips.push(await compose(scene,i,loop));
    console.log(`Rendered ${i+1}/${count}: ${scene.title} (${scene.duration.toFixed(2)}s)`);
  }
  if(browserErrors.length)throw new Error('Browser errors: '+browserErrors.join('\n'));
  const list=path.join(work,'concat.txt');
  fs.writeFileSync(list,clips.map((clip,i)=>`file '${clip.replace(/\\/g,'/').replace(/'/g,"'\\''")}'\nduration ${scenes[i].duration.toFixed(6)}\n`).join(''));
  const metadata=path.join(work,'chapters.ffmeta');
  const chapters=chapterList(preview?scenes.slice(0,1):scenes);
  fs.writeFileSync(metadata,';FFMETADATA1\ntitle='+plan.title+'\ncomment=Local Chinese synthesized narration; source '+plan.sourceRevision+'\n'+chapters.map((c,i)=>`[CHAPTER]\nTIMEBASE=1/1000\nSTART=${Math.round(c.start*1000)}\nEND=${Math.round((chapters[i+1]?.start??(preview?scenes[0].duration:total))*1000)}\ntitle=${c.label}\n`).join(''));
  const destination=preview?path.join(work,'preview.mp4'):output;
  await execute(ffmpeg,['-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',list,'-f','ffmetadata','-i',metadata,'-map','0:v:0','-map','0:a:0','-map_metadata','1','-map_chapters','1','-c:v','copy','-c:a','aac','-b:a','112k','-ar','48000','-movflags','+faststart',destination]);
  if(preview){console.log('PREVIEW:',destination);console.log('FRAME:',path.join(work,'preview.png'));return;}
  const srt=scenes.map((s,i)=>`${i+1}\n${timestamp(s.start+.35,true)} --> ${timestamp(s.start+s.duration-.25,true)}\n${s.say}\n`).join('\n');
  const vtt='WEBVTT\n\n'+scenes.map(s=>`${timestamp(s.start+.35)} --> ${timestamp(s.start+s.duration-.25)}\n${s.say}\n`).join('\n');
  fs.writeFileSync(path.join(learn,stem+'.srt'),'\ufeff'+srt);
  fs.writeFileSync(path.join(learn,stem+'.vtt'),vtt);
  fs.writeFileSync(path.join(learn,stem+'-chapters.json'),JSON.stringify({title:plan.title,sourceRevision:plan.sourceRevision,duration:total,chapters},null,2));
  fs.copyFileSync(path.join(work,'poster.jpg'),path.join(learn,stem+'-poster.jpg'));
  fs.writeFileSync(path.join(learn,stem+'.html'),playerHtml(scenes,total));
  console.log('VIDEO:',output);
  console.log('PLAYER:',path.join(learn,stem+'.html'));
  console.log('DURATION:',total.toFixed(3),'seconds; SIZE:',(fs.statSync(output).size/1024/1024).toFixed(2),'MiB');
}
run().catch(error=>{console.error(error.stack);process.exitCode=1;}).finally(async()=>{
  if(socket?.readyState===WebSocket.OPEN){try{await cdp('Browser.close');}catch{}socket.close();}
  if(browser)browser.kill();
  for(const entry of calls.values())clearTimeout(entry.timeout);
});

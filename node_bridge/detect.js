// Puppeteer-based MediaPipe bridge: runs `@mediapipe/tasks-vision` (the WASM
// build the reference engine's Electron CLI uses) inside a real headless Chromium,
// then scrapes landmark results back as JSON. This eliminates the
// WASM-in-Node polyfill nightmare — Chromium is the library's native habitat.
//
// Usage:
//   node detect.js <video> <output.json> [--target-size 1280x720] [--max-frames N]

const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');

const puppeteer = require('puppeteer');

function probeVideo(videoPath) {
  return new Promise((resolve, reject) => {
    const args = ['-v', 'error', '-select_streams', 'v:0',
                  '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
                  '-of', 'json', videoPath];
    const p = spawn('ffprobe', args);
    let out = '';
    p.stdout.on('data', d => out += d);
    p.on('close', code => {
      if (code !== 0) return reject(new Error(`ffprobe exit ${code}`));
      const info = JSON.parse(out).streams[0];
      const [n, d] = info.r_frame_rate.split('/').map(Number);
      resolve({
        width: info.width, height: info.height,
        fps: n / d,
        nbFrames: parseInt(info.nb_frames || 0, 10) || null,
        duration: parseFloat(info.duration || 0),
      });
    });
  });
}

async function main() {
  const argv = process.argv.slice(2);
  if (argv.length < 2) {
    console.error('Usage: node detect.js <video> <output.json> [--target-size WxH] [--max-frames N]');
    process.exit(2);
  }
  const videoPath = path.resolve(argv[0]);
  const outPath = path.resolve(argv[1]);
  let targetW = null, targetH = null, maxFrames = null, preview = false, holistic = false;
  let poseModel = 'full', handConf = 0.5;
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--target-size' && i + 1 < argv.length) {
      const [w, h] = argv[++i].split('x').map(Number);
      targetW = w; targetH = h;
    } else if (argv[i] === '--max-frames' && i + 1 < argv.length) {
      maxFrames = parseInt(argv[++i], 10);
    } else if (argv[i] === '--preview') {
      preview = true;
    } else if (argv[i] === '--holistic') {
      holistic = true;
    } else if (argv[i] === '--pose-model' && i + 1 < argv.length) {
      poseModel = argv[++i] === 'heavy' ? 'heavy' : 'full';
    } else if (argv[i] === '--hand-conf' && i + 1 < argv.length) {
      handConf = parseFloat(argv[++i]);
    }
  }
  const info = await probeVideo(videoPath);
  const W = targetW || info.width;
  const H = targetH || info.height;
  console.error(`[bridge] video ${info.width}x${info.height}@${info.fps.toFixed(2)}fps, ${info.nbFrames} frames`);
  console.error(`[bridge] target ${W}x${H} at src_fps=${info.fps.toFixed(3)}`);

  const modelsDir = path.resolve(__dirname, '..', 'models');
  const readB64 = name => fs.readFileSync(path.join(modelsDir, name)).toString('base64');
  // Holistic = the reference's fast path: ONE combined model (pose+face+hands, internal ROI
  // cropping) instead of separate pose + dual-hand + face + manual crops.
  const holisticTaskB64 = holistic ? readB64('holistic_landmarker.task') : null;
  const poseTaskB64 = holistic ? null : readB64(`pose_landmarker_${poseModel}.task`);
  const handTaskB64 = holistic ? null : readB64('hand_landmarker.task');
  const faceTaskB64 = holistic ? null : readB64('face_landmarker.task');
  console.error(`[bridge] backend model: ${holistic ? 'HolisticLandmarker (1 model, fast)'
    : `pose=${poseModel} + hands(conf ${handConf}) + face (separate)`}`);

  const pageUrl = 'file:///' + path.join(__dirname, 'detect_browser.html').replace(/\\/g, '/');

  // In preview mode launch a CLEAN STANDALONE WINDOW, not a browser: Chromium
  // `--app=<url>` opens a dedicated app window with NO tabs / address bar / toolbar,
  // and we drop the automation infobar. Headless (no preview) needs none of this.
  const winW = (targetW || 1280) + 16, winH = (targetH || 720) + 39;
  console.error(`[bridge] launching ${preview ? 'standalone preview window' : 'headless Chromium'}...`);
  const launchOpts = {
    headless: !preview,
    defaultViewport: preview ? null : undefined,
    args: [
      '--no-sandbox',
      '--disable-web-security',
      '--allow-file-access-from-files',
      '--autoplay-policy=no-user-gesture-required',
      '--enable-features=WebAssemblyExperimentalJSPI',
      ...(preview ? [`--app=${pageUrl}`, `--window-size=${winW},${winH}`, '--disable-infobars'] : []),
    ],
  };
  if (preview) launchOpts.ignoreDefaultArgs = ['--enable-automation'];
  const browser = await puppeteer.launch(launchOpts);

  // With --app the window is already open at pageUrl; grab that page instead of
  // opening a second (tabbed) one.
  let page;
  if (preview) {
    await new Promise(r => setTimeout(r, 400));
    const pages = await browser.pages();
    page = pages.find(p => p.url().startsWith('file:')) || pages[0] || await browser.newPage();
  } else {
    page = await browser.newPage();
  }
  page.on('console', msg => console.error(`  [chromium ${msg.type()}]`, msg.text()));
  page.on('pageerror', e => console.error('  [chromium pageerror]', e.message));

  console.error('[bridge] loading', pageUrl);
  if (!page.url().startsWith('file:')) await page.goto(pageUrl, { waitUntil: 'load' });
  await page.waitForFunction('window.bridgeReady === true', { timeout: 30000 });

  // The video file URL — Chromium can load file:// directly when launched
  // with --disable-web-security (above).
  const videoUrl = 'file:///' + videoPath.replace(/\\/g, '/');
  console.error('[bridge] initializing detectors + loading video...');
  const initRes = await page.evaluate(async (args) => {
    return await window.bridgeInit(args);
  }, { poseTaskB64, handTaskB64, faceTaskB64, holisticTaskB64, videoUrl,
       targetW: W, targetH: H, srcFps: info.fps, preview, handConf });
  console.error('[bridge] init result:', initRes);

  const pick = (res) => ({
    world: res.world, raw: res.raw, vis: res.vis, hands: res.hands,
    hand_crops: res.hand_crops, face: res.face, face_matrix: res.face_matrix,
    face_blendshapes: res.face_blendshapes,
  });
  let frames = [];

  if (preview) {
    // REALTIME: the page plays the video and processes frames live (like the GUI),
    // accumulating records tagged with their media-time frame index. We just wait
    // for it to finish, then densify to a 30fps-indexed frames.json.
    console.error('[bridge] realtime preview running (play the window)…');
    await page.evaluate(async () => await window.bridgeRunRealtime());
    const nFrames = Math.max(1, Math.round((info.duration || 0) * info.fps) || 0);
    let dumped = false;
    while (true) {
      const done = await page.evaluate(() => window.bridgeIsDone());
      const n = await page.evaluate(() => (window.bridgeGetFrames() || []).length);
      process.stderr.write(`\r[bridge] realtime processed ${n} frames`);
      if (!dumped && n >= 250) {           // snapshot the live overlay for verification
        const dataUrl = await page.evaluate(() => window.bridgeDumpCanvas());
        fs.writeFileSync(outPath + '.preview.png', Buffer.from(dataUrl.split(',')[1], 'base64'));
        dumped = true;
      }
      if (done) break;
      await new Promise(r => setTimeout(r, 250));
    }
    process.stderr.write('\n');
    const recs = await page.evaluate(() => window.bridgeGetFrames());
    // densify by idx (realtime may drop frames; fill gaps with the previous record)
    const dense = new Array(Math.max(nFrames, recs.length ? recs[recs.length - 1].idx + 1 : 0)).fill(null);
    for (const r of recs) if (r.idx >= 0 && r.idx < dense.length) dense[r.idx] = pick(r);
    let last = null;
    frames = dense.map(f => { if (f) { last = f; return f; } return last || { world: null, raw: null, hands: {}, face: null, face_matrix: null, face_blendshapes: null }; });
  } else {
    // Deterministic seek loop (accurate, headless).
    let idx = 0;
    while (true) {
      if (maxFrames && idx >= maxFrames) break;
      const res = await page.evaluate(async () => await window.bridgeNext());
      if (res.done) break;
      frames.push(pick(res));
      idx++;
      if (idx === 25) {
        const dataUrl = await page.evaluate(() => window.bridgeDumpCanvas());
        fs.writeFileSync(outPath + '.canvas25.png', Buffer.from(dataUrl.split(',')[1], 'base64'));
      }
      if (idx % 50 === 0) console.error(`[bridge] frame ${idx}`);
    }
  }
  await browser.close();

  fs.writeFileSync(outPath, JSON.stringify({ frame_size: [W, H], frames }));
  const nh = frames.filter(f => f.hands && Object.keys(f.hands).length).length;
  const nf = frames.filter(f => f.face).length;
  console.error(`[bridge] wrote ${outPath}: ${frames.length} frames, ${nh} with hands, ${nf} with face, ${W}x${H}`);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });

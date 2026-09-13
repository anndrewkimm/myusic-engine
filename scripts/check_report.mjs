// Exercise the standalone report in installed Edge using its local debugging protocol.
// node scripts/check_report.mjs data/processed/recommendations/fairy/explorer.html
import {spawn} from 'node:child_process';
import {mkdir, mkdtemp, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const report = path.resolve(process.argv[2]);
const artifacts = path.resolve('artifacts/browser-check');
await mkdir(artifacts, {recursive: true});
const profile = await mkdtemp(path.join(artifacts, 'edge-'));
const browserPath = process.env.MYUSIC_TEST_BROWSER || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const browser = spawn(browserPath, [
  '--headless', '--disable-gpu', '--no-first-run', '--disable-background-networking',
  '--remote-debugging-port=0', '--remote-debugging-address=127.0.0.1',
  `--user-data-dir=${profile}`, 'about:blank',
], {windowsHide: true, stdio: 'ignore'});
let socket;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
try {
  let port;
  for (let i = 0; i < 100; i++) {
    try { port = (await readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; break; }
    catch { await pause(200); }
  }
  assert(port, 'Headless browser did not start');
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method:'PUT'})).json();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {socket.onopen = resolve; socket.onerror = reject;});
  let nextId = 0;
  const pending = new Map(), errors = [], requests = [];
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const entry = pending.get(message.id);
      if (entry) {pending.delete(message.id); message.error ? entry.reject(message.error) : entry.resolve(message.result);}
    }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
    if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') errors.push(message.params.entry.text);
    if (message.method === 'Network.requestWillBeSent') requests.push(message.params.request.url);
  };
  const call = (method, params={}) => new Promise((resolve, reject) => {
    const id = ++nextId; pending.set(id, {resolve, reject}); socket.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const result = await call('Runtime.evaluate', {expression, returnByValue: true});
    assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await call('Page.enable'); await call('Runtime.enable'); await call('Log.enable'); await call('Network.enable');
  await call('Emulation.setDeviceMetricsOverride', {width:1320,height:1000,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate', {url:pathToFileURL(report).href});
  let ready = false;
  for(let i=0;i<100;i++){ready=await evaluate('!!document.getElementById("count")?.textContent');if(ready)break;await pause(100);}
  assert(ready, 'Report did not render');
  const count = await evaluate('document.querySelectorAll("#rows tr").length');
  assert(count > 0, 'Expected recommendation rows');
  await evaluate('document.getElementById("search").value="zzzz-no-such-track-zzzz";document.getElementById("search").dispatchEvent(new Event("input"))');
  assert.equal(await evaluate('document.querySelectorAll("#rows tr").length'), 0);
  await evaluate('document.getElementById("search").value="";document.getElementById("search").dispatchEvent(new Event("input"))');
  assert.equal(await evaluate('document.querySelectorAll("#rows tr").length'), count);
  const image = await call('Page.captureScreenshot', {format:'png', captureBeyondViewport:true});
  await writeFile(path.join(artifacts,'report-desktop.png'), Buffer.from(image.data,'base64'));
  const points = await evaluate('document.querySelectorAll("#map circle").length');
  if(points){await evaluate('document.querySelector("#map circle").dispatchEvent(new MouseEvent("click"))');assert.equal(await evaluate('document.querySelectorAll("#rows tr").length'),1);}
  await call('Emulation.setDeviceMetricsOverride', {width:390,height:844,deviceScaleFactor:1,mobile:true});
  assert(await evaluate('document.documentElement.scrollWidth <= 390'), 'Mobile page overflows viewport');
  const mobile = await call('Page.captureScreenshot', {format:'png', captureBeyondViewport:true});
  await writeFile(path.join(artifacts,'report-mobile.png'), Buffer.from(mobile.data,'base64'));
  assert.deepEqual(errors, [], 'Browser console errors');
  assert(requests.every(url => url.startsWith('file:') || url.startsWith('data:')), 'Unexpected remote request');
  console.log(JSON.stringify({rows:count,mapPoints:points,search:'passed',mobile:'passed',errors,remoteRequests:0}));
} finally {
  socket?.close();
  browser.kill();
}

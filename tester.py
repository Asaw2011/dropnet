"""Record a drop video in the browser (or drop in a phone slo-mo) and see what the model says.

  python tester.py          then open http://localhost:8765

Videos get saved to data/my_videos/ named by what really happened, so they can be
rerun with test_video.py or used for finetune_real.py later.
"""
import glob
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import onnxruntime as ort

from test_video import MIN_DETECT, OUT, TRUTH, read_video, score, write_marked, write_peak
from synth import T

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE = f"{ROOT}/data/my_videos"
LOG = f"{SAVE}/results.jsonl"
PORT = 8765
sess = ort.InferenceSession(f"{ROOT}/runs/dropnet.onnx", providers=["CPUExecutionProvider"])


def to_mp4(src, dst, fps):
    # browser webm has junk timestamps and iphone .mov is hevc, so normalize everything.
    # long side capped at 1280, keeps enough pixels on the eye when the video is a whole face
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-vf", "scale='if(gt(iw,ih),min(1280,iw),-2)':'if(gt(iw,ih),-2,min(1280,ih))'",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", dst]
    subprocess.run(cmd, check=True)


def find_vid(vid_id):
    hits = [p for p in glob.glob(f"{SAVE}/*_{vid_id}.mp4") if not p.endswith("_marked.mp4")]
    return hits[0] if hits else None


def upload(raw, ext, fps):
    os.makedirs(SAVE, exist_ok=True)
    vid_id = time.strftime("%Y%m%d_%H%M%S")
    src = f"{SAVE}/pending_{vid_id}_orig.{ext}"
    open(src, "wb").write(raw)
    to_mp4(src, f"{SAVE}/pending_{vid_id}.mp4", fps)
    return info(vid_id)


def info(vid_id):
    cap = cv2.VideoCapture(find_vid(vid_id))
    return {"id": vid_id, "fps": cap.get(cv2.CAP_PROP_FPS), "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}


def run_test(vid_id, label, eye):
    os.makedirs(OUT, exist_ok=True)
    path = find_vid(vid_id)
    if path is None:
        return {"error": "lost that video, upload it again"}
    old = os.path.basename(path)[:-4]
    name = f"{label}_{vid_id}"
    if old != name:
        # relabel: move the video + original, drop stale outputs
        for p in glob.glob(f"{SAVE}/{old}*"):
            os.rename(p, p.replace(f"{SAVE}/{old}", f"{SAVE}/{name}"))
        for p in glob.glob(f"{OUT}/{old}_*"):
            os.remove(p)
        path = f"{SAVE}/{name}.mp4"

    frames, full, used_fps = read_video(path, None, eye)
    p, starts = score(sess, frames)
    if len(p) == 0:
        return {"error": f"too short, need at least {T} frames"}
    peak = float(p.max())
    detected = peak >= MIN_DETECT
    when = starts[int(p.argmax())] / used_fps
    truth = TRUTH.get(label)
    write_marked(f"{OUT}/{name}_marked.mp4", full, p, used_fps)
    write_peak(f"{OUT}/{name}_peak.png", frames, p,
               f"{name}: {'DROP DETECTED' if detected else 'not detected'}, peak p={peak:.2f} at {when:.2f}s")
    row = {"video": name, "truth": truth, "verdict": "detected" if detected else "not detected",
           "peak_p": round(peak, 3), "at_s": round(when, 2), "fps": round(used_fps, 1),
           "correct": None if truth is None else detected == (truth == "went in"),
           "eye": [round(v, 4) for v in eye] if eye else None}
    # one row per video, a re-test replaces the old one
    rows = [r for r in history()[::-1] if r["video"].split("_", 1)[1] != vid_id] + [row]
    with open(LOG, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return {**row, "curve": [round(float(v), 3) for v in p]}


def history():
    if not os.path.exists(LOG):
        return []
    return [json.loads(l) for l in open(LOG) if l.strip()][::-1]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj):
        self.send(200, json.dumps(obj).encode(), "application/json")

    def send_file(self, path, ctype):
        # browsers won't seek (and safari won't even play) video without range requests
        data = open(path, "rb").read()
        rng = self.headers.get("Range", "")
        if not rng.startswith("bytes="):
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        a, _, b = rng[6:].split(",")[0].partition("-")
        start = int(a) if a else max(0, len(data) - int(b))
        end = min(int(b), len(data) - 1) if a and b else len(data) - 1
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start:end + 1])

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            self.send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif u.path == "/history":
            self.json(history())
        elif u.path == "/info":
            self.json(info(q["id"]) if find_vid(q.get("id", "")) else {"error": "not found"})
        elif u.path.startswith("/vid/"):
            f = find_vid(os.path.basename(u.path))
            if not f:
                return self.send(404, b"nope", "text/plain")
            self.send_file(f, "video/mp4")
        elif u.path.startswith("/out/"):
            f = os.path.join(OUT, os.path.basename(u.path))
            if not os.path.exists(f):
                return self.send(404, b"nope", "text/plain")
            self.send_file(f, "video/mp4" if f.endswith(".mp4") else "image/png")
        else:
            self.send(404, b"nope", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/upload":
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            ext = "".join(c for c in q.get("ext", "webm") if c.isalnum())[:5] or "webm"
            fps = float(q["fps"]) if q.get("fps") else None
            try:
                self.json(upload(raw, ext, fps))
            except subprocess.CalledProcessError:
                self.json({"error": "couldn't decode that video"})
        elif u.path == "/run":
            label = q.get("label", "unknown")
            if label not in TRUTH:
                label = "unknown"
            eye = [float(v) for v in q["eye"].split(",")] if q.get("eye") else None
            self.json(run_test(q.get("id", ""), label, eye))
        else:
            self.send(404, b"nope", "text/plain")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DropNet Tester</title>
<style>
:root{--bg:#F7F9FC;--card:#fff;--ink:#0E1A33;--mut:#5A6478;--line:#DDE3EE;--blue:#2F5FD0;--green:#1F7A4D;--amber:#B45309;--red:#B42318}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 -apple-system,Inter,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:26px;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 20px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}
h2{font-size:17px;margin:0 0 12px}
.cam{position:relative;background:#000;border-radius:10px;overflow:hidden;aspect-ratio:4/3}
.cam video{width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}
.rec{position:absolute;top:10px;left:10px;background:var(--red);color:#fff;font-weight:600;font-size:13px;padding:3px 9px;border-radius:20px;display:none}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
button,label.btn{border:0;border-radius:9px;padding:10px 16px;font-family:inherit;font-weight:600;font-size:15px;cursor:pointer;background:var(--blue);color:#fff}
button.sec,label.btn{background:#E8EDF7;color:var(--ink)}button:disabled{opacity:.45;cursor:default}
button.stop{background:var(--red)}button.mini{padding:4px 10px;font-size:13px;background:#E8EDF7;color:var(--ink)}
.seg{display:flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;width:max-content;max-width:100%}
.seg input{display:none}.seg span{display:block;padding:8px 12px;cursor:pointer;font-size:14px;border-right:1px solid var(--line)}
.seg label:last-child span{border-right:0}.seg input:checked+span{background:var(--ink);color:#fff}
.small{font-size:13px;color:var(--mut)}
#pick{display:none;margin-top:20px}
.stage{position:relative;margin:0 auto;background:#000;border-radius:10px;overflow:hidden;touch-action:none}
.stage video{display:block;width:100%;height:100%}
.stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
.verdict{font-size:30px;font-weight:700;margin:4px 0}.det{color:var(--green)}.nod{color:var(--amber)}
.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:13px;font-weight:600}
.ok{background:#E3F4EA;color:var(--green)}.bad{background:#FDE8E6;color:var(--red)}
#result video,#result img{width:100%;border-radius:8px;margin-top:10px;display:block}
#curve{width:100%;height:90px;display:block;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:7px 6px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:600}
</style></head><body><main>
<h1>DropNet tester</h1>
<p class="sub">Record a drop (saline or artificial tears only), box the eye, say what really happened, and see if the network agrees.</p>
<div class="grid">
<section class="card">
  <h2>1. Record or upload</h2>
  <div class="cam"><video id="live" autoplay muted playsinline></video><div class="rec" id="recdot">REC</div></div>
  <div class="row">
    <button id="camBtn" class="sec">Turn on camera</button>
    <button id="recBtn" disabled>Record</button>
    <label class="btn">Upload video<input type="file" id="file" accept="video/*" hidden></label>
  </div>
  <p class="small" id="camInfo">A laptop webcam is only 30 fps, so a phone slow-mo video (AirDrop it here and upload) works much better.</p>
</section>
<section class="card" id="result"><h2>3. What the network says</h2><p class="small">Nothing yet. Record or upload a video.</p></section>
</div>
<section class="card" id="pick">
  <h2>2. Box the eye, then run it</h2>
  <p class="small" style="margin-top:0">Pause on a frame right before the drop where the eye is open, then drag a box from one corner of the eye to the other. It follows the eye through the rest of the video.</p>
  <div class="stage" id="stage"><video id="pv" playsinline muted controls></video><canvas id="ov"></canvas></div>
  <div class="row" style="justify-content:space-between">
    <div class="seg" id="truth">
      <label><input type="radio" name="t" value="in" checked><span>Went in</span></label>
      <label><input type="radio" name="t" value="miss"><span>Missed</span></label>
      <label><input type="radio" name="t" value="none"><span>No drop</span></label>
      <label><input type="radio" name="t" value="unknown"><span>Not sure</span></label>
    </div>
    <div class="row" style="margin:0">
      <button class="sec" id="playBtn">Play / pause</button>
      <button class="sec" id="clearBtn">Clear box</button>
      <button id="runBtn" disabled>Run test</button>
    </div>
  </div>
  <p class="small" id="boxInfo">Drag on the video to draw the box. The video controls are along the bottom edge.</p>
</section>
<section class="card" style="margin-top:20px"><h2>All your tests</h2><div id="hist" class="small">None yet.</div></section>
</main>
<script>
const $=id=>document.getElementById(id);
let stream=null,rec=null,chunks=[],camFps=null,cur=null,box=null,drag=null;
$('camBtn').onclick=async()=>{
  try{stream=await navigator.mediaDevices.getUserMedia({video:{frameRate:{ideal:60},width:{ideal:1280},height:{ideal:960}},audio:false})}
  catch(e){$('camInfo').textContent='Camera blocked or not available here ('+e.name+'). You can still upload a video.';return}
  $('live').srcObject=stream;camFps=stream.getVideoTracks()[0].getSettings().frameRate||30;
  $('camInfo').textContent='Camera on at '+Math.round(camFps)+' fps. Get close, hit Record, give the drop, hit Stop.';
  $('recBtn').disabled=false;$('camBtn').disabled=true};
$('recBtn').onclick=()=>{
  if(rec&&rec.state==='recording'){rec.stop();return}
  chunks=[];const mime=['video/webm;codecs=vp9','video/webm','video/mp4'].find(m=>MediaRecorder.isTypeSupported(m))||'';
  rec=new MediaRecorder(stream,mime?{mimeType:mime,videoBitsPerSecond:8e6}:{});
  rec.ondataavailable=e=>e.data.size&&chunks.push(e.data);
  rec.onstop=()=>{$('recdot').style.display='none';$('recBtn').textContent='Record';$('recBtn').classList.remove('stop');
    upload(new Blob(chunks,{type:rec.mimeType}),rec.mimeType.includes('mp4')?'mp4':'webm',Math.round(camFps))};
  rec.start(100);$('recdot').style.display='block';$('recBtn').textContent='Stop';$('recBtn').classList.add('stop')};
$('file').onchange=e=>{const f=e.target.files[0];if(!f)return;upload(f,(f.name.split('.').pop()||'mov').toLowerCase(),'');e.target.value=''};
async function upload(blob,ext,fps){
  $('camInfo').textContent='Uploading and converting...';
  const r=await (await fetch(`/upload?ext=${ext}&fps=${fps}`,{method:'POST',body:blob})).json();
  if(r.error){$('camInfo').textContent=r.error;return}
  $('camInfo').textContent='Got it. Now box the eye below.';openPick(r,'in')}
function openPick(v,label){
  cur=v;box=null;$('runBtn').disabled=true;$('pick').style.display='block';
  document.querySelector(`input[name=t][value=${label}]`).checked=true;
  const st=$('stage'),maxH=Math.min(560,innerHeight*0.7),wAvail=st.parentElement.clientWidth-36;
  let w=Math.min(wAvail,maxH*v.w/v.h);st.style.width=w+'px';st.style.height=(w*v.h/v.w)+'px';
  const c=$('ov');c.width=v.w;c.height=v.h;draw();
  $('pv').src='/vid/'+v.id+'?'+Date.now();$('pv').pause();
  $('pick').scrollIntoView({behavior:'smooth',block:'start'})}
function draw(){const c=$('ov'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);if(!box)return;
  const [x,y,w,h]=[box.x*c.width,box.y*c.height,box.w*c.width,box.h*c.height];
  g.fillStyle='rgba(0,0,0,.35)';g.fillRect(0,0,c.width,c.height);g.clearRect(x,y,w,h);
  g.strokeStyle='#6FD3A0';g.lineWidth=Math.max(2,c.width/300);g.strokeRect(x,y,w,h)}
const ov=$('ov');
function pt(e){const r=ov.getBoundingClientRect();return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]}
ov.onpointerdown=e=>{const [x,y]=pt(e);if(y>0.9)return;$('pv').pause();drag=[x,y];ov.setPointerCapture(e.pointerId)};
ov.onpointermove=e=>{if(!drag)return;const [x,y]=pt(e);
  box={x:Math.max(0,Math.min(drag[0],x)),y:Math.max(0,Math.min(drag[1],y)),w:Math.abs(x-drag[0]),h:Math.abs(y-drag[1])};draw()};
ov.onpointerup=()=>{drag=null;if(box&&box.w>0.01&&box.h>0.005){$('runBtn').disabled=false;
  $('boxInfo').textContent='Box set at '+$('pv').currentTime.toFixed(2)+'s. Pick what really happened and hit Run test.'}else{box=null;draw()}};
// clicks near the bottom go to the video controls
ov.addEventListener('pointerdown',e=>{const [,y]=pt(e);if(y>0.9){ov.style.pointerEvents='none';setTimeout(()=>ov.style.pointerEvents='',1500)}});
$('playBtn').onclick=()=>{const v=$('pv');v.paused?v.play():v.pause()};
$('clearBtn').onclick=()=>{box=null;draw();$('runBtn').disabled=true};
$('runBtn').onclick=async()=>{
  const t=document.querySelector('input[name=t]:checked').value,v=$('pv');v.pause();
  const eye=[v.currentTime,box.x,box.y,box.w,box.h].map(n=>n.toFixed(4)).join(',');
  $('runBtn').disabled=true;$('result').innerHTML='<h2>3. What the network says</h2><p class="small">Testing it, a few seconds...</p>';
  $('result').scrollIntoView({behavior:'smooth',block:'start'});
  try{show(await (await fetch(`/run?id=${cur.id}&label=${t}&eye=${eye}`,{method:'POST'})).json())}
  catch(e){$('result').innerHTML='<h2>3. What the network says</h2><p>Something broke: '+e+'</p>'}
  $('runBtn').disabled=false;loadHist()};
function show(r){
  const el=$('result');
  if(r.error){el.innerHTML='<h2>3. What the network says</h2><p>'+r.error+'</p>';return}
  const det=r.verdict==='detected';
  const judge=r.correct===null?'':`<span class="pill ${r.correct?'ok':'bad'}">${r.correct?'Right':'Wrong'} (really: ${r.truth})</span>`;
  el.innerHTML=`<h2>3. What the network says</h2>
   <div class="verdict ${det?'det':'nod'}">${det?'Drop detected':'Drop not detected'}</div>
   <div>${judge} <span class="small">highest score ${r.peak_p.toFixed(2)} at ${r.at_s}s, needs 0.70 to count</span></div>
   <canvas id="curve" width="1000" height="180"></canvas>
   <p class="small" style="margin:8px 0 0">What the network saw, with its score on each frame:</p>
   <video src="/out/${r.video}_marked.mp4?${Date.now()}" controls autoplay loop muted playsinline></video>
   <p class="small" style="margin:10px 0 0">The 12 frames it was most sure about:</p>
   <img src="/out/${r.video}_peak.png?${Date.now()}">`;
  const c=$('curve'),g=c.getContext('2d'),p=r.curve,W=c.width,Hh=c.height;
  const y=v=>Hh-10-v*(Hh-20);
  g.strokeStyle='#B45309';g.setLineDash([8,6]);g.beginPath();g.moveTo(0,y(.7));g.lineTo(W,y(.7));g.stroke();g.setLineDash([]);
  g.fillStyle='#5A6478';g.font='22px sans-serif';g.fillText('0.70 cutoff',8,y(.7)-6);
  g.strokeStyle='#2F5FD0';g.lineWidth=3;g.beginPath();p.forEach((v,i)=>{const x=i/(Math.max(1,p.length-1))*W;i?g.lineTo(x,y(v)):g.moveTo(x,y(v))});g.stroke()}
async function retest(name){
  const id=name.split('_').slice(1).join('_'),label=name.split('_')[0];
  const v=await (await fetch('/info?id='+id)).json();if(v.error)return alert(v.error);
  openPick(v,['in','miss','none'].includes(label)?label:'unknown')}
async function loadHist(){
  const h=await (await fetch('/history')).json();if(!h.length)return;
  const scored=h.filter(r=>r.correct!==null),right=scored.filter(r=>r.correct).length;
  $('hist').innerHTML=(scored.length?`<p style="color:var(--ink);font-size:15px;margin:0 0 8px"><b>${right} of ${scored.length}</b> right so far</p>`:'')+
   '<table><tr><th>Video</th><th>Really</th><th>Network said</th><th>Score</th><th></th><th></th></tr>'+
   h.map(r=>`<tr><td>${r.video}</td><td>${r.truth||'?'}</td><td>${r.verdict}${r.eye?'':' <span class="small">(no eye box)</span>'}</td><td>${r.peak_p.toFixed(2)}</td><td>${r.correct===null?'':r.correct?'<span class="pill ok">Right</span>':'<span class="pill bad">Wrong</span>'}</td><td><button class="mini" onclick="retest('${r.video}')">Re-test</button></td></tr>`).join('')+'</table>'}
loadHist();
</script></body></html>"""


if __name__ == "__main__":
    print(f"open http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

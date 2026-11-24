# VIDEO SERVER v1 (runs inside a Kaggle notebook, 2x Tesla T4). Public file: contains no secrets.
import os, sys, json, re, time, secrets, subprocess, threading, urllib.request, traceback, queue

TOPIC = "cvideo-e19fb1c63aaab4d899fb8061"
CMD_TOPIC = TOPIC + "-cmd"
API_KEY = secrets.token_hex(16)
PORT = 8002
FAKE = os.environ.get("VIDEO_FAKE") == "1"          # local test without GPU or models
WORK = os.environ.get("VIDEO_WORK", "/kaggle/temp")
OUT = WORK + "/out"
LOG = WORK + "/install.log"
TUNLOG = WORK + "/tunnel.log"
RUN_ID = secrets.token_hex(4)
T_START = int(time.time())
IDLE_LIMIT = 2 * 3600       # nobody made a video for 2 hours: stop by itself, so the weekly GPU quota is not burned for nothing
MODELS = {
    "CogVideoX-2B": {"repo": "THUDM/CogVideoX-2b", "frames": 49, "fps": 8, "w": 720, "h": 480, "steps": 40, "cfg": 6.0},
}
DEFAULT_MODEL = "CogVideoX-2B"
STATE = {"stage": "start", "pct": 0, "msg": "", "url": "", "key": "", "verified": False, "v": 1, "run": RUN_ID,
         "models": list(MODELS.keys()), "gpus": 0}
os.makedirs(OUT, exist_ok=True)
os.environ["PATH"] = "/root/.local/bin:/usr/local/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = WORK + "/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def notify(stage, pct, msg, url="", key="", verified=False):
    STATE.update(stage=stage, pct=pct, msg=msg, url=url, key=key, verified=verified)
    print("[%s%%] %s: %s" % (pct, stage, msg), flush=True)
    if FAKE:
        return
    try:
        req = urllib.request.Request("https://ntfy.sh/" + TOPIC, data=json.dumps(STATE).encode("utf-8"), method="POST")
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        print("notify failed:", e, flush=True)


def sh(cmd, log=LOG):
    with open(log, "ab") as f:
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode


def tail(path, n=30):
    try:
        with open(path, errors="ignore") as f:
            return "".join(f.readlines()[-n:])[-1800:]
    except Exception:
        return ""


def read_all(path):
    try:
        with open(path, errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def fail(msg, extra=""):
    notify("error", 0, (msg + " " + extra).strip()[:3000])
    raise SystemExit(msg)


# ---------- stop command from the app ----------
_last_cmd = [0.0]
STOPPING = [False]


def stop_requested():
    if STOPPING[0]:
        return True
    if FAKE or time.time() - _last_cmd[0] < 15:
        return False
    _last_cmd[0] = time.time()
    try:
        raw = urllib.request.urlopen("https://ntfy.sh/%s/json?poll=1&since=%d" % (CMD_TOPIC, T_START - 30), timeout=15).read().decode("utf-8", "ignore")
    except Exception:
        return False
    for line in raw.splitlines():
        try:
            ev = json.loads(line)
            if ev.get("event") != "message":
                continue
            m = json.loads(ev.get("message", "{}"))
        except Exception:
            continue
        if m.get("cmd") == "stop" and (m.get("run") == RUN_ID or m.get("key") == API_KEY):
            STOPPING[0] = True
            return True
    return False


tun = None


def stop_now(msg="Itinigil mo ang server. Hindi na ito gumagamit ng GPU quota."):
    notify("stopped", 0, msg)
    try:
        if tun is not None:
            tun.terminate()
    except Exception:
        pass
    subprocess.run("pkill -f cloudflared", shell=True)
    raise SystemExit(msg)


subprocess.run("pkill -f cloudflared", shell=True)
for _f in (TUNLOG,):
    try:
        os.remove(_f)
    except Exception:
        pass

notify("start", 2, "Nagsisimula na")

NGPU = 1
if not FAKE:
    if sh("nvidia-smi -L") != 0:
        fail("Walang GPU. Sa Settings ng notebook, piliin ang Accelerator: GPU T4 x2.")
    _g = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True).stdout
    if "P100" in _g:
        fail("P100 ang GPU at hindi ito tugma. Piliin ang Accelerator: GPU T4 x2.")
    NGPU = max(1, len([l for l in _g.splitlines() if l.strip()]))
STATE["gpus"] = NGPU
if FAKE:
    print("FAKEKEY", API_KEY, flush=True)

# ---------- install ----------
if not FAKE:
    notify("install", 8, "Ini-install ang mga kailangan (diffusers)")
    rc = sh("pip install -q --upgrade diffusers transformers accelerate sentencepiece protobuf imageio imageio-ffmpeg fastapi 'uvicorn[standard]' python-multipart")
    if rc != 0:
        fail("Hindi na-install ang mga package.", tail(LOG))
    if stop_requested():
        stop_now()

import numpy as np

# ---------- job store ----------
JOBS = {}
JLOCK = threading.Lock()
Q = queue.Queue()
READY = [False]
LAST_USE = [time.time()]


class Cancelled(Exception):
    pass


def new_job(prompt, model, steps, seed):
    jid = secrets.token_hex(6)
    j = {"id": jid, "state": "queued", "pct": 0, "msg": "Nasa pila", "prompt": prompt, "model": model, "steps": steps,
         "seed": seed, "cancel": False, "created": time.time(), "started": 0, "finished": 0, "file": "", "error": "", "gpu": -1}
    with JLOCK:
        JOBS[jid] = j
        # forget old finished jobs and their files (keep the newest 60)
        done = sorted([x for x in JOBS.values() if x["state"] in ("done", "error", "cancelled")], key=lambda x: x["finished"])
        for old in done[:-60]:
            try:
                if old["file"]:
                    os.remove(old["file"])
            except Exception:
                pass
            JOBS.pop(old["id"], None)
    Q.put(jid)
    return j


# ---------- models ----------
PIPES = {}
TE = {}
TE_LOCK = threading.Lock()


def load_models():
    import torch
    cfg = MODELS[DEFAULT_MODEL]
    notify("model", 25, "Dina-download ang video model (mga 14 GB)")
    from huggingface_hub import snapshot_download
    path = None
    for attempt in range(4):
        try:
            path = snapshot_download(cfg["repo"], allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"],
                                     ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.ot"])
            break
        except Exception as e:
            print("download retry", attempt, e, flush=True)
            time.sleep(10)
    if path is None:
        fail("Hindi na-download ang model.", tail(LOG))
    if stop_requested():
        stop_now()
    notify("model", 55, "Ikinakarga ang text encoder")
    from transformers import AutoTokenizer, T5EncoderModel
    TE["tok"] = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
    TE["model"] = T5EncoderModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=torch.float16).eval()
    notify("model", 65, "Ikinakarga ang video model sa GPU")
    from diffusers import CogVideoXPipeline
    for gi in range(NGPU):
        pipe = CogVideoXPipeline.from_pretrained(path, text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
        pipe.to("cuda:%d" % gi)
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        PIPES[gi] = pipe
        notify("model", 70 + 6 * (gi + 1), "Nakakarga ang GPU %d" % (gi + 1))


def encode_pair(text, pipe, gi):
    """Prompt and empty negative prompt -> T5 embeddings. The big text encoder (9.5 GB) and the video model (4 GB) do not fit
    together on one 15 GB T4, so the video model steps aside to RAM for a few seconds while the text encoder works."""
    import torch
    tok, te = TE["tok"], TE["model"]
    dev = "cuda:%d" % gi
    ids = tok([text, ""], padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt").input_ids
    with TE_LOCK:
        try:
            pipe.transformer.to("cpu")
            torch.cuda.empty_cache()
            te.to(dev)
            with torch.no_grad():
                emb = te(ids.to(dev))[0]
            emb = emb.to(torch.float16).clone()
        finally:
            te.to("cpu")
            torch.cuda.empty_cache()
            pipe.transformer.to(dev)
    return emb[0:1], emb[1:2]


def save_mp4(frames, path, fps):
    import imageio
    arr = [np.asarray(f) for f in frames]
    w = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                           output_params=["-crf", "20", "-movflags", "+faststart"])
    for a in arr:
        w.append_data(a)
    w.close()


def run_job(j, gi):
    import torch
    cfg = MODELS[j["model"]]
    pipe = PIPES[gi]
    j["msg"] = "Inihahanda ang prompt"
    pe, ne = encode_pair(j["prompt"], pipe, gi)
    steps = int(j["steps"] or cfg["steps"])

    def cb(p, i, t, kw):
        if j["cancel"] or STOPPING[0]:
            raise Cancelled()
        j["pct"] = min(95, int((i + 1) * 95 / steps))
        j["msg"] = "Ginagawa ang video: hakbang %d sa %d" % (i + 1, steps)
        return kw

    gen = torch.Generator(device="cpu").manual_seed(int(j["seed"]))
    frames = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, num_frames=cfg["frames"], num_inference_steps=steps,
                  guidance_scale=cfg["cfg"], height=cfg["h"], width=cfg["w"], generator=gen,
                  callback_on_step_end=cb).frames[0]
    j["msg"] = "Sine-save ang video"
    j["pct"] = 97
    path = OUT + "/" + j["id"] + ".mp4"
    save_mp4(frames, path, cfg["fps"])
    return path


def worker(gi):
    while True:
        jid = Q.get()
        j = JOBS.get(jid)
        if j is None or j["cancel"]:
            if j is not None:
                j.update(state="cancelled", msg="Kinansela", finished=time.time())
            continue
        j.update(state="running", started=time.time(), gpu=gi, msg="Sinisimulan")
        LAST_USE[0] = time.time()
        try:
            if FAKE:
                for i in range(5):
                    if j["cancel"]:
                        raise Cancelled()
                    j["pct"] = (i + 1) * 18
                    time.sleep(0.3)
                path = OUT + "/" + j["id"] + ".mp4"
                open(path, "wb").write(b"\x00\x00\x00\x18ftypmp42" + b"0" * 200)
            else:
                path = run_job(j, gi)
            j.update(state="done", pct=100, msg="Tapos na", file=path, finished=time.time())
        except Cancelled:
            j.update(state="cancelled", msg="Kinansela", finished=time.time())
        except Exception as e:
            err = traceback.format_exc()
            print(err, flush=True)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            j.update(state="error", msg="Pumalpak", error=(str(e) + " | " + err[-700:])[:1200], finished=time.time())
        LAST_USE[0] = time.time()


# ---------- HTTP API ----------
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()


def auth(req):
    if req.headers.get("authorization", "") != "Bearer " + API_KEY:
        raise HTTPException(status_code=401, detail="bad key")


@app.get("/health")
def health():
    if not READY[0]:
        return JSONResponse({"ok": False}, status_code=503)
    return {"ok": True, "v": 1, "gpus": NGPU}


@app.get("/models")
def models(req: Request):
    auth(req)
    return {"models": list(MODELS.keys()), "default": DEFAULT_MODEL}


@app.post("/generate")
async def generate(req: Request):
    auth(req)
    body = await req.json()
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    model = body.get("model") or DEFAULT_MODEL
    if model not in MODELS:
        model = DEFAULT_MODEL
    seed = body.get("seed")
    if seed is None:
        seed = secrets.randbelow(2 ** 31)
    j = new_job(prompt[:2000], model, int(body.get("steps") or 0), int(seed))
    return {"id": j["id"]}


def pub(j):
    pos = 0
    if j["state"] == "queued":
        with Q.mutex:
            lst = list(Q.queue)
        pos = (lst.index(j["id"]) + 1) if j["id"] in lst else 0
    el = (j["finished"] or time.time()) - j["started"] if j["started"] else 0
    return {"id": j["id"], "state": j["state"], "pct": j["pct"], "msg": j["msg"], "error": j["error"],
            "seconds": int(el), "queue": pos, "gpu": j["gpu"]}


@app.get("/status/{jid}")
def status(jid: str, req: Request):
    auth(req)
    j = JOBS.get(jid)
    if j is None:
        raise HTTPException(status_code=404, detail="unknown job")
    LAST_USE[0] = time.time()
    return pub(j)


@app.post("/cancel/{jid}")
def cancel(jid: str, req: Request):
    auth(req)
    j = JOBS.get(jid)
    if j is None:
        raise HTTPException(status_code=404, detail="unknown job")
    j["cancel"] = True
    return {"ok": True}


@app.get("/file/{jid}")
def file(jid: str, req: Request):
    auth(req)
    j = JOBS.get(jid)
    if j is None or j["state"] != "done" or not os.path.exists(j["file"]):
        raise HTTPException(status_code=404, detail="no file")
    LAST_USE[0] = time.time()
    return FileResponse(j["file"], media_type="video/mp4", filename=jid + ".mp4")


import uvicorn
_srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
threading.Thread(target=_srv.run, daemon=True).start()
for _i in range(40):
    time.sleep(0.5)
    try:
        urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3)
    except urllib.error.HTTPError:
        break          # 503 = it answers (models not ready yet)
    except Exception:
        continue
    break

# ---------- models, workers, smoke test ----------
try:
    if not FAKE:
        load_models()
    for _gi in range(NGPU):
        threading.Thread(target=worker, args=(_gi,), daemon=True).start()
    if not FAKE:
        notify("test", 88, "Sinusubukan ang unang maikling video")
        import torch
        t0 = time.time()
        for _gi in range(NGPU):
            pe, ne = encode_pair("a red balloon floating over green hills, sunny day", PIPES[_gi], _gi)
            fr = PIPES[_gi](prompt_embeds=pe, negative_prompt_embeds=ne, num_frames=17, num_inference_steps=8, guidance_scale=6.0,
                            height=480, width=720, generator=torch.Generator(device="cpu").manual_seed(1)).frames[0]
            sd = float(np.asarray(fr[len(fr) // 2]).astype("float32").std())
            print("smoke GPU", _gi, "std", sd, "secs", round(time.time() - t0, 1), flush=True)
            if not (sd > 3.0):
                fail("Itim o sira ang test na video sa GPU %d (std %.2f). Hindi tugma ang model sa T4." % (_gi, sd))
except SystemExit:
    raise
except Exception:
    fail("Pumalpak ang pagkarga ng model.", traceback.format_exc()[-1500:])
if stop_requested():
    stop_now()

# ---------- tunnel ----------
if FAKE:
    url = "http://127.0.0.1:%d" % PORT
else:
    notify("tunnel", 95, "Ginagawa ang online na link")
    _dl = ("for i in 1 2 3 4 5 6 7 8; do curl -fL --connect-timeout 20 --max-time 120 -o /usr/local/bin/cloudflared "
           "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && test -s /usr/local/bin/cloudflared && break; "
           "rm -f /usr/local/bin/cloudflared; sleep 10; done; test -s /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared")
    if sh(_dl) != 0:
        fail("Hindi na-download ang cloudflared.", tail(LOG))
    tun = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://127.0.0.1:%d" % PORT, "--no-autoupdate"],
                           stdout=open(TUNLOG, "ab"), stderr=subprocess.STDOUT)
    url = ""
    for _i in range(120):
        time.sleep(1)
        if stop_requested():
            stop_now()
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", read_all(TUNLOG))
        if m:
            url = m.group(0)
            break
        if tun.poll() is not None:
            fail("Namatay ang tunnel.", tail(TUNLOG))
    if not url:
        fail("Walang link ang tunnel.", tail(TUNLOG))
    time.sleep(3)

READY[0] = True
LAST_USE[0] = time.time()
notify("ready", 100, "Handa na", url, API_KEY, True)

# ---------- keep running ----------
last_beat = time.time()
while True:
    time.sleep(5)
    if stop_requested():
        stop_now()
    if time.time() - LAST_USE[0] > IDLE_LIMIT:
        stop_now("Tumigil ang server dahil 2 oras na walang gumagawa ng video. Pindutin ang Simulan para tumakbo ulit.")
    if not FAKE and tun is not None and tun.poll() is not None:
        fail("Namatay ang tunnel.", tail(TUNLOG))
    if time.time() - last_beat > 120:
        last_beat = time.time()
        notify("ready", 100, "Online", url, API_KEY, True)

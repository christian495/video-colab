import os, sys, json, subprocess, time, urllib.request, platform

TOPIC = "cvideo-e19fb1c63aaab4d899fb8061"
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def sh(cmd, t=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
        return (p.stdout + p.stderr).strip()
    except Exception as e:
        return "ERR " + str(e)[:200]


def notify(msg):
    try:
        req = urllib.request.Request("https://ntfy.sh/" + TOPIC, data=msg.encode("utf-8"), method="POST")
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        print("notify failed:", e, flush=True)


say("=== VIDEO GPU PROBE v0 ===")
say(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
say("\n--- GPU ---")
say(sh("nvidia-smi"))
say("\n--- CPU / RAM / DISK ---")
say(sh("nproc; free -g | head -2; df -h /kaggle/working /kaggle/temp /tmp 2>/dev/null | head -6"))
say("Python " + platform.python_version())

try:
    import torch
    say("\n--- TORCH ---")
    say("torch %s | cuda %s" % (torch.__version__, torch.version.cuda))
    n = torch.cuda.device_count()
    say("GPU count: %d" % n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        say("  %d %s %.1f GB cc %d.%d" % (i, p.name, p.total_memory / 1e9, p.major, p.minor))
    say("bf16 supported: %s | float8 dtype present: %s" % (torch.cuda.is_bf16_supported(), hasattr(torch, "float8_e4m3fn")))
    import time as _t
    a = torch.randn(4096, 4096, device="cuda:0", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda:0", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = _t.time()
    for _ in range(50):
        c = a @ b
    torch.cuda.synchronize()
    dt = _t.time() - t0
    say("fp16 matmul: %.1f TFLOPS" % (50 * 2 * 4096 ** 3 / dt / 1e12))
except Exception as e:
    say("torch check failed: " + repr(e)[:300])

say("\n--- INTERNET ---")
for url in ("https://huggingface.co", "https://github.com", "https://pypi.org", "https://ntfy.sh"):
    say("%s -> %s" % (url, sh("curl -s -m 20 -o /dev/null -w '%{http_code}' " + url)))

say("\n--- HUGGING FACE MODELS ---")
say(sh("pip -q install -U huggingface_hub 2>&1 | tail -1", 300))
try:
    from huggingface_hub import HfApi
    api = HfApi()
    for repo in ("Lightricks/LTX-2.3", "Lightricks/LTX-2.5", "Lightricks/LTX-Video",
                 "Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.2-S2V-14B", "agosh/LTX-2.5-Comfy-GGUF"):
        try:
            info = api.model_info(repo, files_metadata=True)
            big = sorted([((s.size or 0), s.rfilename) for s in info.siblings], reverse=True)[:5]
            say("\n%s | gated=%s | files=%d" % (repo, info.gated, len(info.siblings)))
            for size, name in big:
                say("   %7.2f GB  %s" % (size / 1e9, name))
        except Exception as e:
            say("\n%s -> not accessible: %s" % (repo, repr(e)[:140]))
except Exception as e:
    say("huggingface_hub failed: " + repr(e)[:300])

say("\n=== END PROBE ===")
summary = "\n".join(LINES)
notify(summary[:3800])

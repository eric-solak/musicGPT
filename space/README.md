---
title: small-lm
emoji: 🎸
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# small-lm demo

Live playground for the model in the main repo -- a ~45M-parameter decoder-only
transformer, trained from scratch (no HuggingFace `transformers`/`datasets`),
on Wikipedia with music articles upweighted.

## Deploying this Space (free, ~10 minutes)

1. **Create the Space.** On huggingface.co: New Space -> pick a name -> SDK:
   **Gradio** -> Hardware: **CPU basic** (free). This model is small enough
   that CPU inference is fast (a 150-token generation takes a couple of
   seconds).

2. **Upload the checkpoint to a separate Model repo** (keeps the Space repo
   small and avoids committing a ~180MB binary through plain git):
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload <your-username>/small-lm-checkpoint ../out/ckpt_best.pt
   ```

3. **Set the Space's env var** (Space Settings -> Variables and secrets):
   `CKPT_REPO = <your-username>/small-lm-checkpoint`

4. **Push this folder's contents to the Space repo:**
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cp app.py requirements.txt README.md <space-name>/
   cp ../model.py <space-name>/          # app.py imports this directly
   cd <space-name>
   git add -A && git commit -m "small-lm demo" && git push
   ```

   `model.py` is copied in verbatim because `app.py` imports `GPT` and
   `ModelConfig` from it directly -- the Space is a standalone repo and
   doesn't see the rest of your project.

5. Wait for the build to finish (Space logs show progress), then the demo is
   live at `https://huggingface.co/spaces/<your-username>/<space-name>` --
   link it from your resume/GitHub README.

## Local test before deploying

```bash
cd space
cp ../model.py .
cp ../out/ckpt_best.pt .   # or set CKPT_REPO instead
pip install -r requirements.txt
python app.py
```

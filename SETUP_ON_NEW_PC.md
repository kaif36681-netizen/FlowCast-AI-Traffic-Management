# Setting Up On Your Computer

Read this first if the project has just been given to you on a USB drive.

**Time needed: about 15 minutes, and you need internet for step 3.**

If you're short on time or the internet is unreliable, skip to
[If setup fails](#if-setup-fails) at the bottom — you can present without
installing anything.

---

## Step 1 — Copy the folder off the USB

Copy the whole `FlowCast` folder from the USB drive to your **C: drive**, so you
end up with:

```
C:\FlowCast\flowcast\
```

**Copy it to the hard drive. Don't run it from the USB** — it'll be slow and
sometimes fails on write permissions.

Short paths with no spaces avoid a lot of problems. `C:\FlowCast` is ideal.

---

## Step 2 — Check Python is installed

Open **PowerShell** (press Start, type `powershell`, press Enter) and run:

```powershell
python --version
```

**If you see a version number like `Python 3.12.4`** — good, go to step 3.

**If you see an error, or the Microsoft Store opens** — Python isn't installed.
Get it from [python.org/downloads](https://www.python.org/downloads/) and
**tick the box that says "Add Python to PATH"** during installation. That box is
easy to miss and everything fails without it. Then close PowerShell, open a new
one, and check again.

---

## Step 3 — Create the environment and install the packages

This is the step that needs internet. It downloads roughly 2 GB.

```powershell
cd C:\FlowCast\flowcast
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**You'll know it worked** when `(.venv)` appears at the start of your prompt,
like this:

```
(.venv) PS C:\FlowCast\flowcast>
```

### If the activate line gives a security error

Windows blocks scripts by default. Run this, then try the activate line again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

It only affects this one window and reverts when you close it.

### If `pip install` fails on `torch`

PyTorch is large and sometimes has no version for very new Python releases. You
don't need it — it's only used by the neural network, whose results are already
saved. Install everything else instead:

```powershell
pip install numpy pandas pyarrow PyYAML scipy scikit-learn xgboost joblib matplotlib seaborn streamlit plotly pytest
```

Everything in the presentation still works.

---

## Step 4 — Check it works

```powershell
python demo.py
```

You should see the project story print out over about 30 seconds. **This is what
you'll run in the presentation.**

Then check the dashboard:

```powershell
streamlit run dashboard\app.py
```

Your browser should open at `localhost:8501` with a dark dashboard. Streamlit may
ask for an email the first time — just press **Enter** to skip it.

Press **Ctrl+C** in PowerShell to stop it.

---

## That's it

Everything is already trained. You do **not** need to run
`python src\run_pipeline.py` — that takes 25 minutes and the results are already
saved in the folder.

---

## Before you present

- [ ] PowerShell open, in the folder, `(.venv)` showing on the prompt
- [ ] `python demo.py` run once so you know what appears
- [ ] Dashboard already running in a browser tab, on the **Live prediction** view
- [ ] Slides open in PowerPoint (or the PDF as backup)
- [ ] `GUIDE.md` read properly — that's the one that matters
- [ ] `VIVA_QA.md` skimmed, especially the starred questions

**Have the dashboard already running before you start speaking.** Don't launch
it live — a ten-second wait feels like a minute in front of a room.

---

## If setup fails

You can present with **zero installation**. Open the folder:

```
BACKUP_no_setup_needed\
```

It contains:

| File | What it is |
| --- | --- |
| `FlowCast_Presentation.pdf` | The full slide deck — opens on any computer |
| `demo_output.txt` | Exactly what the live demo prints, as text |
| `*.png` | The charts: daily traffic pattern, congestion heatmap, weather effects |

Open the PDF, present the slides, and read the demo output from the text file
instead of running it. Nobody will know the difference, and it removes every
technical risk from the room.

**Honestly — have this folder open in a second window even if setup worked.**
Live demos fail at the worst moment, and having a fallback ready turns a disaster
into a five-second recovery.

---

## Quick command reference

Every new PowerShell window needs these two lines first:

```powershell
cd C:\FlowCast\flowcast
.\.venv\Scripts\Activate.ps1
```

| Command | What it does | Time |
| --- | --- | --- |
| `python demo.py` | The live demo — safe, can't fail | 30 sec |
| `streamlit run dashboard\app.py` | Opens the dashboard | 10 sec |
| `python -m pytest tests\ -q` | 21 checks, all pass | 3 sec |
| `python src\run_pipeline.py` | Rebuilds everything — **not during a presentation** | 25 min |

---

## Where to start reading

1. **`START_HERE.md`** — one page, what's what
2. **`GUIDE.md`** — the project explained properly. Read this one carefully.
3. **`VIVA_QA.md`** — 40 questions you'll be asked, with answers

If you only have an hour, read `GUIDE.md`. Everything else supports it.

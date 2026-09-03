# FlowCast — Understand Your Own Project

**Read this before you present anything.**

This explains what the project is, how every part works, and what to say when
someone asks. No jargon that isn't explained. If you read this once carefully
you'll be able to answer almost any question about it.

Written for someone who knows basic Python but hasn't done machine learning
before.

---

## Part 1 — What is this, in one page?

There's a busy main road in a city. It's divided into **25 numbered stretches**
(called "segments"). Sensors in the road count vehicles and measure speed, and
they save a reading **every 30 minutes**.

A traffic control room watches this. Their problem: they can see the road as it
is *now*, but by the time they spot a jam it's too late to prevent it.

**FlowCast predicts what the road will look like 30 minutes from now.**

For each of the 25 stretches it predicts four things:

| What it predicts | Type of problem |
| --- | --- |
| How many vehicles will pass | A number → **regression** |
| How fast they'll go (km/h) | A number → **regression** |
| How long to drive it (minutes) | A number → **regression** |
| Traffic level: Free-flow / Moderate / Heavy / Severe | A category → **classification** |
| Chance of an accident | A probability → **classification** |

Two words you'll use constantly:

- **Regression** = predicting a number (how many? how fast?)
- **Classification** = predicting a category (which group does this belong to?)

That's the whole project. Everything else is detail.

---

## Part 2 — Where does the data come from?

Three files sitting in the `data/raw/` folder on your computer. Nothing is
downloaded from the internet.

**`traffic_sensor_log.csv`** — 31 MB, the main file.
One row = one sensor reading. 25 stretches × every 30 minutes × 5 months
(1 Jan – 31 May 2025) = **178,468 rows**.

Each row has: which stretch, date and time, vehicle count, average speed,
how full the road was (occupancy %), travel time, accident count, traffic light
timing, and the road's maximum capacity.

**`weather_observations.csv`** — 3 weather stations, one reading per hour:
temperature, rainfall, visibility.

**`calendar_events.csv`** — one row per day for 151 days, marking public
holidays (6), special events (6) and roadworks (11).

> **If asked "is this real data?"** — Say honestly: it's synthetic data
> generated for the project brief, deliberately made messy to imitate real
> sensor feeds. That's normal and it's not a weakness. The messiness is the
> point — see Part 4.

---

## Part 3 — How the whole thing works

Think of it as a factory line. Raw files go in one end, predictions come out the
other. Running `python src/run_pipeline.py` runs the whole line.

```
    data/raw/*.csv
         |
    [1] INGEST     check the data, throw out impossible readings
         |
    [2] CLEAN      fix duplicates, missing slots, spelling, merge the 3 files
         |
    [3] FEATURES   build "clues" the computer can learn from
         |
    [4] EDA        draw charts, write the data-quality report
         |
    [5] MODELS     learn the patterns, compare 6 different methods
         |
    [6] DEEP       the neural network (an extra, more advanced method)
         |
    [7] DASHBOARD  show the results on screen
```

Each step is one Python file in the `src/` folder, named after the step. If
someone asks "show me the code that does X", you can find it by the name.

---

## Part 4 — What was wrong with the data (this is the important part)

Real sensor data is always broken. Finding and fixing the breakage is most of
the actual work in any data project. Here's what was wrong:

### Problem 1: Impossible readings — 712 rows

Some rows said things like **negative vehicle counts** or **speeds of 300 km/h**.
Those aren't unusual traffic, they're a broken sensor.

**What we did:** moved them to a separate "quarantine" file with a note saying
why, instead of deleting them. That way anyone reviewing the work can check the
decision.

**Why not just fix them to a sensible number?** Because that would be inventing
data. If a sensor reports nonsense, we don't know what actually happened — so we
mark the reading as missing and let step 4 estimate it from its neighbours.

### Problem 2: Duplicates — 1,759 rows

The same stretch, at the same timestamp, recorded twice. This happens when a
sensor retries a failed transmission.

**What we did:** kept one copy — specifically the one with the fewest blank
fields, since a retry sometimes carries more information than the original.

### Problem 3: Missing time slots — 5,203 (the one that matters most)

Some half-hour slots are **completely absent from the file**. Not blank — simply
not there.

**Why this is dangerous.** Suppose the file has 7:00, then 9:30 (8:00, 8:30 and
9:00 are missing). If your code asks for "the previous reading" before 9:30, it
hands you **7:00** — because that's the row sitting above it. Your program now
believes 7:00 was half an hour before 9:30. Every calculation after that is
quietly wrong, and nothing crashes to warn you.

**What we did:** rebuilt the complete timetable first — every stretch, every
half hour, all 181,200 slots — so the gaps become visible empty rows instead of
invisible absences.

> **This is the single best thing to talk about in your presentation.** It's the
> kind of bug that ruins real projects, it's invisible, and catching it shows
> you actually looked at the data instead of just loading it.

### Problem 4: Weather spelled 13 different ways

The weather file contained `Clear`, `clear`, `CLEAR`, `Rain`, `rain`, `rainy`,
`RAIN`, `Fog`, `foggy`, `FOG`, `Cloudy`, `cloudy`, `Overcast`.

To a computer those are 13 completely different things. **We reduced them to 4:**
Clear, Cloudy, Rain, Fog.

One judgement call worth mentioning: `Overcast` appeared 358 times but wasn't in
the project's official list of weather types. We mapped it to `Cloudy` — the
closest official category — and wrote that decision down rather than silently
dropping those rows.

### Problem 5: Two different date formats

The traffic file writes dates as `2025-03-05` (year-month-day).
The weather file writes them as `05/03/2025` (day-month-year).

If you don't handle this, **5 March silently becomes 3 May** for every date up to
the 12th of a month. We told the code explicitly which format each file uses.

### Problem 6: Missing values — about 4,400 per column

Some readings were blank. Two different fixes depending on how long the gap was:

- **Short gap (1–2 slots):** estimate by drawing a straight line between the
  readings either side. Reasonable — traffic doesn't teleport.
- **Long gap (hours):** don't draw a line, because a straight line across three
  hours would erase a rush-hour peak. Instead use the typical value for *that
  stretch, at that time of day, on that day of the week*.

---

## Part 5 — The trap hidden in the data

This is the second thing worth presenting, and it's the one that separates a
real project from a tutorial.

The data has a column called **`congestion_level`** with values Free-flow /
Moderate / Heavy / Severe. It looks like useful information.

It isn't. It turns out to be **nothing more than the vehicle count sorted into
four buckets**:

```
vehicles ÷ road capacity   →   under 50%  = Free-flow
                               50 – 80%   = Moderate
                               80 – 100%  = Heavy
                               over 100%  = Severe
```

We checked this rule against the rows that already had a label. It matched
**99.64%** of the time.

**Why this matters.** If you give the computer the vehicle count and ask it to
predict congestion level, it will score about 99% accuracy — because you handed
it the answer. It looks brilliant. It predicts nothing.

This is called **data leakage**: accidentally giving the model information it
wouldn't have in real life.

**Two more leaks were found in this data:**

- `vehicle_count` is identical to `traffic_volume` in 97.4% of rows — the same
  number in two columns.
- `travel_time` is just distance ÷ speed. So predicting travel time is really
  predicting speed with extra steps.

**How the project prevents all three:** every piece of information the model is
allowed to see must come from **at least 30 minutes before** the moment being
predicted. And there are **5 automated tests** whose only job is to fail loudly
if anyone ever breaks that rule.

---

## Part 6 — How does a computer "learn" this?

### The basic idea

The computer isn't told any rules about traffic. It's shown thousands of
examples of the form:

> "The situation looked like *this*, and 30 minutes later *this* happened."

From enough examples it finds patterns. That's all machine learning is here.

### What are "features"?

A **feature** is one piece of information given to the model — one clue.

The computer knows nothing about the world. It doesn't know rush hour exists.
So we build the clues for it. This project builds **93 features** for every
prediction:

| Type of clue | Examples | Why it helps |
| --- | --- | --- |
| **Time** | hour, day of week, is it a weekend, is it rush hour | Traffic follows the clock |
| **Lags** (recent history) | volume 30 min ago, 1 hour ago, 2 hours ago | The recent past predicts the near future |
| **Yesterday / last week** | same slot yesterday, same slot last Tuesday | Traffic repeats weekly |
| **Rolling averages** | average of last 4 slots, last 8 slots | Smooths out random noise |
| **Change** | is volume rising or falling? | Catches a jam building |
| **Weather** | raining?, low visibility?, temperature | Rain slows traffic |
| **Calendar** | holiday?, event?, roadworks? | Breaks the normal pattern |
| **Combinations** | rain **during** rush hour | Rain at 3am doesn't matter; rain at 6pm does |

Building good features is usually more important than which model you pick.

> **Likely question: "Why sin and cos for the hour?"**
> If you give the computer hour = 23 and hour = 0, it thinks they're 23 apart —
> but they're actually adjacent (11pm and midnight). Sine and cosine wrap the
> clock into a circle so midnight sits right next to 11pm. It's one line of code
> and it genuinely helps.

### Training and testing — the exam analogy

The data covers 1 January to 31 May. It's split into three parts **by date**:

| Part | Dates | Rows | Used for |
| --- | --- | --- | --- |
| **Training** | 8 Jan – 18 Apr | 120,950 | Learning the patterns |
| **Validation** | 19 Apr – 9 May | 25,925 | Tuning settings |
| **Test** | 10 – 31 May | 25,925 | Final marking — never seen before |

The test weeks are **hidden completely** during learning. Then the model is
asked to predict them, and its answers are compared to what really happened.

It's exactly like an exam. If the student sees the paper in advance, a high mark
proves nothing.

> **Very likely question: "Why split by date instead of randomly?"**
>
> This is the question a good mentor asks, so learn this answer.
>
> Random splitting would put 3pm Tuesday in training and 3:30pm Tuesday in
> testing. The model would effectively be predicting a gap it had already seen
> both sides of — which is impossible in real life, because in real life the
> future hasn't happened yet. Splitting by date means the model only ever learns
> from the past and is tested on the future. That's the honest setup.

### The six methods compared

We didn't pick one method. We tried six and compared them fairly on the same
hidden test weeks.

| Method | How it works, in one line |
| --- | --- |
| **Mean baseline** | Always guess the average. Deliberately stupid — a floor. |
| **Persistence** | Assume the next 30 min looks like the last 30 min. Free, no computer. |
| **Linear Regression** | Draw the best straight-line relationship through the data. |
| **Decision Tree** | A flowchart of yes/no questions: "raining? → rush hour? → …" |
| **Random Forest** | Build hundreds of different trees, average their answers. |
| **XGBoost** | Build trees one at a time, each one fixing the previous one's mistakes. |

**Why include the deliberately stupid ones?** Because a score means nothing on
its own. "8.9% error" sounds good — but is it? Only the comparison tells you.

---

## Part 7 — The results

### Predicting traffic volume (the main target)

| Method | Average error | Better than lazy guess? |
| --- | ---: | --- |
| Mean baseline | 83.3% | — |
| **Persistence (lazy guess)** | **16.0%** | the bar to beat |
| Linear Regression (built by hand) | 12.3% | yes |
| Linear Regression (library) | 12.3% | yes |
| Decision Tree | 10.4% | yes |
| Random Forest | 9.3% | yes |
| **XGBoost — the winner** | **8.9%** | **45% less error** |

**What "8.9% error" means in plain terms:** if 200 vehicles actually pass, the
prediction is typically between about 182 and 218.

**The headline sentence to say out loud:**

> "The system predicts traffic volume 30 minutes ahead with 8.9% average error,
> which is 45% more accurate than simply assuming nothing changes."

The second half of that sentence is what makes the first half meaningful.

### The other targets

| Target | Result | Goal | Met? |
| --- | --- | --- | --- |
| Speed | 8.3% error | — | ✅ |
| Travel time | 9.1% error | — | ✅ |
| Congestion level | 0.77 score | 0.80 | ❌ |
| Accident risk | 0.64 score | 0.75 | ❌ |

---

## Part 8 — The two things that didn't work (present these confidently)

Failures presented honestly, with evidence, are worth more than successes
presented vaguely. Don't hide these — lead with them.

### Failure 1: Congestion classification, 0.77 vs a target of 0.80

The score used is **macro-F1**. What it means: it measures how well the model
does on **each of the four categories separately, then averages them — treating
all four as equally important.**

Here's the catch. The four categories are very unbalanced:

| Category | Share of all slots |
| --- | ---: |
| Free-flow | 61% |
| Moderate | 24% |
| Heavy | 9% |
| **Severe** | **5.5%** |

Because Severe is rare, the model sees few examples and does worse on it — and
macro-F1 punishes that heavily by design.

**Is macro-F1 the right metric?** Yes, and say so. Plain accuracy would be
misleading: a model that always guessed "Free-flow" would score 60% accuracy
while being completely useless. And Heavy and Severe are precisely the cases the
control room cares about. So the demanding metric is the correct one.

**The honest position to take:**

> "The model reaches 0.77 against a target of 0.80. Overall accuracy is 87.7%
> and it beats the lazy guess by a wide margin. The gap comes from the rarest
> category. I'd argue the 0.80 target was set without looking at how unbalanced
> the classes are — but I'm reporting the miss rather than switching to a metric
> that would have flattered the result."

### Failure 2: Accident risk, 0.64 vs a target of 0.75 — with proof of *why*

This is the strongest thing in the whole project. Learn it properly.

Accidents are extremely rare: **0.9% of slots** — about 1 in 110.

When a model underperforms there are only two possible reasons, and they need
opposite responses:

1. **The model is bad** → try harder, better settings, better features
2. **The information isn't in the data** → stop, and say so

Most students guess. We tested it.

**The experiment:** build a deliberately **cheating** model — one allowed to see
information from the *exact moment being predicted*, which no real forecast
could ever have. If the cheat helps a lot, the information exists and we're just
extracting it badly. If the cheat barely helps, the information isn't there at
all.

**The result:**

| Model | Score |
| --- | ---: |
| Honest model (past information only) | 0.632 |
| **Cheating model (sees the present)** | **0.638** |
| Gain from cheating | **+0.006** |

Cheating bought essentially nothing. So the ceiling on this dataset is about
0.64 and the 0.75 target is **unreachable — not because of bad modelling, but
because the data doesn't contain what's needed.**

**What would actually be needed:** road layout and lane counts, where crashes
have historically happened, junction and merge locations, and second-by-second
speed variation. A 30-minute average smooths away exactly the sudden braking
that precedes a crash.

**And it's still useful anyway.** Even at 0.64, the *ranking* works:

> **Top-decile lift = 2.64.** If patrols cover only the 10% of road-hours the
> model ranks riskiest, they find **2.6 times as many incidents** as patrolling
> at random. That's real operational value from a "failed" model.

**The sentence to say:**

> "It missed the target, so I tested whether that was my modelling or the data.
> I built a version allowed to cheat and it gained 0.006 — nothing. That tells
> me the data doesn't contain what predicts accidents. The recommendation isn't
> more tuning, it's collecting road-geometry and historical-collision data."

---

## Part 9 — The advanced parts, explained simply

These are the harder pieces. Understand the idea, don't memorise the maths. If
asked something beyond what's here, **say you're not sure** — that's a far better
answer than bluffing.

### Linear regression written from scratch

The project brief required implementing one algorithm by hand rather than just
calling a library, to show the maths isn't a black box.

**The idea:** you're trying to find the best straight-line relationship. Start
with a random guess. Measure how wrong it is. Nudge the guess in the direction
that makes it less wrong. Repeat a few hundred times. This is called
**gradient descent** — think of walking downhill in fog by always stepping in the
steepest downward direction.

**The proof it's correct:** the hand-written version scored **65.81** and the
library version scored **65.80** — matching to two decimal places. There's also
a "gradient check" that verifies the calculus against a numerical approximation,
agreeing to 11 decimal places.

**One bug worth mentioning** (it makes you sound like you actually built it):
the first version diverged to infinity. Vehicle counts run to about 2,000 while
other features sit between −1 and 1, so a step size suitable for one was wildly
too big for the other. Fix: rescale everything to a comparable range first.

### The neural network (LSTM)

**What it is:** a model that reads a *sequence* in order and carries a memory as
it goes. The other models see "volume 30 min ago, 1 hour ago, 2 hours ago" as
three unrelated numbers. An LSTM reads them as a story: *rising steadily for two
hours*.

**What happened:** it scored 55.59 against XGBoost's 55.86. Very slightly better
on one measure, slightly *worse* on another — using 64,269 internal settings and
twelve times the training time.

**The recommendation in the report is: don't use it.** That's a legitimate
engineering conclusion. The simpler model already captures the pattern, and
extra complexity has to earn its place.

> This is a good thing to say. "I built it, benchmarked it fairly, and concluded
> it wasn't worth deploying" is a stronger answer than pretending it was a
> triumph.

### Confidence ranges

Rather than only "we predict 200 vehicles", the system says "we predict 200,
likely between 150 and 250."

**The honesty check:** the range promises to be right 80% of the time. Measured
on the hidden test weeks, it was right **80.5%** of the time. So the promise is
real.

The range also **widens when prediction is harder** — about ±90 vehicles on a
quiet dry night, ±230 during rush hour in rain. The model knows when it's
guessing.

### Calibrated accident probability

**The problem:** because accidents are so rare, the model has to be told to take
the rare cases seriously — otherwise it just learns "predict no accident" and is
right 99% of the time while being useless. But that adjustment distorts the
output numbers: it was reporting **48%** risk on a road where the real rate is
**0.96%**.

**The fix:** a second small adjustment step that squashes the numbers back onto
the real scale, learned from data the model hadn't seen. Result: average
prediction went from 48% to **0.80%**, against a true rate of 0.96%.

**Key point:** the ordering of segments is completely unchanged — only the scale.
So the ranking still works exactly as before, but the numbers now mean what a
reader assumes they mean.

---

## Part 10 — Running it

**Every new terminal window needs these two lines first:**

```powershell
cd C:\FlowCast\flowcast
.\.venv\Scripts\Activate.ps1
```

You'll know it worked when `(.venv)` appears at the start of the prompt.

Then:

| Command | What it does | Time |
| --- | --- | --- |
| `python demo.py` | **Prints the whole story. Use this in the presentation.** | 30 sec |
| `streamlit run dashboard\app.py` | Opens the dashboard in a browser | 10 sec |
| `python -m pytest tests\ -q` | Runs 21 checks, all should pass | 3 sec |
| `python src\run_pipeline.py` | Rebuilds everything from the raw files | ~25 min |

**Do not run the full pipeline live in front of anyone.** Run `demo.py` instead.

---

## Part 11 — Suggested 30-minute structure

| Time | What |
| --- | --- |
| 0–3 min | The problem: control rooms react instead of predicting |
| 3–6 min | The data: three files, 178,468 readings |
| 6–12 min | **What was broken and how I fixed it** — the missing slots especially |
| 12–15 min | **The leakage trap** — congestion was just volume in disguise |
| 15–18 min | How prediction works: features, date-based split |
| 18–22 min | Results, always against the baselines |
| 22–27 min | **The two failures** — and the cheating experiment |
| 27–30 min | Live demo + questions |

Give the most time to the data problems and the failures. Those are what
demonstrate you understood the work rather than just ran it.

---

## Part 12 — Being straight about how it was built

Internmo's published guidelines say submissions must be the student's own
original work, and list work *generated without disclosure* alongside plagiarism.
Note the wording — **disclosed** assistance is treated differently from hidden
assistance.

A short, honest note costs nothing and protects the certificate. Something like:

> **Tools and assistance**
>
> This project was built with substantial AI assistance for the implementation
> code. I used it as a working reference and pair-programmer, and I have worked
> through the reasoning behind each decision — in particular the time-based split,
> the handling of the missing time windows, the data-leakage problem in the
> `congestion_level` column, and the diagnostic showing that the accident-risk
> target is unreachable with this dataset. I'm happy to explain or walk through
> any part of the pipeline.

Then it's declared, and questions become a conversation instead of an exam.

If you have a spare weekend, the strongest possible version is to rebuild the
core yourself using this as a reference — load the CSVs, clean them, build ten
features, train one model, print the score. That's maybe 150 lines, it's
genuinely within reach for a 3rd-year student, and everything else becomes a
documented extension.

---

## Part 13 — The five things to remember

If you remember nothing else:

1. **It predicts traffic 30 minutes ahead** for 25 road segments, four things
   per segment.
2. **8.9% error, which is 45% better than assuming nothing changes.** Always say
   the second half.
3. **5,203 time slots were missing from the file** — invisible, and it would have
   corrupted every calculation.
4. **`congestion_level` was secretly just the vehicle count** — using it would
   have given a fake 99% score.
5. **Accident prediction failed, and I proved why** — a cheating model gained
   only 0.006, so the data simply doesn't contain the answer.

Points 3, 4 and 5 are the ones that will impress. Lead with them.

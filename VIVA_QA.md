# FlowCast — Questions You'll Be Asked

Thirty-two questions a mentor is likely to ask, with answers you can actually
defend. Read the guide first — this assumes you have.

**The most important rule on this page:** if you don't know, say so. "I'm not
sure, I'd have to check" is a completely acceptable answer and costs you far
less than a wrong one delivered confidently. Mentors ask follow-ups. Bluffing
collapses on the second question; honesty doesn't.

Questions marked ⭐ are the ones most likely to come up.

---

## A. The basics

**1. ⭐ What does your project do?**

It predicts traffic conditions 30 minutes ahead for a 25-segment city road. For
each segment it forecasts how many vehicles will pass, how fast they'll go, how
long the journey takes, how congested it'll be, and the chance of an accident.
The point is that a control room currently reacts to jams after they form — this
gives them half an hour of warning so they can adjust signal timings or position
patrols first.

**2. Who would use it?**

Three roles. A traffic operations analyst watching the live view. An incident
response coordinator using the risk ranking to position patrols. A transport
planner looking at historical patterns to justify infrastructure decisions.

**3. Where did the data come from?**

Three CSV files provided with the project brief: five months of sensor readings
at 30-minute intervals, hourly weather from three stations, and a daily calendar
of holidays, events and roadworks. It's synthetic data, deliberately made messy
to imitate real sensor feeds.

**4. How much data?**

178,468 raw sensor readings, covering 25 segments from 1 January to 31 May 2025.
After cleaning, 172,800 usable rows with 93 features each.

**5. What technologies did you use?**

Python. Pandas for the data handling, scikit-learn and XGBoost for the models,
PyTorch for the neural network, Streamlit and Plotly for the dashboard, pytest
for the tests.

---

## B. The data cleaning (spend time here — it's your strongest material)

**6. ⭐ What was wrong with the data?**

Six things. 712 physically impossible readings like negative vehicle counts and
300 km/h speeds. 1,759 duplicate rows. 5,203 time slots missing from the file
entirely. Weather conditions spelled 13 different ways. Two files using
different date formats. And around 4,400 blank values in each sensor column.

**7. ⭐ You keep mentioning the missing time slots. Why do they matter more than
the others?**

Because they're invisible and they corrupt everything downstream silently.

The rows aren't blank — they're absent. So if the file has 7:00 then jumps to
9:30, and my code asks for "the previous reading" before 9:30, it gets 7:00.
The program now believes 7:00 was 30 minutes before 9:30. Every lag feature is
wrong and nothing crashes to tell you.

I rebuilt the complete timetable first — all 25 segments × every half hour ×
151 days = 181,200 slots — so gaps became visible empty rows instead of
invisible absences.

**8. Why quarantine the bad rows instead of deleting them?**

So the decision is auditable. They're written to a separate file with the reason
attached. If a reviewer disagrees with my threshold they can see exactly what was
removed and why. Deleting rows silently is how you lose the ability to defend
your own pipeline.

**9. Why didn't you just correct the impossible values to sensible numbers?**

That would be inventing data. If a sensor reports −40 vehicles I don't know what
actually happened, so replacing it with a plausible-looking number creates a
fake reading that looks real. I mark it missing and let the imputation step
estimate it from surrounding readings — which is at least an honest estimate
rather than a fabrication.

**10. How did you fill the missing values?**

Two ways, depending on gap length. Short gaps of one or two slots get linear
interpolation — the readings either side genuinely constrain what's between them.
Long gaps don't, because a straight line across three hours would flatten a
rush-hour peak. Those use the median for that segment, at that time of day, on
that day of the week.

**11. Which duplicate did you keep?**

The most complete one. I scored each row by how many fields weren't blank and
kept the highest, because a retransmission sometimes carries more data than the
original attempt.

**12. What was the `Overcast` problem?**

The weather file had `Overcast` 358 times, but the project's data dictionary only
defined four conditions and Overcast wasn't among them. I mapped it to `Cloudy`
as the closest defined category, and logged that decision rather than dropping
the rows. It's a judgement call and I'd rather it be visible than hidden.

---

## C. Data leakage (the second strongest section)

**13. ⭐ What is data leakage?**

Accidentally giving the model information it wouldn't have in real life. It makes
scores look excellent while the model is actually useless, because in deployment
that information won't exist.

**14. ⭐ What leakage did you find?**

Three cases. `congestion_level` turned out to be just the vehicle count sorted
into four buckets — I verified the rule reproduces the existing labels 99.64% of
the time. `vehicle_count` was identical to `traffic_volume` in 97.4% of rows.
And `travel_time` is distance divided by speed, so predicting it is really
predicting speed.

**15. How do you prevent it?**

Structurally, not by remembering to be careful. Every dynamic feature is shifted
back by at least one 30-minute window inside its own segment. The list of
allowed features is stored in a separate file that acts as a contract. And five
of the 21 automated tests exist purely to fail if anyone ever breaks the rule —
including one that fails if any feature correlates above 0.99 with the target.

**16. If congestion is just volume in buckets, why predict it at all?**

Because the *future* value isn't derivable from anything I'm allowed to see. I'm
predicting congestion 30 minutes from now using only information from before
now. That's a genuine forecasting problem. What would be cheating is using the
current volume to "predict" the current congestion level.

**17. Why do you use last week's weather instead of the actual weather?**

I use the previous 30-minute window's weather, not the target window's. In real
deployment you'd have a weather forecast, not a weather observation, for a time
that hasn't happened. Using the actual observed weather would be a mild form of
leakage. It's the conservative choice.

---

## D. The modelling

**18. ⭐ Why did you split the data by date instead of randomly?**

Because random splitting leaks the future into the past. If 3:00pm Tuesday goes
into training and 3:30pm Tuesday into testing, the model has effectively seen
both sides of the gap it's being asked to predict — which is impossible in real
life. Splitting by date means the model learns only from the past and is tested
only on the future. Training is January to mid-April, validation is three weeks,
and the final three weeks of May are held back completely.

**19. What's the difference between validation and test?**

Validation is for tuning — comparing settings and choosing between options. Test
is used exactly once, at the end, for the final number. If I tuned against the
test set, the score would reflect how well I'd fitted my own scoring data rather
than genuine performance.

**20. ⭐ Which model won and why?**

XGBoost, on every regression target. It reached 8.9% average error on traffic
volume against Random Forest's 9.3% and Decision Tree's 10.4%. It works by
building small decision trees one after another, where each new tree focuses on
correcting the errors the previous ones made.

**21. ⭐ Why include a "mean baseline" and "persistence" if they're obviously bad?**

Because a score is meaningless without a comparison. "8.9% error" tells you
nothing on its own. Persistence — just assuming the next half hour looks like
the last — gets 16%. That's free and needs no computer. The real result is that
XGBoost cuts the error by 45% *against that*. Without the baseline I'd be
reporting a number nobody could interpret.

**22. What is a "feature" and how many do you have?**

A feature is one piece of information given to the model as a clue. I have 93,
built from 17 raw columns: time-of-day encodings, recent history at various
lags, rolling averages, weather flags, calendar flags, and combinations like
rain-during-rush-hour.

**23. Why sine and cosine for the hour?**

If you feed the raw hour number, the model sees 23 and 0 as 23 units apart, when
11pm and midnight are actually adjacent. Sine and cosine map the clock onto a
circle so midnight sits next to 11pm.

**24. What does 8.9% MAPE actually mean?**

Mean Absolute Percentage Error. If 200 vehicles actually pass, the prediction is
typically off by about 18 — so somewhere around 182 to 218.

**25. Why RMSE for one thing and F1 for another?**

They measure different kinds of problem. RMSE is for predicting numbers — it's
the typical size of the error, and it punishes large errors more than small ones.
F1 is for predicting categories, balancing how often the model is right when it
says "Severe" against how many actual Severe cases it catches.

---

## E. The failures (rehearse these — they're your best material)

**26. ⭐ Two targets missed. Why?**

Different reasons, and that distinction matters.

Congestion missed by 0.03 because of how the metric works with very unbalanced
categories — Severe is only 5.5% of slots, and macro-F1 weights it equally with
Free-flow at 61%. Overall accuracy is 87.7%.

Accident risk missed by a lot, and I tested why rather than guessing. The
information isn't in the data.

**27. ⭐ How do you know the accident data is the problem and not your model?**

I built a deliberately cheating version. It was allowed to see information from
the exact moment being predicted — occupancy, speed and congestion at the target
window — which no real forecast could ever have. If the information existed and
I was just extracting it badly, cheating would have helped a lot.

It gained 0.006. Essentially nothing. So the ceiling on this dataset is about
0.64 and no amount of tuning reaches 0.75.

**28. So the accident model is useless?**

Not quite. The absolute accuracy is poor but the *ranking* is useful. If patrols
cover only the 10% of road-hours the model ranks riskiest, they find 2.6 times
as many incidents as patrolling at random. My recommendation in the report is to
judge the feature on that ranking measure rather than on the original metric.

**29. What data would you need to actually predict accidents?**

Road geometry and lane counts, locations of junctions and merges, historical
collision locations, and sub-30-minute speed variation. A 30-minute average
smooths away exactly the sudden braking that precedes a crash.

**30. Why is macro-F1 the right metric if it made you fail?**

Because the alternative would flatter the result dishonestly. Plain accuracy
would let a model that always guesses "Free-flow" score 60% while being useless,
and Heavy and Severe are precisely the cases the control room cares about.
Choosing the easier metric after seeing the result would be fitting the
measurement to the answer.

---

## F. The advanced parts

**31. ⭐ Did the neural network beat the simpler models?**

Barely, and I don't recommend using it. It scored 55.59 against XGBoost's 55.86
on one measure — under half a percent better — and was slightly worse on
percentage error. It used 64,269 internal parameters and twelve times the
training time. My conclusion in the report is that the engineered history
features already capture the sequential pattern, so the extra complexity doesn't
earn its place.

**32. Then why build it at all?**

The project brief required it, and the benchmark is what justifies the
recommendation. "I built it, tested it fairly, and it wasn't worth deploying" is
a real finding. Without building it, "we don't need a neural network" would just
be an opinion.

**33. How does the confidence range work, and is it real?**

Instead of only "200 vehicles" it gives a range. It's checked: the range promises
to contain the true value 80% of the time, and on the hidden test weeks it
actually did 80.5% of the time. It also widens appropriately — about ±90 vehicles
on a quiet dry night versus ±230 in rush-hour rain.

**34. Why were the accident probabilities showing 44% when accidents are 0.9%?**

Because accidents are so rare, the model has to be told to weight the rare cases
heavily — otherwise it learns "never predict an accident" and is right 99% of the
time while being useless. That weighting distorts the output scale. I added a
calibration step, fitted on data the model hadn't seen, that maps the scores back
onto the real scale. The average prediction went from 48% to 0.80% against a true
rate of 0.96%. Importantly the ordering doesn't change at all — only the scale —
so the ranking still works.

**35. What did you write from scratch rather than using a library?**

Linear regression with gradient descent. The idea is: start with a guess, measure
how wrong it is, adjust in the direction that reduces the error, repeat. I
verified it two ways — it matches the library version to two decimal places
(65.81 versus 65.80), and there's a gradient check comparing the calculus against
a numerical approximation, which agrees to eleven decimal places.

---

## G. Awkward questions

**36. ⭐ Did you write all this code yourself?**

Answer honestly. Something like: *"I used AI assistance substantially for the
implementation. I worked through the reasoning behind the key decisions and I'm
happy to walk through any part of it — particularly the time-based split, the
missing-window problem, the leakage in the congestion column, and the diagnostic
showing the accident target is unreachable."*

Then demonstrate it by explaining one. That's a far better position than being
caught out on a follow-up.

**37. This is a lot for a 3rd-year project. How much do you actually understand?**

Same principle — be specific about what you do and don't understand rather than
claiming all of it. "I understand the data pipeline and the model comparison
well. The LSTM internals I understand at a conceptual level but not the maths of
the gates." That's a credible, respectable answer.

**38. What would you do next?**

Four things. Get a full year of data — five months has no monsoon or summer, so
seasonal robustness is untested. Acquire road-geometry and collision data to make
accident prediction viable. Add drift monitoring, because if a parallel road
closes the corridor's behaviour changes and nothing here would notice. And
resolve the congestion metric question with whoever set the target.

**39. What was the hardest part?**

The missing time windows, because nothing failed — the code ran fine and produced
plausible numbers that were quietly wrong. Finding it required checking that the
row count matched what the timetable should contain, rather than trusting the
file.

**40. If you had to cut something, what would go?**

The neural network. It's the most complex part and delivers the least. The
project would be simpler and the recommendation wouldn't change.

---

## Three things to avoid saying

**"The model is 96% accurate."** That's the R² statistic and it sounds far
stronger than it is. Say "8.9% average error, 45% better than the baseline" —
it's the honest framing and it invites a better conversation.

**"It failed because I ran out of time."** You didn't — you proved the data
doesn't support it. That's a completely different and much stronger statement.

**Anything you can't explain if asked twice.** If you can't survive a follow-up,
don't make the claim. Say what you know and be straightforward about the edges.

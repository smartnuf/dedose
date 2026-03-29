# AGENTS.md — Delta-Sigma Taper Scheduler (`dedose`)

## 1. Purpose of this document

This file provides **context, constraints, and design intent** for AI agents (e.g. Codex CLI) working on this repository.

It explains:

* *what the project is trying to achieve*
* *why the design choices were made*
* *what must NOT be broken during refactoring or extension*

Agents should treat this as a **contract for behaviour and architecture**, not just documentation.

---

## 2. Project overview

This project implements a **tablet dose tapering scheduler** under the constraint:

> Tablets must be taken in whole units (no splitting), but the *target dose trajectory is continuous*.

The solution combines:

* a **continuous-time exponential decay model**
* a **discrete quantisation mechanism (delta-sigma style)**

The result is a sequence of integer daily doses whose **long-term average matches a smooth taper curve**.

---

## 3. Core problem statement

We want a function:

```
ideal_dose(day) ∈ ℝ
actual_dose(day) ∈ ℕ
```

such that:

* `ideal_dose(0) = T`
* `ideal_dose(∞) → F`
* decay occurs over ~`D` days
* tablets are indivisible → must output integers
* long-term average of `actual_dose` ≈ `ideal_dose`

This is fundamentally a **quantisation of a decaying signal**.

---

## 4. Mathematical model

### 4.1 Exponential taper

The ideal (floating-point) dose is:

```
x(n) = F + (T - F) * exp(-k n)
```

where:

```
k = ln((T - F) / ε) / D
```

and:

* `T` = initial dose
* `F` = final asymptotic dose
* `D` = taper horizon (days)
* `ε` = closeness to final value at day D

This ensures:

```
x(D) = F + ε
```

---

### 4.2 Delta-sigma quantisation

We convert `x(n)` to integers using an accumulator:

```
y(n) = floor(r(n-1) + x(n))
r(n) = r(n-1) + x(n) - y(n)
```

Where:

* `y(n)` = tablets taken (integer)
* `r(n)` = residual (floating-point state)

This ensures:

```
Σ y(n) ≈ Σ x(n)
```

i.e. **error is bounded and does not accumulate drift**.

---

## 5. Key design principles

### 5.1 Persistence is essential

The scheduler is **stateful**, not recomputed statelessly.

State includes:

* residual accumulator
* last processed date
* cumulative totals

Rationale:

* correctness depends on historical quantisation error
* recomputation without replay breaks accuracy

---

### 5.2 Time is discrete and sequential

Days must be processed **in order**.

Fast-forwarding is implemented as:

> replay all prior days deterministically

Agents MUST NOT:

* skip state updates
* attempt closed-form shortcuts for quantised output

---

### 5.3 Integer constraint is non-negotiable

All outputs:

```
actual_dose ∈ ℕ
```

Agents MUST NOT:

* introduce fractional tablets
* “round differently” without preserving accumulator logic

---

### 5.4 The floating-point state is the ground truth

The variable:

```
current_ideal
```

represents the **true underlying taper**

The integer output is only a **projection** of this.

---

### 5.5 Safety constraints

Optional but important:

* `max_daily_tablets` may cap output
* doses must never be negative

Agents should preserve or strengthen safety guards.

---

## 6. File structure

Initial structure:

```
/src/dedose.py      # core script (single-file implementation)
/AGENTS.md          # this file
```

Future expected structure:

```
/src/dedose/
    __init__.py
    model.py        # exponential model
    quantizer.py    # delta-sigma logic
    plan.py         # persistent state
    cli.py          # command interface

/tests/
README.md
```

Agents may refactor toward this modular form.

---

## 7. CLI behaviour contract

The tool must support:

* create named plan
* query today's dose
* query arbitrary date (with replay)
* persistent storage

Persistence location (current):

```
~/.delta_sigma_taper_state.json
```

Agents may improve this (e.g. SQLite), but must preserve:

* determinism
* backward compatibility (or provide migration)

---

## 8. What MUST be preserved

Agents modifying code must preserve:

1. **Dose integral consistency**

   ```
   sum(actual) ≈ sum(ideal)
   ```

2. **Deterministic replay**
   Same inputs → same outputs

3. **State continuity**
   No loss of residual or history

4. **Monotonic decay of ideal signal**

---

## 9. Acceptable extensions

Agents are encouraged to add:

* alternative taper laws:

  * linear
  * hyperbolic
  * piecewise
* visualisation tools
* export (CSV/calendar)
* simulation mode (no persistence)
* unit tests (important)
* validation checks

---

## 10. Non-goals

This project does NOT:

* provide medical advice
* optimise pharmacokinetics
* replace clinician guidance

It is purely:

> a **signal-processing-based scheduling tool**

---

## 11. Mental model (important)

This system is equivalent to:

> A **1-bit DAC / delta-sigma modulator**
> tracking a **slowly varying reference signal**

Where:

* reference = exponential taper
* output = integer tablets
* accumulator = error feedback

Agents familiar with DSP should use this analogy.

---

## 12. Guidance for AI agents

When modifying this project:

* Think in terms of **signals and invariants**, not just code
* Do not “simplify away” the accumulator
* Avoid hidden state
* Prefer explicit, inspectable variables
* Maintain numerical stability

If unsure:

> preserve behaviour first, improve structure second

---

## 13. Future directions

Planned or desirable:

* multi-dose-per-day schedules
* stochastic vs deterministic quantisation comparison
* UI / calendar integration
* integration with wearable / symptom tracking

---

End of AGENTS.md

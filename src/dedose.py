#!/usr/bin/env python3
"""
dedose.py

A persistent named taper scheduler using:
  - an exponential floating-point target dose
  - delta-sigma style quantization to whole tablets

It can:
  - create named plans
  - tell you how many tablets to take today
  - fast-forward automatically from the plan start date to any date,
    assuming you took the advised dose on each previous day
  - show status

Examples
--------
Create a plan:
    python dedose.py init myplan --start 2026-03-29 --T 1.0 --D 56 --F 0.25

Ask what to take today:
    python dedose.py dose myplan

Ask what to take on a specific date (and fast-forward to it):
    python dedose.py dose myplan --date 2026-04-10

Show status:
    python dedose.py status myplan

List plans:
    python dedose.py list

Delete a plan:
    python dedose.py delete myplan
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List


DEFAULT_STATE_FILE = Path.home() / ".dedose_state.json"
STATE_FILE = DEFAULT_STATE_FILE


def set_state_file(path_str: str | None) -> None:
    """Allow overriding the persistence path (useful for tests)."""
    global STATE_FILE
    if path_str:
        STATE_FILE = Path(path_str).expanduser()
    else:
        STATE_FILE = DEFAULT_STATE_FILE
DATE_FMT = "%Y-%m-%d"


def parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, DATE_FMT).date()
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{s}'. Use YYYY-MM-DD.") from exc


def today_local() -> date:
    return date.today()


def load_store() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {"plans": {}}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise SystemExit(f"Failed to read state file {STATE_FILE}: {exc}") from exc
    if not isinstance(data, dict) or "plans" not in data or not isinstance(data["plans"], dict):
        raise SystemExit(f"State file {STATE_FILE} is malformed.")
    return data


def save_store(store: Dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, STATE_FILE)
    except Exception as exc:
        raise SystemExit(f"Failed to write state file {STATE_FILE}: {exc}") from exc


@dataclass
class Plan:
    name: str
    start_date: str          # YYYY-MM-DD
    T: float                 # initial ideal tablets/day
    D: float                 # decay horizon in days
    F: float                 # final asymptotic tablets/day
    epsilon: float           # closeness parameter for day D
    max_daily_tablets: int | None

    # persistent evolving state
    last_processed_date: str | None
    residual: float          # delta-sigma residual accumulator
    current_ideal: float     # floating-point ideal state at last_processed_date
    total_tablets_given: int
    total_ideal_sum: float

    def start(self) -> date:
        return parse_date(self.start_date)

    def last_processed(self) -> date | None:
        return parse_date(self.last_processed_date) if self.last_processed_date else None

    def fresh_state(self) -> "Plan":
        """
        Return a copy reset to its initial state.
        Useful for simulations/visualisations that should not mutate persistence.
        """
        return Plan(
            name=self.name,
            start_date=self.start_date,
            T=self.T,
            D=self.D,
            F=self.F,
            epsilon=self.epsilon,
            max_daily_tablets=self.max_daily_tablets,
            last_processed_date=None,
            residual=0.0,
            current_ideal=self.T,
            total_tablets_given=0,
            total_ideal_sum=0.0,
        )

    def horizon_rate(self) -> float:
        """
        Exponential target:
            x(n) = F + (T - F) * exp(-k n)

        k is chosen so that at n = D:
            x(D) = F + epsilon

        Thus:
            exp(-kD) = epsilon / (T - F)
            k = ln((T - F)/epsilon) / D

        Requires T > F and epsilon > 0.
        """
        if self.T < self.F:
            raise ValueError("This script expects T >= F for a taper.")
        if self.T == self.F:
            return 0.0
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0.")
        if self.D <= 0:
            raise ValueError("D must be > 0.")
        span = self.T - self.F
        if self.epsilon >= span:
            raise ValueError("epsilon must be smaller than T - F.")
        return math.log(span / self.epsilon) / self.D

    def ideal_for_date(self, target_date: date) -> float:
        n = (target_date - self.start()).days
        if n < 0:
            raise ValueError("target_date is before the plan start date.")
        if self.T == self.F:
            return self.F
        k = self.horizon_rate()
        return self.F + (self.T - self.F) * math.exp(-k * n)

    def quantize(self, ideal: float) -> int:
        """
        Delta-sigma style whole-tablet quantization.
        residual carries the fractional debt/credit forward.

        A simple first-order form:
            y = floor(residual + ideal)
            residual <- residual + ideal - y
        """
        x = self.residual + ideal
        tablets = math.floor(x)

        if tablets < 0:
            tablets = 0

        if self.max_daily_tablets is not None:
            tablets = min(tablets, self.max_daily_tablets)

        self.residual = self.residual + ideal - tablets

        # Keep residual in a sane range against roundoff drift.
        if abs(self.residual) < 1e-15:
            self.residual = 0.0

        return tablets

    def advance_through(self, target_date: date) -> int:
        """
        Advance state day-by-day from the next unprocessed day through target_date,
        assuming all advised historic doses were taken.

        Returns the integer tablets for target_date.
        """
        if target_date < self.start():
            raise SystemExit("Requested date is before the plan start date.")

        if self.last_processed() is None:
            current = self.start()
        else:
            current = self.last_processed() + timedelta(days=1)

        last_dose = None
        while current <= target_date:
            ideal = self.ideal_for_date(current)
            dose = self.quantize(ideal)

            self.last_processed_date = current.strftime(DATE_FMT)
            self.current_ideal = ideal
            self.total_tablets_given += dose
            self.total_ideal_sum += ideal

            last_dose = dose
            current += timedelta(days=1)

        if last_dose is None:
            # Already processed this date in the past; just recompute dose is not possible
            # without replay, so we report the already-processed day's ideal state only.
            # In ordinary use, dose is asked once per new day, so this path is uncommon.
            raise SystemExit(
                "That date has already been processed. "
                "Use 'status' to inspect current state, or create a fresh plan if needed."
            )

        return last_dose


def simulate_schedule(plan: Plan, days: int) -> List[Dict[str, Any]]:
    """
    Simulate a plan for a given number of days from the start date, returning history
    entries containing daily and cumulative ideal vs quantized doses.
    """
    if days <= 0:
        raise SystemExit("Days must be a positive integer.")

    sim = plan.fresh_state()
    history: List[Dict[str, Any]] = []
    current = sim.start()
    cumulative_ideal = 0.0
    cumulative_actual = 0

    for day_index in range(days):
        ideal = sim.ideal_for_date(current)
        tablets = sim.quantize(ideal)
        cumulative_ideal += ideal
        cumulative_actual += tablets
        history.append(
            {
                "day_index": day_index,
                "date": current,
                "ideal": ideal,
                "tablets": tablets,
                "cum_ideal": cumulative_ideal,
                "cum_actual": cumulative_actual,
                "residual": sim.residual,
            }
        )
        current += timedelta(days=1)

    return history


def streak_transition_points(history: List[Dict[str, Any]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Derive streak lengths for taking vs resting segments:
      - taking_stops: (day_index, run_length) recorded when a run of taking tablets ends
        because the current day has zero tablets.
      - rest_stops: (day_index, run_length) recorded when a rest period ends and dosing resumes.
    """
    taking_run = 0
    rest_run = 0
    taking_stops: list[tuple[int, int]] = []
    rest_stops: list[tuple[int, int]] = []

    for entry in history:
        idx = entry["day_index"]
        taking_today = entry["tablets"] > 0

        if taking_today:
            taking_run += 1
            if rest_run > 0:
                rest_stops.append((idx, rest_run))
                rest_run = 0
        else:
            rest_run += 1
            if taking_run > 0:
                taking_stops.append((idx, taking_run))
                taking_run = 0

    return taking_stops, rest_stops


def plan_from_store(store: Dict[str, Any], name: str) -> Plan:
    raw = store["plans"].get(name)
    if raw is None:
        raise SystemExit(f"No such plan: {name}")
    try:
        return Plan(**raw)
    except TypeError as exc:
        raise SystemExit(f"Stored plan '{name}' is malformed: {exc}") from exc


def save_plan(store: Dict[str, Any], plan: Plan) -> None:
    store["plans"][plan.name] = asdict(plan)
    save_store(store)


def cmd_init(args: argparse.Namespace) -> None:
    store = load_store()
    if args.name in store["plans"] and not args.force:
        raise SystemExit(
            f"Plan '{args.name}' already exists. Use a different name or --force to overwrite."
        )

    if args.T < 0 or args.F < 0:
        raise SystemExit("T and F must be non-negative.")
    if args.T < args.F:
        raise SystemExit("This taper script expects T >= F.")
    if args.D <= 0:
        raise SystemExit("D must be > 0.")
    if args.T > args.F:
        span = args.T - args.F
        if args.epsilon <= 0 or args.epsilon >= span:
            raise SystemExit("For T > F, epsilon must satisfy 0 < epsilon < (T - F).")

    plan = Plan(
        name=args.name,
        start_date=parse_date(args.start).strftime(DATE_FMT),
        T=float(args.T),
        D=float(args.D),
        F=float(args.F),
        epsilon=float(args.epsilon),
        max_daily_tablets=args.max_daily_tablets,
        last_processed_date=None,
        residual=0.0,
        current_ideal=float(args.T),
        total_tablets_given=0,
        total_ideal_sum=0.0,
    )
    store["plans"][args.name] = asdict(plan)
    save_store(store)

    print(f"Created plan '{args.name}'.")
    print(f"  Start date          : {plan.start_date}")
    print(f"  Initial ideal T     : {plan.T}")
    print(f"  Horizon D (days)    : {plan.D}")
    print(f"  Final ideal F       : {plan.F}")
    print(f"  Epsilon             : {plan.epsilon}")
    print(f"  Max daily tablets   : {plan.max_daily_tablets}")


def cmd_dose(args: argparse.Namespace) -> None:
    store = load_store()
    plan = plan_from_store(store, args.name)

    target_date = parse_date(args.date) if args.date else today_local()
    dose = plan.advance_through(target_date)
    save_plan(store, plan)

    avg_given = (
        plan.total_tablets_given / ((target_date - plan.start()).days + 1)
        if target_date >= plan.start()
        else 0.0
    )

    print(f"Plan                 : {plan.name}")
    print(f"Date                 : {target_date.strftime(DATE_FMT)}")
    print(f"Take today           : {dose} tablet(s)")
    print(f"Ideal floating state : {plan.current_ideal:.6f} tablet/day")
    print(f"Residual accumulator : {plan.residual:.6f}")
    print(f"Total given so far   : {plan.total_tablets_given} tablet(s)")
    print(f"Avg given/day so far : {avg_given:.6f}")


def cmd_status(args: argparse.Namespace) -> None:
    store = load_store()
    plan = plan_from_store(store, args.name)

    today = today_local()
    days_since_start = (today - plan.start()).days
    future_ideal = None
    if days_since_start >= 0:
        future_ideal = plan.ideal_for_date(today)

    print(f"Plan                 : {plan.name}")
    print(f"Start date           : {plan.start_date}")
    print(f"T                    : {plan.T}")
    print(f"D                    : {plan.D}")
    print(f"F                    : {plan.F}")
    print(f"Epsilon              : {plan.epsilon}")
    print(f"Max daily tablets    : {plan.max_daily_tablets}")
    print(f"Last processed date  : {plan.last_processed_date}")
    print(f"Current ideal state  : {plan.current_ideal:.6f}")
    print(f"Residual accumulator : {plan.residual:.6f}")
    print(f"Total ideal sum      : {plan.total_ideal_sum:.6f}")
    print(f"Total tablets given  : {plan.total_tablets_given}")

    if future_ideal is not None:
        print(f"Ideal for today      : {future_ideal:.6f} tablet/day")


def cmd_list(args: argparse.Namespace) -> None:
    store = load_store()
    plans = store["plans"]
    if not plans:
        print("No plans found.")
        return
    for name, raw in sorted(plans.items()):
        print(
            f"{name}: start={raw['start_date']}, "
            f"T={raw['T']}, D={raw['D']}, F={raw['F']}, "
            f"last_processed={raw['last_processed_date']}"
        )


def cmd_delete(args: argparse.Namespace) -> None:
    store = load_store()
    if args.name not in store["plans"]:
        raise SystemExit(f"No such plan: {args.name}")
    del store["plans"][args.name]
    save_store(store)
    print(f"Deleted plan '{args.name}'.")


def cmd_plot(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires matplotlib. Install it with 'pip install matplotlib'."
        ) from exc

    store = load_store()
    plan = plan_from_store(store, args.name)
    history = simulate_schedule(plan, args.days)
    x = [entry["day_index"] for entry in history]

    plt.figure(figsize=(8, 4.5))
    if args.mode == "cumulative":
        cum_ideal = [entry["cum_ideal"] for entry in history]
        cum_actual = [entry["cum_actual"] for entry in history]
        plt.plot(x, cum_ideal, label="Cumulative ideal dose", linewidth=2)
        plt.plot(x, cum_actual, label="Cumulative quantized tablets", linewidth=2)
        plt.ylabel("Cumulative tablets")
        plt.title(f"Cumulative tracking for plan '{plan.name}'")
    elif args.mode == "daily":
        daily_ideal = [entry["ideal"] for entry in history]
        daily_actual = [entry["tablets"] for entry in history]
        plt.plot(x, daily_ideal, label="Ideal daily dose", linewidth=2)
        plt.step(x, daily_actual, label="Quantized tablets (step)", where="post", linewidth=2)
        plt.ylabel("Tablets/day")
        plt.title(f"Daily ideal vs quantized tablets for plan '{plan.name}'")
    else:
        taking_stops, rest_stops = streak_transition_points(history)
        if taking_stops:
            x_take = [idx for idx, _ in taking_stops]
            y_take = [length for _, length in taking_stops]
            plt.stem(
                x_take,
                y_take,
                linefmt="C0-",
                markerfmt="C0o",
                basefmt="k-",
                label="Taking streak ending",
            )
        if rest_stops:
            x_rest = [idx for idx, _ in rest_stops]
            y_rest = [length for _, length in rest_stops]
            plt.stem(
                x_rest,
                y_rest,
                linefmt="C1-",
                markerfmt="C1s",
                basefmt="k-",
                label="Rest streak ending",
            )
        plt.ylabel("Consecutive days")
        plt.title(f"Streak transitions for plan '{plan.name}'")

    plt.xlabel("Days since plan start")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Persistent named exponential taper scheduler using delta-sigma whole-tablet quantization."
    )
    p.add_argument(
        "--state-file",
        help="Optional override for the persistent state file path (default ~/.dedose_state.json).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create or overwrite a named plan.")
    p_init.add_argument("name", help="Plan name.")
    p_init.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    p_init.add_argument("--T", type=float, required=True, help="Initial ideal tablets/day.")
    p_init.add_argument("--D", type=float, required=True, help="Decay horizon in days.")
    p_init.add_argument("--F", type=float, required=True, help="Final ideal tablets/day.")
    p_init.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="At day D, ideal dose will be F + epsilon. Default: 0.01",
    )
    p_init.add_argument(
        "--max-daily-tablets",
        type=int,
        default=None,
        help="Optional safety cap for daily tablet count.",
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing plan of the same name.",
    )
    p_init.set_defaults(func=cmd_init)

    p_dose = sub.add_parser("dose", help="Fast-forward through history and tell you the dose for a date.")
    p_dose.add_argument("name", help="Plan name.")
    p_dose.add_argument("--date", help="Date YYYY-MM-DD. Defaults to today.")
    p_dose.set_defaults(func=cmd_dose)

    p_status = sub.add_parser("status", help="Show stored state for a plan.")
    p_status.add_argument("name", help="Plan name.")
    p_status.set_defaults(func=cmd_status)

    p_list = sub.add_parser("list", help="List plans.")
    p_list.set_defaults(func=cmd_list)

    p_delete = sub.add_parser("delete", help="Delete a plan.")
    p_delete.add_argument("name", help="Plan name.")
    p_delete.set_defaults(func=cmd_delete)

    p_plot = sub.add_parser("plot", help="Visualize cumulative ideal vs quantized tablets.")
    p_plot.add_argument("name", help="Plan name.")
    p_plot.add_argument(
        "--days",
        type=int,
        default=120,
        help="Number of days from the start date to include in the visualization (default 120).",
    )
    p_plot.add_argument(
        "--mode",
        choices=["cumulative", "daily", "streaks"],
        default="cumulative",
        help=(
            "Plot cumulative sums (default), a daily step trace, or streak transitions "
            "between taking and rest days."
        ),
    )
    p_plot.add_argument(
        "--output",
        help="Optional file path to save the plot instead of showing it interactively.",
    )
    p_plot.set_defaults(func=cmd_plot)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_state_file(args.state_file)
    args.func(args)


if __name__ == "__main__":
    main()

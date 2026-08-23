def _cycle_int(value, label: str) -> int:
    try:
        num = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label} is not a number: {value!r}") from e
    if num != int(num):
        raise ValueError(f"{label} is not an integer cycle count: {value!r}")
    return int(num)


def e2e_cycles_from_perfetto(data: dict) -> int:
    """End-to-end cycle count: min(ts) .. max(ts+dur) over complete (ph=X) events.

    Perfetto ts/dur in this stack are cycle offsets (see elapsed_cycle), not wall ns.
    """
    events = data.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("perfetto missing traceEvents list")
    xs = [e for e in events if isinstance(e, dict) and e.get("ph") == "X"]
    if not xs:
        raise ValueError("perfetto has no complete (ph=X) events")
    t0 = min(_cycle_int(e["ts"], "ts") for e in xs)
    t1 = max(_cycle_int(e["ts"], "ts") + _cycle_int(e["dur"], "dur") for e in xs)
    return t1 - t0


def mean_cycles(values: list[int]) -> int:
    if not values:
        raise ValueError("mean_cycles requires a non-empty list")
    total = sum(values)
    n = len(values)
    if total % n != 0:
        raise ValueError(f"mean cycles not integral: sum={total} n={n} values={values}")
    return total // n

"""'No tool fits -> []' (abstention) examples.

xLAM has zero examples where the right answer is an empty list, so a model trained only on xLAM
learns "always call something" (the first SFT run scored 0/10 on irrelevant prompts).
We make abstention examples from real records, two ways:

- removed: drop every tool the gold answer uses and keep the distractor tools.
           Hard negatives: the remaining tools are from the same API family/style.
- swapped: keep the query, give it the tool list of an unrelated record.
           Easy negatives: clearly nothing fits.

Guard against making WRONG negatives: if a remaining tool looks like a removed one, by name
(auto_complete vs auto_complete_deprecating) or by topic ("current weather" vs "hourly weather
forecast"), the record is skipped, because the remaining tool might still answer the query.
"""
import random
import re

ID_OFFSET = 10_000_000          # negative ids = ID_OFFSET + source id (ints, so the id column keeps one type)


def _name_tokens(name):
    return {t for t in re.split(r"[_\W]+", str(name).lower()) if len(t) > 1}


def similar_names(a, b, threshold=0.5):
    ta, tb = _name_tokens(a), _name_tokens(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= threshold


# generic API words that say nothing about WHAT a tool does
_STOP = set("""fetch fetches fetching retrieve retrieves get gets given specified specific using based data list
information info details detail returns return from with that this the for and api calculates calculate
compute computes check checks number value values provided into about which their whether optional default
string type types request requests response result results find finds search searches""".split())


def _topic_words(tool):
    text = f"{tool.get('name', '')} {tool.get('description', '')}".lower()
    return {w for w in re.split(r"[^a-z]+", text) if len(w) > 3 and w not in _STOP}


def same_topic(tool_a, tool_b, min_shared=2):
    """True if two tools share >= 2 topic words, e.g. 'current weather for a location' vs 'hourly weather forecast for a location'."""
    return len(_topic_words(tool_a) & _topic_words(tool_b)) >= min_shared


def could_still_answer(candidate, removed_tools):
    return any(similar_names(candidate.get("name"), r.get("name")) or same_topic(candidate, r) for r in removed_tools)


def make_removed(record):
    used = {c["name"] for c in record["answers"]}
    removed = [t for t in record["tools"] if t.get("name") in used]
    left = [t for t in record["tools"] if t.get("name") not in used]
    if not left or any(could_still_answer(t, removed) for t in left):
        return None
    return {**record, "tools": left, "answers": []}


def make_swapped(record, donor):
    used = {c["name"] for c in record["answers"]}
    gold_tools = [t for t in record["tools"] if t.get("name") in used]
    own = {t.get("name") for t in record["tools"]}
    for t in donor["tools"]:
        if t.get("name") in own or could_still_answer(t, gold_tools):
            return None
    return {**record, "tools": donor["tools"], "answers": []}


def build_abstention_records(records, n, seed=3407, id_prefix=None):
    """n negatives (about half 'removed', half 'swapped') from `records`. Deterministic for a given seed.

    id_prefix=None -> int ids (ID_OFFSET + source id); a string -> f"{id_prefix}{source id}".
    """
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    out = []
    for idx in order:
        if len(out) >= n:
            break
        rec = records[idx]
        want_removed = sum(r["neg_kind"] == "removed" for r in out) <= sum(r["neg_kind"] == "swapped" for r in out)
        if want_removed:
            neg, kind = make_removed(rec), "removed"
        else:
            neg, kind = make_swapped(rec, records[rng.randrange(len(records))]), "swapped"
        if neg is None:
            continue
        new_id = ID_OFFSET + rec["id"] if id_prefix is None else f"{id_prefix}{rec['id']}"
        out.append({**neg, "id": new_id, "source_id": rec["id"], "neg_kind": kind})
    return out

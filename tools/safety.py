"""Safety guardrails for the Smart-Home Forecaster.

This module is the project's answer to the capstone's Safety concept: a statement
of the unintended actions the system could take, and the concrete steps that
prevent them. See docs/safety.md for the written analysis.

Design principle (the same one used for freeze/heat risk and savings math):
**safety decisions are made in deterministic code, not by the language model.**
A free-tier LLM can be talked out of a soft instruction in a prompt; a regex
screen that runs before and after the model cannot. The guardrails here are:

  1. EMERGENCY ESCALATION  - life-safety situations (gas, CO, fire, flooding,
     electrical, heat illness) short-circuit normal agent processing and return
     emergency instructions immediately.
  2. HIGH-RISK WORK REFUSAL - refuse step-by-step instructions for work that can
     kill or cause serious damage (service panels, gas lines, structural, roof,
     asbestos) and route the user to a licensed professional.
  3. HUMAN-IN-THE-LOOP - any outward, side-effecting action (sending a message,
     booking, purchasing) requires explicit user confirmation first.
  4. PII GUARD - detect sensitive personal identifiers so they are never echoed
     back or written to logs//memory.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# 1. Emergencies — immediate, life-safety situations.
# --------------------------------------------------------------------------
# Each entry: (regex, emergency type, what the user must do RIGHT NOW).
_EMERGENCIES: list[tuple[str, str, str]] = [
    (
        r"\b(smell(ing)?\s+(of\s+)?gas|gas\s+leak|rotten\s+egg\s+smell|hissing\s+(from\s+)?(the\s+)?gas)\b",
        "natural gas leak",
        "Leave the building NOW. Do not flip switches, unplug anything, or use a phone "
        "indoors — a spark can ignite gas. Once outside and away, call 911 and your gas "
        "utility's emergency line. Do not go back in until they say it is safe.",
    ),
    (
        r"\b(carbon\s*monoxide|\bco\s+alarm|co\s+detector\s+(is\s+)?(going\s+off|beeping|alarm))\b",
        "carbon monoxide",
        "Get everyone and any pets outside into fresh air IMMEDIATELY, then call 911. "
        "Carbon monoxide is invisible and can be fatal. Do not re-enter until emergency "
        "responders clear the home.",
    ),
    (
        r"\b(house\s+is\s+on\s+fire|there'?s\s+a\s+fire|smoke\s+(is\s+)?(filling|pouring)|flames)\b",
        "fire",
        "Get everyone out of the building immediately and call 911 from outside. "
        "Do not stop to collect belongings.",
    ),
    (
        r"\b(pipe\s+(just\s+)?burst|burst\s+pipe|flooding|water\s+is\s+pouring|water\s+everywhere|"
        r"gushing\s+water)\b",
        "burst pipe / flooding",
        "Shut off your main water valve now to stop the flow. If water is near outlets, "
        "wiring, or your electrical panel, do NOT wade in — cut power at the breaker only "
        "if you can reach it safely and dryly, otherwise call 911. Then call a licensed "
        "plumber.",
    ),
    (
        r"\b(sparks?\s+(from|coming)|burning\s+smell\s+from\s+(the\s+)?(outlet|panel|breaker|wiring)|"
        r"outlet\s+is\s+(smoking|sparking)|electrical\s+fire)\b",
        "electrical hazard",
        "Do not touch the outlet or panel. If it is safe and dry to do so, cut power at the "
        "breaker; if there is smoke or flame, get out and call 911. Then call a licensed "
        "electrician — do not use that circuit until it is inspected.",
    ),
    (
        r"\b(heat\s*stroke|heatstroke|passed\s+out\s+from\s+(the\s+)?heat|"
        r"(confused|unconscious|not\s+sweating)\s+.*\bheat\b)\b",
        "heat illness",
        "Call 911 now. Move the person somewhere cool, cool them with wet cloths or a "
        "cool bath, and do not give fluids if they are confused or unconscious. "
        "Heatstroke is a medical emergency.",
    ),
]

# Preventive / hypothetical framing. This matters a lot here: the product's
# flagship feature is *preventing* burst pipes, so "how do I prevent a burst pipe"
# must NOT be treated as an active emergency. When this matches, emergency
# detection is suppressed and the question is answered normally.
_PREVENTIVE_CONTEXT = re.compile(
    r"\b(prevent|prevention|preventing|avoid|avoiding|protect|protecting|prepare|preparing|"
    r"winteriz\w*|in\s+case|what\s+if|risk\s+of|reduce\s+the\s+(chance|risk)|"
    r"keep\s+.*\bfrom\b|stop\s+.*\bfrom\b|before\s+(a|the|it)\b|tips|advice\s+on|"
    r"should\s+i\s+worry|how\s+likely|checklist)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# 2. High-risk work — never give step-by-step DIY instructions for these.
# --------------------------------------------------------------------------
_HIGH_RISK: list[tuple[str, str, str]] = [
    (
        r"\b(breaker\s+box|service\s+panel|electrical\s+panel|main\s+panel|"
        r"rewire|replace\s+.*\b(outlet|switch|wiring)\b|240\s*v|electrical\s+work)\b",
        "electrical service work",
        "a licensed electrician (permit and inspection are typically required)",
    ),
    (
        r"\b(gas\s+line|gas\s+pipe|gas\s+valve|move\s+.*\bgas\b|connect\s+.*\bgas\b|"
        r"gas\s+(furnace|water\s+heater)\s+(install|replace|hook)|flue|gas\s+venting)\b",
        "natural gas work",
        "a licensed plumber/HVAC professional and your gas utility",
    ),
    (
        r"\b(load[-\s]?bearing|remove\s+a?\s*wall|structural|foundation\s+(crack|repair)|"
        r"support\s+beam|joist)\b",
        "structural work",
        "a structural engineer and a licensed general contractor (permit required)",
    ),
    (
        r"\b(asbestos|lead\s+paint|mold\s+remediation)\b",
        "hazardous material",
        "a certified abatement professional — improper removal releases hazardous particles",
    ),
    (
        r"\b(on\s+the\s+roof|roof\s+repair|climb\s+.*\broof\b|re[-\s]?roof|second\s+story\s+ladder)\b",
        "work at height / roof work",
        "a licensed roofing contractor — falls are a leading cause of home-repair injury",
    ),
    (
        r"\b(pool\s+(wiring|bonding|electrical)|tree\s+.*\bpower\s+line|near\s+.*\bpower\s+line)\b",
        "electrical proximity hazard",
        "a licensed professional — contact with energized lines is frequently fatal",
    ),
]

# --------------------------------------------------------------------------
# 3. Outward actions — require explicit human confirmation before executing.
# --------------------------------------------------------------------------
_OUTWARD_ACTION = re.compile(
    r"\b(send|email|e-mail|text|message|contact|call)\s+(my\s+|the\s+)?"
    r"(landlord|hoa|contractor|plumber|electrician|neighbor|property\s+manager)\b"
    r"|\b(book|schedule|make)\s+(an?\s+)?(appointment|service\s+call|visit)\b"
    r"|\b(buy|purchase|order|pay)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# 4. PII — never echo back or persist these.
# --------------------------------------------------------------------------
_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    # Separators only BETWEEN digits — an earlier form was `(?:\d[ -]?){13,16}`,
    # which consumed the space after the final digit and ate the following word.
    (r"\b\d(?:[ -]?\d){12,15}\b", "payment card number"),
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "email address"),
    (r"\b(?:account|acct)\s*#?\s*\d{6,}\b", "account number"),
]

# Patterns that need a second check before they count. The card pattern is a bare
# run of 13-16 digits, which also matches a parcel number, an APN, a long order
# reference and a meter serial — all of which this product legitimately handles.
# Redaction now happens on the live input path, so a false positive corrupts a
# real question rather than merely over-reporting. Luhn costs nothing and removes
# almost all of them: a real card passes it, an arbitrary digit run has about a
# 1-in-10 chance.
_VALIDATORS = {"payment card number": lambda s: _luhn_ok(s)}


def _luhn_ok(candidate: str) -> bool:
    """True when a digit run satisfies the Luhn checksum used by payment cards."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

DISCLAIMER = (
    "This is general guidance, not professional advice. For gas, electrical, structural, "
    "flooding, or medical emergencies, contact 911, your utility, or a licensed professional."
)


def check_emergency(text: str) -> dict:
    """Detect a life-safety emergency in the user's message.

    Returns {"emergency": True, type, instruction} when matched, else
    {"emergency": False}. When True the caller should return the instruction
    IMMEDIATELY rather than running the normal agent loop.

    Preventive/hypothetical phrasing ("how do I prevent a burst pipe") is
    deliberately NOT an emergency — that is this product's core use case.
    """
    lowered = text.lower()
    if _PREVENTIVE_CONTEXT.search(lowered):
        return {"emergency": False, "suppressed_by": "preventive phrasing"}
    for pattern, kind, instruction in _EMERGENCIES:
        if re.search(pattern, lowered):
            return {"emergency": True, "type": kind, "instruction": instruction}
    return {"emergency": False}


def check_high_risk(text: str) -> dict:
    """Detect a request for step-by-step instructions on dangerous work.

    Returns {"high_risk": True, category, refer_to} when matched, else
    {"high_risk": False}. The agent may still explain scope, cost, permits, and
    how to choose a pro — it must not give a DIY procedure.
    """
    lowered = text.lower()
    for pattern, category, refer_to in _HIGH_RISK:
        if re.search(pattern, lowered):
            return {"high_risk": True, "category": category, "refer_to": refer_to}
    return {"high_risk": False}


def needs_confirmation(text: str) -> bool:
    """True if the request implies an outward, side-effecting action that must be
    confirmed by the user before the system performs it."""
    return bool(_OUTWARD_ACTION.search(text))


def _matches(text: str, pattern: str, label: str) -> list[re.Match]:
    """Matches that survive the label's validator, if it has one."""
    validator = _VALIDATORS.get(label)
    return [m for m in re.finditer(pattern, text)
            if validator is None or validator(m.group(0))]


def find_pii(text: str) -> list[str]:
    """Return the kinds of sensitive identifiers present (for redaction/logging)."""
    return [label for pattern, label in _PII_PATTERNS
            if _matches(text, pattern, label)]


def redact_pii(text: str) -> str:
    """Mask sensitive identifiers so they are never echoed back, stored or logged.

    Applied to the user's message at the start of every turn, so the redaction is
    upstream of the model, the conversation checkpointer, episodic memory and the
    telemetry log. Redacting at each sink instead would mean four places to keep
    in step, and the checkpointer — which holds the full message history — was the
    one previously missed.
    """
    out = text
    for pattern, label in _PII_PATTERNS:
        replacement = f"[redacted {label}]"
        # Right-to-left so earlier spans keep their offsets as we substitute.
        for m in reversed(_matches(out, pattern, label)):
            out = out[:m.start()] + replacement + out[m.end():]
    return out


def screen_input(text: str) -> dict:
    """Run every input guardrail and return one consolidated verdict.

    Returns:
        {
          "block": bool,          # True => return `response` immediately, skip the agent
          "response": str | None, # the text to return when blocked
          "emergency": {...},
          "high_risk": {...},
          "needs_confirmation": bool,
          "pii_found": [...],
        }
    """
    emergency = check_emergency(text)
    high_risk = check_high_risk(text)
    pii = find_pii(text)

    response = None
    if emergency["emergency"]:
        response = (
            f"**⚠️ This sounds like an emergency ({emergency['type']}).**\n\n"
            f"{emergency['instruction']}\n\n"
            "I'm an informational assistant — please get emergency help first. "
            "I can help with prevention and next steps once everyone is safe."
        )

    return {
        "block": bool(emergency["emergency"]),
        "response": response,
        "emergency": emergency,
        "high_risk": high_risk,
        "needs_confirmation": needs_confirmation(text),
        "pii_found": pii,
    }


def guidance_for_prompt(screen: dict) -> str:
    """Turn a screen_input() verdict into instructions appended to the agent prompt.

    This is how a non-blocking guardrail (high-risk work, outward actions, PII)
    steers the model without needing it to rediscover the rule on its own.
    """
    notes: list[str] = []
    if screen["high_risk"]["high_risk"]:
        hr = screen["high_risk"]
        notes.append(
            f"SAFETY OVERRIDE: this involves {hr['category']}. Do NOT provide step-by-step "
            f"DIY instructions. Explain the risk plainly, state that it requires "
            f"{hr['refer_to']}, and cover scope/permits/how to choose a pro instead."
        )
    if screen["needs_confirmation"]:
        notes.append(
            "SAFETY OVERRIDE: this implies an outward action (messaging, booking, or "
            "purchasing). You may DRAFT or propose it, but state clearly that you will not "
            "send, book, or buy anything, and ask the user to confirm and do it themselves."
        )
    if screen["pii_found"]:
        notes.append(
            f"PRIVACY: the message contains {', '.join(screen['pii_found'])}. Do not repeat "
            "these values back or store them; refer to them only in general terms."
        )
    return "\n".join(notes)


if __name__ == "__main__":
    samples = [
        "I smell gas in my kitchen, what do I do?",
        "My CO detector is going off",
        "A pipe just burst and water is everywhere!",
        "How do I replace the breaker box myself?",
        "Can I remove a load-bearing wall to open up the kitchen?",
        "Email my landlord about the broken AC",
        "My SSN is 123-45-6789, can you save it?",
        "How do I hang a 20 lb mirror on drywall?",  # benign control
    ]
    for s in samples:
        v = screen_input(s)
        flags = []
        if v["emergency"]["emergency"]:
            flags.append(f"EMERGENCY({v['emergency']['type']})")
        if v["high_risk"]["high_risk"]:
            flags.append(f"HIGH_RISK({v['high_risk']['category']})")
        if v["needs_confirmation"]:
            flags.append("NEEDS_CONFIRM")
        if v["pii_found"]:
            flags.append(f"PII({','.join(v['pii_found'])})")
        print(f"{'BLOCK' if v['block'] else 'pass ':6} {flags or ['clean']}  <- {s}")

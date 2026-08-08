"""
prompts.py

Loads the ORIGINAL prompt templates straight from the repo
(dt_code/prompts_i+2/student_prompts.yaml) and reuses them verbatim -- they're
solid, tested prompt engineering, no reason to rewrite them.

Also defines BLIND_TUTOR_PROMPT: a new prompt for the project's Stage-1 Tutor.

Why a new prompt instead of reusing teacher_prompt.yaml's "teacher_only_prompt"?
Because that prompt deliberately hands the grader CORRECT_STEP ("You are a
professor with an access to CORRECT_STEP..."). That's a sighted grader, useful
for the original paper's questions but NOT what our brief's Stage 1 needs:

    "Input: the problem statement + the student's solution step. (No ground
    truth -- it has to judge like a real tutor would.)"

A single LLM tutor that's shown the answer key will trivially catch every
error -- that erases the exact failure mode (over-validation of wrong answers)
we're trying to measure. So BLIND_TUTOR_PROMPT keeps the same JSON-schema
style and rule vocabulary as the original teacher prompt, but removes
CORRECT_STEP and asks the model to work out correctness on its own, the way a
real single-agent tutor deployed in production would have to.
"""

import yaml
from pathlib import Path

_YAML_PATH = Path(__file__).parent /"prompts_i+2" / "student_prompts.yaml"


def _load_yaml_prompts():
    with open(_YAML_PATH, "r") as f:
        return yaml.safe_load(f)


_PROMPTS = _load_yaml_prompts()


def _render(entry, variables):
    """Turn a prompt-yaml entry + variables dict into one plain-text prompt."""
    parts = [entry.get("role", "")]
    if entry.get("Step_by_Step_Instructions"):
        parts.append("Instructions:\n" + entry["Step_by_Step_Instructions"])
    if entry.get("Constraints"):
        parts.append("Constraints:\n" + entry["Constraints"])
    if entry.get("Response_Format"):
        parts.append(
            "Respond with ONLY a single valid JSON object (no markdown fences, "
            "no extra text) in exactly this format:\n" + entry["Response_Format"]
        )
    text = "\n\n".join(p for p in parts if p)
    for key, val in variables.items():
        text = text.replace("{" + key + "}", str(val))
    return text


def student_prompt(givens, intermediates, conclusion):
    """Original student_prompt from the repo, used unmodified."""
    entry = _PROMPTS["student_prompt"]
    return _render(entry, {
        "givens": givens,
        "intermediates": intermediates,
        "conclusion": conclusion,
    })


# ---------------------------------------------------------------------------
# New: blind tutor prompt (Stage 1 of the brief's pipeline)
# ---------------------------------------------------------------------------
_BLIND_TUTOR_ENTRY = {
    "role": (
        "You are a Tutor grading an undergraduate Discrete Structures student's "
        "propositional-logic proof step. You do NOT have access to an answer "
        "key. You must work out for yourself, from the GIVENS, INTERMEDIATE_STEPS "
        "and CONCLUSION, whether the student's NEXT_STEP is a valid logical "
        "derivation and whether it makes good progress toward the CONCLUSION."
    ),
    "Step_by_Step_Instructions": (
        "1. From GIVENS and INTERMEDIATE_STEPS, work out what next steps would "
        "be validly derivable using standard propositional-logic rules.\n"
        "2. Check whether the student's NEXT_STEP is one of those valid "
        "derivations, and whether the RULE and PARENT_STATEMENTS they cite "
        "actually justify it.\n"
        "3. Judge whether NEXT_STEP is Correct (a valid, useful derivation), "
        "Suboptimal (a valid derivation but not the most useful one), or "
        "Incorrect (not a valid derivation, or wrong rule/parents cited).\n"
        "4. Give brief feedback justifying your verdict."
    ),
    "Constraints": (
        "- VERDICT: Only \"Correct\", \"Suboptimal\", or \"Incorrect\".\n"
        "- Do not invent an answer key; reason only from GIVENS, "
        "INTERMEDIATE_STEPS, CONCLUSION, and standard inference rules.\n"
        "- Rules vocabulary: MP (Modus Ponens), MT (Modus Tollens), Conj "
        "(Conjunction), Add (Addition), DS (Disjunctive Syllogism), HS "
        "(Hypothetical Syllogism), DeM (De Morgan's Laws), Impl (Implication), "
        "Simp (Simplification), Dist (Distribution), Assoc (Associativity), "
        "CP (Contraposition), Com (Commutation), CD (Constructive Dilemma), "
        "DN (Double Negation).\n"
        "- FEEDBACK: 2-3 sentences max, explain your reasoning for the verdict."
    ),
    "Response_Format": (
        '{\n'
        '  "VERDICT": "Correct/Suboptimal/Incorrect",\n'
        '  "CONFIDENCE": 0.0,\n'
        '  "FEEDBACK": "brief explanation of the verdict"\n'
        '}'
    ),
}


def blind_tutor_prompt(givens, intermediates, conclusion, student_next_step,
                        student_rule, student_parents):
    variables = {
        "givens": givens,
        "intermediates": intermediates,
        "conclusion": conclusion,
    }
    text = _render(_BLIND_TUTOR_ENTRY, variables)
    text += (
        f"\n\nGIVENS: {givens}"
        f"\nINTERMEDIATE_STEPS: {intermediates}"
        f"\nCONCLUSION: {conclusion}"
        f"\n\nSTUDENT'S SUBMITTED STEP:"
        f"\nNEXT_STEP: {student_next_step}"
        f"\nRULE: {student_rule}"
        f"\nPARENT_STATEMENTS: {student_parents}"
    )
    return text


# ---------------------------------------------------------------------------
# New: independent Verifier prompt (Stage 2 of the pipeline)
# ---------------------------------------------------------------------------
# Deliberately worded as a SEPARATE persona (senior TA doing spot-audits,
# not "a second tutor") and never shown the Tutor's verdict. This matters:
# per the README's conformity-bias discussion, if the Verifier saw the
# Tutor's output first it would anchor on it. It must commit to its own
# judgment from GIVENS/INTERMEDIATE_STEPS/CONCLUSION alone, exactly like the
# Tutor does, but independently -- the Compare node is what checks agreement
# afterward.
_VERIFIER_ENTRY = {
    "role": (
        "You are a senior teaching assistant conducting an independent "
        "spot-audit of a propositional-logic proof step. Nobody has graded "
        "this step yet -- you are forming the FIRST judgment on it, from "
        "scratch. You do NOT have access to an answer key or to any other "
        "grader's opinion. You must work out for yourself, from the GIVENS, "
        "INTERMEDIATE_STEPS and CONCLUSION, whether the submitted NEXT_STEP "
        "is a valid logical derivation and whether it makes good progress "
        "toward the CONCLUSION."
    ),
    "Step_by_Step_Instructions": (
        "1. Independently work out, from GIVENS and INTERMEDIATE_STEPS, what "
        "next steps would be validly derivable using standard "
        "propositional-logic rules -- do this before looking at what was "
        "submitted.\n"
        "2. Check whether the submitted NEXT_STEP is one of those valid "
        "derivations, and whether RULE and PARENT_STATEMENTS actually "
        "justify it.\n"
        "3. Judge whether NEXT_STEP is Correct (a valid, useful derivation), "
        "Suboptimal (a valid derivation but not the most useful one), or "
        "Incorrect (not a valid derivation, or wrong rule/parents cited).\n"
        "4. Give brief feedback justifying your verdict."
    ),
    "Constraints": (
        "- VERDICT: Only \"Correct\", \"Suboptimal\", or \"Incorrect\".\n"
        "- Do not invent an answer key; reason only from GIVENS, "
        "INTERMEDIATE_STEPS, CONCLUSION, and standard inference rules.\n"
        "- Be skeptical: audits exist to catch errors a first grader might "
        "miss, especially steps that look plausible but use an invalid rule "
        "or the wrong parent statements.\n"
        "- Rules vocabulary: MP (Modus Ponens), MT (Modus Tollens), Conj "
        "(Conjunction), Add (Addition), DS (Disjunctive Syllogism), HS "
        "(Hypothetical Syllogism), DeM (De Morgan's Laws), Impl (Implication), "
        "Simp (Simplification), Dist (Distribution), Assoc (Associativity), "
        "CP (Contraposition), Com (Commutation), CD (Constructive Dilemma), "
        "DN (Double Negation).\n"
        "- FEEDBACK: 2-3 sentences max, explain your reasoning for the verdict."
    ),
    "Response_Format": (
        '{\n'
        '  "VERDICT": "Correct/Suboptimal/Incorrect",\n'
        '  "CONFIDENCE": 0.0,\n'
        '  "FEEDBACK": "brief explanation of the verdict"\n'
        '}'
    ),
}


def verifier_prompt(givens, intermediates, conclusion, student_next_step,
                     student_rule, student_parents):
    """Independent audit prompt. Deliberately does NOT take a tutor_verdict
    argument -- there is nothing to pass, by design. The Verifier only ever
    sees the problem + the student's step, same inputs as the Tutor, never
    the Tutor's output."""
    variables = {
        "givens": givens,
        "intermediates": intermediates,
        "conclusion": conclusion,
    }
    text = _render(_VERIFIER_ENTRY, variables)
    text += (
        f"\n\nGIVENS: {givens}"
        f"\nINTERMEDIATE_STEPS: {intermediates}"
        f"\nCONCLUSION: {conclusion}"
        f"\n\nSUBMITTED STEP:"
        f"\nNEXT_STEP: {student_next_step}"
        f"\nRULE: {student_rule}"
        f"\nPARENT_STATEMENTS: {student_parents}"
    )
    return text


# ---------------------------------------------------------------------------
# New: Recovery tie-break prompt (used only when Tutor and Verifier disagree)
# ---------------------------------------------------------------------------
_RECOVERY_ENTRY = {
    "role": (
        "You are a senior reviewer resolving a disagreement between two "
        "independent graders of a propositional-logic proof step. Grader A "
        "and Grader B reached different verdicts on the same step, each "
        "without seeing the other's opinion. You must weigh both of their "
        "verdicts and reasoning against the GIVENS, INTERMEDIATE_STEPS and "
        "CONCLUSION, and decide the final, tie-broken verdict yourself."
    ),
    "Step_by_Step_Instructions": (
        "1. Independently re-derive what next steps would be validly "
        "derivable from GIVENS and INTERMEDIATE_STEPS.\n"
        "2. Read Grader A's and Grader B's verdicts and feedback.\n"
        "3. Decide which grader (if either) is correct, or reach your own "
        "verdict if both are wrong.\n"
        "4. State the FINAL_VERDICT and briefly explain why."
    ),
    "Constraints": (
        "- FINAL_VERDICT: Only \"Correct\", \"Suboptimal\", or \"Incorrect\".\n"
        "- Do not simply average or default to one grader; resolve the "
        "disagreement on the logical merits.\n"
        "- Rules vocabulary: MP (Modus Ponens), MT (Modus Tollens), Conj "
        "(Conjunction), Add (Addition), DS (Disjunctive Syllogism), HS "
        "(Hypothetical Syllogism), DeM (De Morgan's Laws), Impl (Implication), "
        "Simp (Simplification), Dist (Distribution), Assoc (Associativity), "
        "CP (Contraposition), Com (Commutation), CD (Constructive Dilemma), "
        "DN (Double Negation)."
    ),
    "Response_Format": (
        '{\n'
        '  "FINAL_VERDICT": "Correct/Suboptimal/Incorrect",\n'
        '  "REASONING": "brief explanation of how the disagreement was resolved"\n'
        '}'
    ),
}


def recovery_prompt(givens, intermediates, conclusion, student_next_step,
                     student_rule, student_parents,
                     tutor_verdict, tutor_feedback,
                     verifier_verdict, verifier_feedback):
    variables = {
        "givens": givens,
        "intermediates": intermediates,
        "conclusion": conclusion,
    }
    text = _render(_RECOVERY_ENTRY, variables)
    text += (
        f"\n\nGIVENS: {givens}"
        f"\nINTERMEDIATE_STEPS: {intermediates}"
        f"\nCONCLUSION: {conclusion}"
        f"\n\nSUBMITTED STEP:"
        f"\nNEXT_STEP: {student_next_step}"
        f"\nRULE: {student_rule}"
        f"\nPARENT_STATEMENTS: {student_parents}"
        f"\n\nGRADER A (Tutor) VERDICT: {tutor_verdict}"
        f"\nGRADER A (Tutor) FEEDBACK: {tutor_feedback}"
        f"\n\nGRADER B (Verifier) VERDICT: {verifier_verdict}"
        f"\nGRADER B (Verifier) FEEDBACK: {verifier_feedback}"
    )
    return text
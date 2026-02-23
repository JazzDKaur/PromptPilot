import streamlit as st
import pandas as pd
import re
from groq import Groq
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime
from typing import Dict, Any, List, Optional

# ---------------------------------
# Config
# ---------------------------------
MAX_ATTEMPTS = 5

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="PromptPilot", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-bottom: 80px !important; }
      section[data-testid="stSidebar"] { height: 100vh; }
      .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# Professional Banner
# ---------------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #7C2AE8 0%, #0EA5EA 100%);
        padding: 22px 14px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 18px;
    ">
        <h1 style="margin:0;">PromptPilot</h1>
        <h4 style="margin:6px 0 0 0; font-weight:400;">
            An Interactive AI Prompt Writing Training & Evaluation Platform
        </h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# Initialize Groq Client (safe)
# ---------------------------------
def get_groq_client() -> Optional[Groq]:
    try:
        key = st.secrets.get("GROQ_API_KEY", None)
        if not key:
            return None
        return Groq(api_key=key)
    except Exception:
        return None

client = get_groq_client()

# ---------------------------------
# Groq call helper (safe)
# ---------------------------------
def groq_complete(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.0) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY missing. Add it to Streamlit secrets.")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------
# Load Dataset (robust)
# ---------------------------------
@st.cache_data
def load_usecases(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

DATA_PATH = "PromptUseCases.xlsx"
try:
    usecases_df = load_usecases(DATA_PATH)
except Exception:
    st.error(f"❌ Could not load dataset at: {DATA_PATH}")
    st.info("Fix: Ensure the file exists in your repo and the path matches exactly.")
    st.stop()

# ---------------------------------
# Column mapping (fallback-friendly)
# ---------------------------------
def pick_col(df: pd.DataFrame, candidates: List[str], contains: List[str] = None) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if contains:
        for col in cols:
            low = col.lower()
            if any(k.lower() in low for k in contains):
                return col
    return None

SCENARIO_COL = pick_col(usecases_df, ["Scenario / Case Study", "Scenario", "Case Study"], contains=["scenario", "case"])
SECTOR_COL = pick_col(usecases_df, ["Sector"], contains=["sector"])
MODULE_COL = pick_col(usecases_df, ["Module / Department", "Module", "Department"], contains=["module", "department"])
EXPECTED_COL = pick_col(usecases_df, ["Expected Prompt", "Sample Prompt", "Ideal Prompt"], contains=["expected", "sample", "ideal", "prompt"])

if not SCENARIO_COL:
    st.error("❌ Could not find the Scenario column in your Excel file.")
    st.info("Expected a column like: 'Scenario / Case Study' or containing the word 'Scenario'.")
    st.stop()

# ---------------------------------
# Session State Init
# ---------------------------------
if "attempt_count" not in st.session_state:
    st.session_state.attempt_count = 0
if "show_correct_prompt" not in st.session_state:
    st.session_state.show_correct_prompt = False
if "scenario_row" not in st.session_state:
    st.session_state.scenario_row = usecases_df.sample(1).iloc[0]
if "last_evaluation_text" not in st.session_state:
    st.session_state.last_evaluation_text = ""
if "last_analysis_text" not in st.session_state:
    st.session_state.last_analysis_text = ""
if "last_rubric_rows" not in st.session_state:
    st.session_state.last_rubric_rows = []
if "last_overall_raw" not in st.session_state:
    st.session_state.last_overall_raw = ""
if "latest_overall_100" not in st.session_state:
    st.session_state.latest_overall_100 = None
if "latest_overall_raw" not in st.session_state:
    st.session_state.latest_overall_raw = ""
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""

if "last_relevance" not in st.session_state:
    st.session_state.last_relevance = {"relevant": None, "match_score": None, "reason": ""}
if "last_usability" not in st.session_state:
    st.session_state.last_usability = {"usable": None, "score": None, "reason": ""}

if "last_blocked" not in st.session_state:
    st.session_state.last_blocked = False
if "last_block_reason" not in st.session_state:
    st.session_state.last_block_reason = ""

if "history" not in st.session_state:
    st.session_state.history = []

# cache sample prompt (avoid re-generation on reruns)
if "sample_prompt_cached" not in st.session_state:
    st.session_state.sample_prompt_cached = {}  # key: scenario_text -> prompt

# cache quality rubric (NEW - should run ALWAYS)
if "quality_rubric_cached" not in st.session_state:
    st.session_state.quality_rubric_cached = None  # dict

# ---------------------------------
# Reset helpers
# ---------------------------------
def reset_for_new_scenario():
    st.session_state.attempt_count = 0
    st.session_state.show_correct_prompt = False

    st.session_state.last_evaluation_text = ""
    st.session_state.last_analysis_text = ""
    st.session_state.last_rubric_rows = []
    st.session_state.last_overall_raw = ""
    st.session_state.latest_overall_100 = None
    st.session_state.latest_overall_raw = ""

    st.session_state.last_relevance = {"relevant": None, "match_score": None, "reason": ""}
    st.session_state.last_usability = {"usable": None, "score": None, "reason": ""}

    st.session_state.last_blocked = False
    st.session_state.last_block_reason = ""

    st.session_state.user_prompt = ""
    st.session_state.quality_rubric_cached = None

def next_scenario():
    st.session_state.scenario_row = usecases_df.sample(1).iloc[0]
    reset_for_new_scenario()

# ---------------------------------
# Speedometer
# ---------------------------------
def show_score_speedometer(score_0_to_100: int, title="Overall Score"):
    if score_0_to_100 is None:
        return

    if score_0_to_100 > 80:
        bar_color = "green"
    elif score_0_to_100 >= 50:
        bar_color = "orange"
    else:
        bar_color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_0_to_100,
        number={"suffix": " / 100"},
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": bar_color}}
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Parsing Helpers (Rubrics out of 20)
# ---------------------------------
RUBRICS = ["Clarity", "Completeness", "Context", "Role", "Output Format"]

def _normalize_eval_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("**", "").replace("__", "").replace("`", "")
    t = t.replace("•", "- ").replace("◦", "- ")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def _force_newlines_for_sections(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"(?i)\s+(Clarity|Completeness|Context|Role|Output\s*Format)\s*:", r"\n\1:", t)
    t = re.sub(r"(?i)\s+(OVERALL\s*[_\-\s]*SCORE)\s*:", r"\n\1:", t)
    t = re.sub(r"(?i)\s+(STRENGTHS)\b", r"\n\1", t)
    t = re.sub(r"(?i)\s+(IMPROVEMENT[_\-\s]*AREAS)\b", r"\n\1", t)
    t = re.sub(r"(?i)\s+(Scenario)\s*:", r"\n\1:", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def parse_overall_score(evaluation_text: str):
    t = _force_newlines_for_sections(_normalize_eval_text(evaluation_text))
    m = re.search(r"OVERALL\s*[_\-\s]*SCORE\s*[:\-]?\s*(\d+)\s*/\s*(\d+)", t, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def parse_rubric_lines(evaluation_text: str):
    t = _force_newlines_for_sections(_normalize_eval_text(evaluation_text))
    line_pattern = re.compile(
        r"^(Clarity|Completeness|Context|Role|Output\s*Format)\s*:\s*(\d+)\s*/\s*(\d+)\s*(?:\((.*?)\))?\s*$",
        flags=re.IGNORECASE | re.MULTILINE
    )

    found = []
    for m in line_pattern.finditer(t):
        label = re.sub(r"\s+", " ", m.group(1).strip())
        if re.match(r"output\s*format", label, flags=re.IGNORECASE):
            label = "Output Format"
        else:
            label = label[0].upper() + label[1:].lower()

        score = int(m.group(2))
        mx = int(m.group(3))
        fb = (m.group(4) or "").strip() or "-"

        found.append({"rubric": label, "score": score, "max": mx, "feedback": fb})

    inferred_max = 20
    for r in found:
        if isinstance(r.get("max"), int):
            inferred_max = r["max"]
            break

    by_name = {r["rubric"]: r for r in found}
    ordered = []
    for rname in RUBRICS:
        ordered.append(by_name.get(rname, {"rubric": rname, "score": None, "max": inferred_max, "feedback": "-"}))
    return ordered

def overall_to_100(obtained: int, mx: int):
    if obtained is None or mx in (None, 0):
        return None
    return int(round((obtained / mx) * 100))

def color_for_score(score, mx: int):
    if score is None:
        return "#f2f2f2"
    pct = (score / mx) * 100 if mx else 0
    if pct >= 80:
        return "#d4edda"
    if pct >= 50:
        return "#fff3cd"
    return "#f8d7da"

def render_colored_score_table(rows):
    if not rows:
        st.info("Could not extract rubric scores.")
        return

    html = """
    <div style="width:100%; overflow-x:auto;">
      <table style="border-collapse:collapse; width:100%; font-family:Arial, sans-serif;">
        <thead>
          <tr style="background:#f2f2f2;">
            <th style="border:1px solid #ddd; padding:10px; text-align:left;">Rubric</th>
            <th style="border:1px solid #ddd; padding:10px; text-align:left;">Score</th>
            <th style="border:1px solid #ddd; padding:10px; text-align:left;">Feedback</th>
          </tr>
        </thead>
        <tbody>
    """
    for r in rows:
        bg = color_for_score(r["score"], r["max"])
        score_txt = "-" if r["score"] is None else f'{r["score"]}/{r["max"]}'
        fb = r["feedback"] if r["feedback"] else "-"
        html += f"""
          <tr style="background:{bg}; vertical-align:top;">
            <td style="border:1px solid #ddd; padding:10px; font-weight:600; width:22%;">{r["rubric"]}</td>
            <td style="border:1px solid #ddd; padding:10px; width:12%;">{score_txt}</td>
            <td style="border:1px solid #ddd; padding:10px; width:66%;">{fb}</td>
          </tr>
        """
    html += """
        </tbody>
      </table>
    </div>
    """
    height = min(420, 120 + (len(rows) * 70))
    components.html(html, height=height, scrolling=True)

def extract_strengths_improvements(evaluation_text: str):
    t = _force_newlines_for_sections(_normalize_eval_text(evaluation_text))

    m_strengths = re.search(
        r"(STRENGTHS\s*[\s\S]*?)(?=\n\s*IMPROVEMENT[_\-\s]*AREAS|\Z)",
        t,
        flags=re.IGNORECASE
    )
    m_improve = re.search(
        r"(IMPROVEMENT[_\-\s]*AREAS\s*[\s\S]*?)(?=\n\s*Scenario\s*:|\Z)",
        t,
        flags=re.IGNORECASE
    )

    strengths = m_strengths.group(1).strip() if m_strengths else "STRENGTHS\n- (Not found)"
    improvements = m_improve.group(1).strip() if m_improve else "IMPROVEMENT_AREAS\n- (Not found)"
    return strengths, improvements

# ---------------------------------
# Relevance Gate
# ---------------------------------
def check_prompt_relevance(scenario: str, user_prompt: str) -> Dict[str, Any]:
    gate_prompt = f"""
You are a strict relevance checker.

Return EXACTLY in this format (3 lines only):
RELEVANT: YES or NO
MATCH_SCORE: <0-100>
REASON: <one short sentence>

Scenario:
{scenario}

Learner Prompt:
{user_prompt}
"""
    try:
        text = groq_complete(gate_prompt, temperature=0.0)
    except Exception as e:
        return {"relevant": None, "match_score": None, "reason": f"API_ERROR: {e}", "raw": ""}

    relevant = None
    match_score = None
    reason = ""

    m1 = re.search(r"RELEVANT\s*:\s*(YES|NO)", text, flags=re.IGNORECASE)
    if m1:
        relevant = (m1.group(1).upper() == "YES")

    m2 = re.search(r"MATCH_SCORE\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    if m2:
        match_score = int(m2.group(1))

    m3 = re.search(r"REASON\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if m3:
        reason = m3.group(1).strip()

    return {"relevant": relevant, "match_score": match_score, "reason": reason, "raw": text}

# ---------------------------------
# Usability Gate (heuristic + fallback)
# ---------------------------------
ACTION_VERBS = {
    "list", "identify", "explain", "describe", "provide", "generate", "write", "analyze",
    "compare", "summarize", "outline", "create", "suggest", "recommend", "draft", "compose",
    "extract", "classify", "prioritize", "rank", "format", "produce", "prepare"
}
LABEL_KEYS = {"role", "tone", "context", "format", "output", "audience", "task", "constraints", "deliverable"}

def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))

def _has_action_instruction(prompt: str) -> bool:
    if not prompt:
        return False
    t = re.sub(r"\s+", " ", prompt.strip().lower())
    words = re.findall(r"[a-zA-Z']+", t)
    if len(words) < 4:
        return False
    first_chunk = words[:20]
    if any(w in ACTION_VERBS for w in first_chunk):
        return True
    if len(first_chunk) >= 2 and first_chunk[0] == "please" and first_chunk[1] in ACTION_VERBS:
        return True
    return False

def _looks_like_label_checklist(prompt: str) -> bool:
    if not prompt:
        return True

    t = prompt.strip()
    t_lower = t.lower()
    lines = [ln.strip() for ln in t_lower.splitlines() if ln.strip()]
    wc = _word_count(t)

    if wc >= 30 and _has_action_instruction(t):
        return False

    colon_lines = sum(1 for ln in lines if ":" in ln)
    label_like_lines = 0
    for ln in lines:
        if ":" in ln:
            key = ln.split(":", 1)[0].strip()
            if key in LABEL_KEYS:
                label_like_lines += 1

    filler_words = {"correct", "clear", "complete", "good", "formal", "informal", "polite", "nice"}

    if wc < 20:
        if colon_lines >= 2:
            return True
        if len(lines) >= 2 and all((ln in filler_words) or (ln.split(":")[0] in LABEL_KEYS) for ln in lines):
            return True

    if 20 <= wc < 30:
        if (label_like_lines >= 2 or colon_lines >= 4) and not _has_action_instruction(t):
            return True

    return False

def check_prompt_usability(scenario: str, user_prompt: str) -> Dict[str, Any]:
    p = (user_prompt or "").strip()

    if _looks_like_label_checklist(p):
        return {
            "usable": False,
            "score": 0,
            "reason": "Add ONE clear task line (e.g., 'Summarize...', 'Analyze...', 'Generate...') at the top.",
            "raw": "HEURISTIC_BLOCK"
        }

    if _has_action_instruction(p):
        base_score = 80 if _word_count(p) >= 20 else 65
        return {"usable": True, "score": base_score, "reason": "Contains a clear instruction/task for the AI to perform.", "raw": "HEURISTIC_PASS"}

    gate_prompt = f"""
You are a strict prompt usability checker.

IMPORTANT:
- Structured prompts with sections like Role:, Context:, Constraints:, Output Format: ARE USABLE
  if they include a clear instruction/task (e.g., "Summarize...", "Generate...", "Analyze...").
- Reject only prompts that are mostly labels/keywords and do not ask the AI to DO anything.

Return EXACTLY in this format (3 lines only):
USABLE: YES or NO
USABILITY_SCORE: <0-100>
REASON: <one short sentence>

Scenario:
{scenario}

Learner Prompt:
{p}
"""
    try:
        text = groq_complete(gate_prompt, temperature=0.0)
    except Exception as e:
        return {"usable": None, "score": None, "reason": f"API_ERROR: {e}", "raw": ""}

    usable = None
    score = None
    reason = ""

    m1 = re.search(r"USABLE\s*:\s*(YES|NO)", text, flags=re.IGNORECASE)
    if m1:
        usable = (m1.group(1).upper() == "YES")

    m2 = re.search(r"USABILITY_SCORE\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    if m2:
        score = int(m2.group(1))

    m3 = re.search(r"REASON\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if m3:
        reason = m3.group(1).strip()

    if usable is None:
        usable = False
    if score is None:
        score = 0 if not usable else 60
    if not reason:
        reason = "Could not determine usability clearly."

    return {"usable": usable, "score": score, "reason": reason, "raw": text}

# ---------------------------------
# Strict LLM Evaluation (only if relevant + usable)
# ---------------------------------
def evaluate_prompt(sector: str, module: str, scenario: str, user_prompt: str) -> str:
    eval_prompt = f"""
You are an extremely strict evaluator.

Scoring Rules (IMPORTANT):
- If the prompt is vague, generic, or missing role/context/output format → give LOW scores (0–8).
- If prompt is incomplete or unclear → OVERALL_SCORE must be below 50.
- Only detailed, specific, well-structured prompts can score above 80.
- Do NOT inflate scores.

Return EXACTLY in this template (same labels, same order).
Rules:
- Each rubric MUST be on its OWN LINE.
- Feedback MUST be inside parentheses.
- Use integers only.

Clarity: <0-20>/20 (<one short line>)
Completeness: <0-20>/20 (<one short line>)
Context: <0-20>/20 (<one short line>)
Role: <0-20>/20 (<one short line>)
Output Format: <0-20>/20 (<one short line>)

OVERALL_SCORE: <0-100>/100

STRENGTHS
- <bullet 1>
- <bullet 2>

IMPROVEMENT_AREAS
- <bullet 1>
- <bullet 2>

Scenario:
{scenario}

Learner Prompt:
{user_prompt}
"""
    return groq_complete(eval_prompt, temperature=0.0)

def analyze_prompt_components(user_prompt: str) -> str:
    analysis_prompt = f"""
Analyze the following prompt.

Extract:
1. Role specified
2. Context specified
3. Output format specified
4. Why this prompt is clear or incomplete

Prompt:
{user_prompt}

Return in bullet points.
"""
    return groq_complete(analysis_prompt, temperature=0.0)

def generate_correct_prompt(scenario: str) -> str:
    ideal_prompt = f"""
Write a perfect AI prompt for the following scenario.

Scenario:
{scenario}

Include:
- Role
- Context
- Constraints
- Output format
"""
    return groq_complete(ideal_prompt, temperature=0.0)

# ---------------------------------
# NEW: Prompt Quality Rubric (ALWAYS runs) + Average
# ---------------------------------
QUALITY_CRITERIA = [
    "Clarity",
    "Specificity",
    "Context",
    "Role Definition",
    "Audience Awareness",
    "Output Format",
    "Constraints",
    "Depth of Thinking",
    "Reusability",
    "Alignment to Goal",
]

def evaluate_quality_rubric(scenario: str, user_prompt: str) -> Dict[str, Any]:
    rubric_prompt = f"""
You are an expert prompt-engineering assessor.

Score the learner prompt on 10 criteria using a 1–5 scale:
1=Poor/Very Weak, 2=Basic, 3=Adequate, 4=Strong, 5=Excellent/Expert-Level.

Return EXACTLY in this strict format:
- Provide 10 lines ONLY for criteria + 1 line for TOTAL + 1 line for BAND
- No extra text.

Format:
Clarity: <1-5> - <short reason>
Specificity: <1-5> - <short reason>
Context: <1-5> - <short reason>
Role Definition: <1-5> - <short reason>
Audience Awareness: <1-5> - <short reason>
Output Format: <1-5> - <short reason>
Constraints: <1-5> - <short reason>
Depth of Thinking: <1-5> - <short reason>
Reusability: <1-5> - <short reason>
Alignment to Goal: <1-5> - <short reason>
TOTAL: <10-50>
BAND: <Basic/Casual Prompting | Structured Prompting | Advanced Prompt Engineering | Expert-Level Prompt Design>

Band rules:
10–20 = Basic / Casual Prompting
21–35 = Structured Prompting
36–45 = Advanced Prompt Engineering
46–50 = Expert-Level Prompt Design

Scenario:
{scenario}

Learner Prompt:
{user_prompt}
"""
    raw = groq_complete(rubric_prompt, temperature=0.0)

    rows = []
    total = None
    band = None

    for crit in QUALITY_CRITERIA:
        m = re.search(rf"^{re.escape(crit)}\s*:\s*([1-5])\s*-\s*(.+)$", raw, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            rows.append({"criterion": crit, "score": int(m.group(1)), "reason": m.group(2).strip()})
        else:
            rows.append({"criterion": crit, "score": None, "reason": "-"})

    mt = re.search(r"^TOTAL\s*:\s*(\d+)\s*$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if mt:
        total = int(mt.group(1))

    mb = re.search(r"^BAND\s*:\s*(.+)\s*$", raw, flags=re.IGNORECASE | re.MULTILINE)
    if mb:
        band = mb.group(1).strip()

    # compute total if missing
    if total is None:
        scores = [r["score"] for r in rows if isinstance(r.get("score"), int)]
        if len(scores) == 10:
            total = sum(scores)

    # compute average (always as total/10 if total exists)
    avg = round(total / 10, 2) if isinstance(total, int) else None

    # infer band if missing
    if band is None and isinstance(total, int):
        if 10 <= total <= 20:
            band = "Basic / Casual Prompting"
        elif 21 <= total <= 35:
            band = "Structured Prompting"
        elif 36 <= total <= 45:
            band = "Advanced Prompt Engineering"
        elif 46 <= total <= 50:
            band = "Expert-Level Prompt Design"

    return {"rows": rows, "total": total, "avg": avg, "band": band, "raw": raw}

def render_quality_table(rows: List[Dict[str, Any]]):
    if not rows:
        st.info("No rubric results.")
        return

    def bg(score):
        if score is None:
            return "#f2f2f2"
        if score >= 4:
            return "#d4edda"
        if score == 3:
            return "#fff3cd"
        return "#f8d7da"

    html = """
    <div style="width:100%; overflow-x:auto;">
      <table style="border-collapse:collapse; width:100%; font-family:Arial, sans-serif;">
        <thead>
          <tr style="background:#f2f2f2;">
            <th style="border:1px solid #ddd; padding:10px; text-align:left;">Criterion</th>
            <th style="border:1px solid #ddd; padding:10px; text-align:left; width:110px;">Score (1–5)</th>
            <th style="border:1px solid #ddd; padding:10px; text-align:left;">Reason</th>
          </tr>
        </thead>
        <tbody>
    """
    for r in rows:
        s = r.get("score")
        html += f"""
          <tr style="background:{bg(s)}; vertical-align:top;">
            <td style="border:1px solid #ddd; padding:10px; font-weight:600;">{r.get("criterion","-")}</td>
            <td style="border:1px solid #ddd; padding:10px;">{s if s is not None else "-"}</td>
            <td style="border:1px solid #ddd; padding:10px;">{r.get("reason","-")}</td>
          </tr>
        """
    html += """
        </tbody>
      </table>
    </div>
    """
    height = min(520, 140 + (len(rows) * 55))
    components.html(html, height=height, scrolling=True)

# ---------------------------------
# Scenario Data
# ---------------------------------
scenario_row = st.session_state.scenario_row
scenario_text = str(scenario_row.get(SCENARIO_COL, "")).strip()
selected_sector = str(scenario_row.get(SECTOR_COL, "")).strip() if SECTOR_COL else ""
selected_module = str(scenario_row.get(MODULE_COL, "")).strip() if MODULE_COL else ""

expected_prompt_from_excel = ""
if EXPECTED_COL:
    try:
        val = scenario_row.get(EXPECTED_COL, "")
        expected_prompt_from_excel = "" if pd.isna(val) else str(val)
    except Exception:
        expected_prompt_from_excel = ""

# ---------------------------------
# Sidebar controls
# ---------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    st.caption("Scenario metadata")
    st.write(f"**Sector:** {selected_sector if selected_sector else '-'}")
    st.write(f"**Module:** {selected_module if selected_module else '-'}")

    st.divider()

    if st.button("➡️ Next Scenario (Anytime)"):
        next_scenario()
        st.rerun()

    if st.button("🔄 Reset Attempts (Same Scenario)"):
        reset_for_new_scenario()
        st.rerun()

    st.divider()
    st.caption("API status")
    if client is None:
        st.error("API_KEY not found in Streamlit secrets.")
        st.info("Add it in Streamlit Cloud → App → Settings → Secrets.")
    else:
        st.success("Client ready ✅")

# ---------------------------------
# Two-column layout (Scenario | Write Prompt)
# ---------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("📘 Scenario")
    st.markdown(
        f"<div style='padding:15px;background:#f0f8ff;border-radius:8px; white-space:pre-wrap;'>{scenario_text}</div>",
        unsafe_allow_html=True
    )

    if st.button("💡 Get Hint"):
        st.info(
            "Hint:\n"
            "- Start with one clear **task instruction** (e.g., Summarize / Analyze / Generate)\n"
            "- Define the **role**\n"
            "- Add **context** (industry, constraints, audience)\n"
            "- Specify **output format** (table, bullets, steps)\n"
            "- Add **constraints** (tone, length, exclusions)"
        )

with right:
    st.subheader("✍️ Write Your Prompt")
    st.text_area("Enter your prompt", height=200, key="user_prompt")
    evaluate_clicked = st.button("✅ Evaluate Prompt", disabled=(st.session_state.attempt_count >= MAX_ATTEMPTS))

# ---------------------------------
# Trigger Evaluation
# ---------------------------------
if evaluate_clicked:
    current_prompt = (st.session_state.user_prompt or "").strip()

    if not current_prompt:
        st.warning("Write a prompt first.")
    else:
        st.session_state.attempt_count += 1

        # Relevance + Usability checks (these still gate the strict 0–20 rubric)
        rel = check_prompt_relevance(scenario_text, current_prompt)
        st.session_state.last_relevance = {
            "relevant": rel.get("relevant"),
            "match_score": rel.get("match_score"),
            "reason": rel.get("reason") or rel.get("raw") or ""
        }

        usab = check_prompt_usability(scenario_text, current_prompt)
        st.session_state.last_usability = {
            "usable": usab.get("usable"),
            "score": usab.get("score"),
            "reason": usab.get("reason") or usab.get("raw") or ""
        }

        # Prompt breakdown analysis (best-effort)
        try:
            st.session_state.last_analysis_text = analyze_prompt_components(current_prompt)
        except Exception as e:
            st.session_state.last_analysis_text = f"- (API error in analysis) {e}"

        # ✅ QUALITY RUBRIC should run ALWAYS (best-effort) — irrespective of gates
        try:
            qr = evaluate_quality_rubric(scenario_text, current_prompt)
            st.session_state.quality_rubric_cached = qr
        except Exception as e:
            st.session_state.quality_rubric_cached = {"rows": [], "total": None, "avg": None, "band": None, "raw": f"API_ERROR: {e}"}

        hist_record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario_text,
            "attempt_no": st.session_state.attempt_count,
            "prompt": current_prompt,
            "relevant": st.session_state.last_relevance["relevant"],
            "match_score": st.session_state.last_relevance["match_score"],
            "relevance_reason": st.session_state.last_relevance["reason"],
            "usable": st.session_state.last_usability["usable"],
            "usability_score": st.session_state.last_usability["score"],
            "usability_reason": st.session_state.last_usability["reason"],
            "overall_raw": "",
            "overall_100": None,
            "rubrics": [],
            "evaluation_text": "",
            "quality_rubric": st.session_state.quality_rubric_cached,
        }

        allow_strict_scoring = (rel.get("relevant") is True) and (usab.get("usable") is True)

        if allow_strict_scoring:
            st.session_state.last_blocked = False
            st.session_state.last_block_reason = ""

            try:
                evaluation_text = evaluate_prompt(selected_sector, selected_module, scenario_text, current_prompt)
            except Exception as e:
                evaluation_text = f"API_ERROR: {e}"

            st.session_state.last_evaluation_text = evaluation_text

            rubric_rows = parse_rubric_lines(evaluation_text)
            overall_obt, overall_max = parse_overall_score(evaluation_text)
            latest_overall_100 = overall_to_100(overall_obt, overall_max)

            st.session_state.last_rubric_rows = rubric_rows
            st.session_state.last_overall_raw = f"{overall_obt}/{overall_max}" if overall_obt is not None else ""
            st.session_state.latest_overall_100 = latest_overall_100
            st.session_state.latest_overall_raw = st.session_state.last_overall_raw

            hist_record["overall_raw"] = st.session_state.last_overall_raw
            hist_record["overall_100"] = st.session_state.latest_overall_100
            hist_record["rubrics"] = rubric_rows
            hist_record["evaluation_text"] = evaluation_text

        else:
            st.session_state.last_blocked = True
            st.session_state.last_block_reason = "relevance" if rel.get("relevant") is not True else "usability"

            st.session_state.last_evaluation_text = ""
            st.session_state.last_rubric_rows = []
            st.session_state.last_overall_raw = ""
            st.session_state.latest_overall_100 = None
            st.session_state.latest_overall_raw = ""

        if st.session_state.attempt_count >= MAX_ATTEMPTS:
            st.session_state.show_correct_prompt = True

        st.session_state.history.append(hist_record)

# ---------------------------------
# Attempt Banner ABOVE Tabs
# ---------------------------------
has_any_attempt = (st.session_state.attempt_count > 0)
if has_any_attempt:
    st.markdown(
        f"""
        <div style="background:#e8f3ff; padding:12px 14px; border-radius:10px; margin-top:6px; margin-bottom:10px; border:1px solid #cfe6ff;">
            <b>Attempt:</b> {st.session_state.attempt_count} / {MAX_ATTEMPTS}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------
# Tabs
# ---------------------------------
tab_names = ["📊 Prompt Score", "🧠 Prompt Quality Rubric (1–5)", "🔍 Prompt Breakdown", "🕘 History"]
if st.session_state.show_correct_prompt:
    tab_names.append("✅ Sample Prompt 1")
    tab_names.append("📌 Sample Prompt 2")

tabs = st.tabs(tab_names)

# ---------------------------------
# TAB 1: Prompt Score (0–20 rubric, gated)
# ---------------------------------
with tabs[0]:
    if not has_any_attempt:
        st.info("Write a prompt and click **Evaluate Prompt** to see the score.")
    else:
        rel = st.session_state.last_relevance
        usab = st.session_state.last_usability

        if rel.get("match_score") is not None:
            st.caption(f"Relevance Match Score: {rel.get('match_score')}/100")

        if st.session_state.last_blocked:
            if st.session_state.last_block_reason == "relevance":
                st.error("❌ Prompt is NOT relevant to the scenario. Please rewrite your prompt.")
                if rel.get("reason"):
                    st.info(f"Reason: {rel.get('reason')}")
                st.warning("No rubric scoring shown because the prompt is off-topic.")
            else:
                st.error("❌ Prompt is not a usable instruction. Please rewrite.")
                if usab.get("score") is not None:
                    st.caption(f"Usability Score: {usab.get('score')}/100")
                if usab.get("reason"):
                    st.info(f"Tip: {usab.get('reason')}")
                st.warning("No rubric scoring shown because the prompt is not a clear instruction/task.")
        else:
            st.success("✅ Prompt passed relevance + usability checks. Rubric scoring is shown below.")
            st.markdown("## Evaluation of the Learner Prompt")

            c1, c2 = st.columns([1.4, 1])

            with c1:
                st.markdown("### SCORES")
                render_colored_score_table(st.session_state.last_rubric_rows)

                if st.session_state.last_overall_raw:
                    st.caption(f"Latest OVERALL_SCORE: {st.session_state.last_overall_raw}")
                else:
                    st.warning("Could not extract latest OVERALL_SCORE from the evaluation output.")

            with c2:
                if st.session_state.latest_overall_100 is not None:
                    show_score_speedometer(st.session_state.latest_overall_100, title="Latest Overall Score")
                else:
                    st.warning("Could not extract OVERALL_SCORE for the speedometer.")

# ---------------------------------
# TAB 2: Prompt Quality Rubric (1–5) — ALWAYS runs
# ---------------------------------
with tabs[1]:
    if not has_any_attempt:
        st.info("This rubric appears after you evaluate a prompt.")
    else:
        st.subheader("🧠 Prompt Quality Rubric (1–5 scale)")
        st.caption("Runs for every evaluated prompt (even if relevance/usability fails).")

        qr = st.session_state.quality_rubric_cached
        if not qr:
            st.info("No rubric result found yet.")
        else:
            total = qr.get("total")
            avg = qr.get("avg")
            band = qr.get("band")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Score (10–50)", total if isinstance(total, int) else "—")
            with c2:
                st.metric("Average (1–5)", avg if isinstance(avg, (int, float)) else "—")
            with c3:
                st.metric("Band", band if band else "—")

            st.markdown("### Criteria Scores")
            render_quality_table(qr.get("rows", []))

            with st.expander("Show raw AI rubric output"):
                st.code(qr.get("raw", ""), language="text")

# ---------------------------------
# TAB 3: Prompt Breakdown
# ---------------------------------
with tabs[2]:
    if not has_any_attempt:
        st.info("Prompt breakdown will appear after evaluation.")
    else:
        st.subheader("🔍 Your Prompt Breakdown")
        st.markdown(
            f"<div style='background:#eef;padding:12px;border-radius:8px; white-space:pre-wrap;'>{st.session_state.last_analysis_text}</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        rel = st.session_state.last_relevance
        usab = st.session_state.last_usability

        st.subheader("🧭 Scenario Relevance Check")
        if rel.get("relevant") is True:
            st.markdown(
                f"""
                <div style='background:#d4edda;padding:12px;border-radius:8px; border:1px solid #c3e6cb; white-space:pre-wrap;'>
                ✅ Relevant to scenario<br/>
                <b>Match Score:</b> {rel.get('match_score') if rel.get('match_score') is not None else '-'} / 100<br/>
                <b>Reason:</b> {rel.get('reason') if rel.get('reason') else '-'}
                </div>
                """,
                unsafe_allow_html=True
            )
        elif rel.get("relevant") is False:
            st.markdown(
                f"""
                <div style='background:#f8d7da;padding:12px;border-radius:8px; border:1px solid #f5c6cb; white-space:pre-wrap;'>
                ❌ Not relevant to scenario<br/>
                <b>Match Score:</b> {rel.get('match_score') if rel.get('match_score') is not None else '-'} / 100<br/>
                <b>Reason:</b> {rel.get('reason') if rel.get('reason') else '-'}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No relevance check data available yet (or API error).")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.subheader("🧩 Prompt Usability Check")
        if usab.get("usable") is True:
            st.markdown(
                f"""
                <div style='background:#d4edda;padding:12px;border-radius:8px; border:1px solid #c3e6cb; white-space:pre-wrap;'>
                ✅ Usable prompt (clear instruction)<br/>
                <b>Usability Score:</b> {usab.get('score') if usab.get('score') is not None else '-'} / 100<br/>
                <b>Reason:</b> {usab.get('reason') if usab.get('reason') else '-'}
                </div>
                """,
                unsafe_allow_html=True
            )
        elif usab.get("usable") is False:
            st.markdown(
                f"""
                <div style='background:#f8d7da;padding:12px;border-radius:8px; border:1px solid #f5c6cb; white-space:pre-wrap;'>
                ❌ Not a usable instruction prompt<br/>
                <b>Usability Score:</b> {usab.get('score') if usab.get('score') is not None else '-'} / 100<br/>
                <b>Tip:</b> {usab.get('reason') if usab.get('reason') else '-'}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No usability check data available yet (or API error).")

        if st.session_state.last_evaluation_text:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.subheader("💪 Strengths & 🎯 Improvement Areas (LLM Feedback)")
            strengths, improvements = extract_strengths_improvements(st.session_state.last_evaluation_text)

            st.markdown(
                f"""
                <div style='background:#fff3cd;padding:12px;border-radius:8px; border:1px solid #ffeeba; white-space:pre-wrap;'>
                {strengths}\n\n{improvements}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Strengths/Improvement feedback appears after a relevant + usable prompt is evaluated.")

# ---------------------------------
# TAB 4: History
# ---------------------------------
with tabs[3]:
    st.subheader("🕘 Attempt History (This Session)")
    if not st.session_state.history:
        st.info("No attempts yet. Your previous prompts will appear here after you evaluate.")
    else:
        for rec in reversed(st.session_state.history):
            status_rel = "✅ Relevant" if rec.get("relevant") else "❌ Not Relevant"
            status_use = "✅ Usable" if rec.get("usable") else "❌ Not Usable"
            score = rec.get("overall_raw") if rec.get("overall_raw") else "-"
            match = rec.get("match_score")
            match_txt = f"{match}/100" if match is not None else "-"
            us = rec.get("usability_score")
            us_txt = f"{us}/100" if us is not None else "-"

            qr = rec.get("quality_rubric") or {}
            qr_total = qr.get("total")
            qr_avg = qr.get("avg")
            qr_band = qr.get("band")

            qr_summary = ""
            if isinstance(qr_total, int) or isinstance(qr_avg, (int, float)) or qr_band:
                qr_summary = f" • Quality: {qr_total if qr_total is not None else '-'} (avg {qr_avg if qr_avg is not None else '-'}) • {qr_band if qr_band else '-'}"

            header = (
                f"Attempt {rec.get('attempt_no')} • {rec.get('time')} • "
                f"{status_rel} • {status_use} • Score: {score} • Match: {match_txt} • Usability: {us_txt}"
                f"{qr_summary}"
            )

            with st.expander(header, expanded=False):
                st.markdown("**Prompt:**")
                st.markdown(
                    f"<div style='background:#f7f7f7;padding:10px;border-radius:8px; white-space:pre-wrap;'>{rec.get('prompt','')}</div>",
                    unsafe_allow_html=True
                )

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown("**Relevance Reason:**")
                st.markdown(
                    f"<div style='background:#eef;padding:10px;border-radius:8px; white-space:pre-wrap;'>{rec.get('relevance_reason','-')}</div>",
                    unsafe_allow_html=True
                )

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown("**Usability Reason:**")
                st.markdown(
                    f"<div style='background:#eef;padding:10px;border-radius:8px; white-space:pre-wrap;'>{rec.get('usability_reason','-')}</div>",
                    unsafe_allow_html=True
                )

                if rec.get("rubrics"):
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    st.markdown("**Rubric Scores (0–20):**")
                    render_colored_score_table(rec["rubrics"])

                if qr and qr.get("rows"):
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    st.markdown("**Quality Rubric (1–5):**")
                    st.caption(f"Total: {qr.get('total','-')} • Average: {qr.get('avg','-')} • Band: {qr.get('band','-')}")
                    render_quality_table(qr.get("rows", []))

# ---------------------------------
# TAB 5: Sample Prompt 1 (cached)
# ---------------------------------
if st.session_state.show_correct_prompt:
    with tabs[4]:
        st.subheader("✅ Sample Prompt 1 (AI-generated)")
        scenario_key = scenario_text.strip()
        if scenario_key in st.session_state.sample_prompt_cached:
            correct_prompt = st.session_state.sample_prompt_cached[scenario_key]
        else:
            try:
                correct_prompt = generate_correct_prompt(scenario_text)
            except Exception as e:
                correct_prompt = f"API_ERROR: {e}"
            st.session_state.sample_prompt_cached[scenario_key] = correct_prompt

        st.markdown(
            f"<div style='background:#d0f0c0;padding:15px;border-radius:8px; white-space:pre-wrap;'>{correct_prompt}</div>",
            unsafe_allow_html=True
        )

# ---------------------------------
# TAB 6: Sample Prompt 2 (from Excel)
# ---------------------------------
if st.session_state.show_correct_prompt:
    with tabs[5]:
        st.subheader("📌 Sample Prompt 2 (From Excel)")
        if expected_prompt_from_excel.strip():
            st.markdown(
                f"<div style='background:#f7f7f7;padding:15px;border-radius:8px; border:1px solid #ddd; white-space:pre-wrap;'>{expected_prompt_from_excel}</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("No Sample Prompt found for this scenario (missing Expected Prompt column or empty value).")


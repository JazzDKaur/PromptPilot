import streamlit as st
import pandas as pd
import re
from groq import Groq
import plotly.graph_objects as go
import streamlit.components.v1 as components


st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #1f4e79, #2e75b6);
        padding: 22px 10px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 18px;
    ">
        <h1 style="margin-bottom:5px;">PromptPilot</h1>
        <h4 style="font-weight:400;">
            An Interactive AI Prompt Writing Training & Evaluation Platform
        </h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="PromptPilot", layout="wide")



# ---------------------------------
# Initialize Groq Client
# ---------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_usecases(file_path):
    return pd.read_excel(file_path)

usecases_df = load_usecases("Dataset/PromptUseCases.xlsx")

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

# Latest score for gauge
if "latest_overall_100" not in st.session_state:
    st.session_state.latest_overall_100 = None

if "latest_overall_raw" not in st.session_state:
    st.session_state.latest_overall_raw = ""

# Bind prompt input to session state
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""

# ---------------------------------
# Speedometer
# ---------------------------------
def show_score_speedometer(score_0_to_100: int, title="Overall Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_0_to_100,
        number={"suffix": " / 100"},
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# Parsing Helpers
# ---------------------------------
RUBRICS = ["Clarity", "Completeness", "Context", "Role", "Output Format"]

def _normalize_eval_text(text: str) -> str:
    if not text:
        return ""
    t = text
    t = t.replace("**", "").replace("__", "").replace("`", "")
    t = t.replace("•", "- ").replace("◦", "- ")
    # keep newlines; only collapse spaces/tabs
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def _force_newlines_for_sections(text: str) -> str:
    """
    ✅ Critical fix:
    If model prints "Clarity: 18/20 Completeness: 19/20 ..." in same line,
    force a newline before each rubric / section header so parsing becomes stable.
    """
    if not text:
        return ""

    t = text

    # Force newline before rubric labels (except at start)
    t = re.sub(r"(?i)\s+(Clarity|Completeness|Context|Role|Output\s*Format)\s*:", r"\n\1:", t)

    # Force newline before major sections
    t = re.sub(r"(?i)\s+(OVERALL\s*[_\-\s]*SCORE)\s*:", r"\n\1:", t)
    t = re.sub(r"(?i)\s+(STRENGTHS)\b", r"\n\1", t)
    t = re.sub(r"(?i)\s+(IMPROVEMENT[_\-\s]*AREAS)\b", r"\n\1", t)
    t = re.sub(r"(?i)\s+(Scenario)\s*:", r"\n\1:", t)

    # clean extra blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def parse_overall_score(evaluation_text: str):
    t = _force_newlines_for_sections(_normalize_eval_text(evaluation_text))
    m = re.search(r"OVERALL\s*[_\-\s]*SCORE\s*[:\-]?\s*(\d+)\s*/\s*(\d+)", t, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def parse_rubric_lines(evaluation_text: str):
    """
    ✅ FIXED PARSER:
    - First force newlines so each rubric is its own line.
    - Then parse per-line (MULTILINE) so feedback can't “eat” the next rubric.
    """
    t = _force_newlines_for_sections(_normalize_eval_text(evaluation_text))

    # Line-based pattern: rubric: 18/20 (feedback)
    line_pattern = re.compile(
        r"^(Clarity|Completeness|Context|Role|Output\s*Format)\s*:\s*(\d+)\s*/\s*(\d+)\s*(?:\((.*?)\))?\s*$",
        flags=re.IGNORECASE | re.MULTILINE
    )

    found = []
    for m in line_pattern.finditer(t):
        label = m.group(1).strip()
        label = re.sub(r"\s+", " ", label)
        if re.match(r"output\s*format", label, flags=re.IGNORECASE):
            label = "Output Format"
        else:
            label = label[0].upper() + label[1:].lower()

        score = int(m.group(2))
        mx = int(m.group(3))
        fb = (m.group(4) or "").strip()
        fb = fb if fb else "-"  # if model didn't provide parentheses feedback

        found.append({"rubric": label, "score": score, "max": mx, "feedback": fb})

    # infer max (fallback 20)
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
# LLM Calls
# ---------------------------------
def evaluate_prompt(sector, module, scenario, user_prompt):
    # ✅ Make feedback always in parentheses (line-based)
    eval_prompt = f"""
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
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

def analyze_prompt_components(user_prompt):
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
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

def generate_correct_prompt(scenario):
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
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": ideal_prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

# ---------------------------------
# UI
# ---------------------------------
#st.title("PromptPilot")
#st.markdown("Practice, Evaluate, and Improve Your AI Prompt Writing Skills")

scenario_row = st.session_state.scenario_row
scenario_text = scenario_row["Scenario / Case Study"]
selected_sector = scenario_row["Sector"]
selected_module = scenario_row["Module / Department"]

# ✅ NEW: Expected prompt from Excel col 4 (index 3)
expected_prompt_from_excel = ""
try:
    expected_prompt_from_excel = str(scenario_row.iloc[3]) if not pd.isna(scenario_row.iloc[3]) else ""
except Exception:
    expected_prompt_from_excel = ""

left, right = st.columns([1, 1])

with left:
    st.subheader("📘 Scenario")
    st.markdown(
        f"<div style='padding:15px;background:#f0f8ff;border-radius:8px'>{scenario_text}</div>",
        unsafe_allow_html=True
    )
    if st.button("💡 Get Hint"):
        st.info(
            "Hint:\n"
            "- Clearly define your **role**\n"
            "- Add **context** (industry, constraints, audience)\n"
            "- Specify **expected output format** (table, bullets, steps)"
        )

with right:
    st.subheader("✍️ Write Your Prompt")
    st.text_area("Enter your prompt", height=200, key="user_prompt")
    evaluate_clicked = st.button("✅ Evaluate Prompt", disabled=(st.session_state.attempt_count >= 2))

# ---------------------------------
# Trigger Evaluation
# ---------------------------------
if evaluate_clicked:
    current_prompt = (st.session_state.user_prompt or "").strip()

    if not current_prompt:
        st.warning("Write a prompt first.")
    else:
        st.session_state.attempt_count += 1

        evaluation_text = evaluate_prompt(selected_sector, selected_module, scenario_text, current_prompt)
        analysis_text = analyze_prompt_components(current_prompt)

        st.session_state.last_evaluation_text = evaluation_text
        st.session_state.last_analysis_text = analysis_text

        if st.session_state.attempt_count >= 2:
            st.session_state.show_correct_prompt = True

        rubric_rows = parse_rubric_lines(evaluation_text)
        overall_obt, overall_max = parse_overall_score(evaluation_text)
        latest_overall_100 = overall_to_100(overall_obt, overall_max)

        st.session_state.last_rubric_rows = rubric_rows
        st.session_state.last_overall_raw = f"{overall_obt}/{overall_max}" if overall_obt is not None else ""

        st.session_state.latest_overall_100 = latest_overall_100
        st.session_state.latest_overall_raw = st.session_state.last_overall_raw

# ---------------------------------
# Tabs
# ---------------------------------
has_result = bool(st.session_state.last_evaluation_text)

tab_names = ["📊 Score", "🔍 Performance Breakdown"]
if st.session_state.show_correct_prompt:
    tab_names.append("✅ Sample Prompt 1")
    tab_names.append("📌 Sample Prompt 2")  # ✅ NEW TAB

tabs = st.tabs(tab_names)

# TAB 1: Score
with tabs[0]:
    if not has_result:
        st.info("Write a prompt and click **Evaluate Prompt** to see the score.")
    else:
        st.markdown("## Evaluation of the Learner Prompt")
        c1, c2 = st.columns([1.4, 1])

        with c1:
            st.markdown("### SCORES")
            render_colored_score_table(st.session_state.last_rubric_rows)

            if st.session_state.last_overall_raw:
                st.caption(f"Latest OVERALL_SCORE: {st.session_state.last_overall_raw}")
            else:
                st.warning("Could not extract latest OVERALL_SCORE from the evaluation output.")

            st.info(f"Attempt: {st.session_state.attempt_count} / 2")

        with c2:
            if st.session_state.latest_overall_100 is not None:
                show_score_speedometer(st.session_state.latest_overall_100, title="Latest Overall Score")
            else:
                st.warning("Could not extract OVERALL_SCORE for the speedometer.")

# TAB 2: Performance Breakdown (Breakdown first, then strengths below)
with tabs[1]:
    if not has_result:
        st.info("Prompt breakdown will appear after evaluation.")
    else:
        st.subheader("🔍 Your Prompt Breakdown")
        st.markdown(
            f"<div style='background:#eef;padding:12px;border-radius:8px'>{st.session_state.last_analysis_text}</div>",
            unsafe_allow_html=True
        )

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

# TAB 3: Sample Prompt (after 2 attempts)
if st.session_state.show_correct_prompt:
    with tabs[2]:
        st.subheader("✅ Sample Prompt 1" )
        correct_prompt = generate_correct_prompt(scenario_text)
        st.markdown(
            f"<div style='background:#d0f0c0;padding:15px;border-radius:8px; white-space:pre-wrap;'>{correct_prompt}</div>",
            unsafe_allow_html=True
        )

# TAB 4: Expected Prompt (Excel col 4) (after 2 attempts)
if st.session_state.show_correct_prompt:
    with tabs[3]:
        st.subheader("📌 Sample Prompt 2")
        if expected_prompt_from_excel.strip():
            st.markdown(
                f"<div style='background:#f7f7f7;padding:15px;border-radius:8px; border:1px solid #ddd; white-space:pre-wrap;'>{expected_prompt_from_excel}</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("No Expected Prompt found in Excel Column 4 for this scenario.")

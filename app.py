import re
from datetime import datetime, timedelta
from io import BytesIO
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st
import google.genai as genai
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, HRFlowable)

# Use `streamlit run app.py` to run the app locally

# Configure Gemini API
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

AVAILABLE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]
DEFAULT_MODEL = "gemini-3.5-flash-lite"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# CSV analysis
# ---------------------------------------------------------------------------
@st.cache_data
def analyze_workout_csv(uploaded_file):
    """Analyze workout CSV and return summary, capability reference, error."""
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = ["exercise_title", "start_time"]
        if not all(col in df.columns for col in required_cols):
            return None, None, "CSV missing required columns: exercise_title, start_time"

        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df = df.dropna(subset=["start_time"])
        if df.empty:
            return None, None, "No valid workout data found"

        date_range = (df["start_time"].max() - df["start_time"].min()).days
        unique_dates = df["start_time"].dt.date.nunique()

        if date_range >= 7:
            freq_per_week = unique_dates / (date_range / 7)
        else:
            freq_per_week = unique_dates

        exercise_reference = []

        for exercise in sorted(df["exercise_title"].unique()):
            ex_data = df[df["exercise_title"] == exercise]

            # Normal sets only (exclude warmup, dropset, etc.) if column exists
            if "set_type" in ex_data.columns:
                ex_data_normal = ex_data[ex_data["set_type"] == "normal"]
            else:
                ex_data_normal = ex_data

            weights = (ex_data_normal["weight_kg"].dropna()
                       if "weight_kg" in ex_data_normal.columns else pd.Series(dtype=float))
            reps = (ex_data_normal["reps"].dropna()
                    if "reps" in ex_data_normal.columns else pd.Series(dtype=float))

            min_weight = max_weight = 0
            if not weights.empty:
                min_weight, max_weight = weights.min(), weights.max()

            max_reps = 0
            if not reps.empty:
                if max_weight > 0:
                    reps_at_max = ex_data_normal[ex_data_normal["weight_kg"] == max_weight]["reps"]
                    max_reps = reps_at_max.max() if not reps_at_max.empty else reps.max()
                else:
                    max_reps = reps.max()

            if max_weight > 0:
                exercise_reference.append(
                    f"{exercise}: User can perform {max_weight:.0f}kg "
                    f"(range: {min_weight:.0f}-{max_weight:.0f}kg) for {max_reps:.0f} reps"
                )
            elif max_reps > 0:
                exercise_reference.append(
                    f"{exercise}: User can perform {max_reps:.0f} reps (bodyweight)"
                )
            else:
                exercise_reference.append(f"{exercise}: User has experience with this exercise")

        summary = f"""**Your Workout History Summary:**
- Total workouts logged: {unique_dates}
- Date range: {df['start_time'].min().strftime('%d %b %Y')} to {df['start_time'].max().strftime('%d %b %Y')} ({date_range} days)
- Average frequency: {freq_per_week:.1f} workouts per week
- Total unique exercises: {df['exercise_title'].nunique()}

"""
        reference_text = "USER'S CURRENT CAPABILITY REFERENCE:\n" + "\n".join(exercise_reference)
        return summary, reference_text, None

    except Exception as e:
        return None, None, f"Error reading CSV: {str(e)}"


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def md_inline(text):
    """Escape XML, then convert markdown inline formatting to ReportLab tags."""
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def is_table_row(line):
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def is_separator_row(line):
    return re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line.strip()) is not None


def is_hr(line):
    return re.fullmatch(r"-{3,}|\*{3,}|_{3,}", line.strip()) is not None


def build_table(table_lines, cell_style, header_style, avail_width):
    rows = []
    for ln in table_lines:
        if is_separator_row(ln):
            continue
        rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    if not rows:
        return None

    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    data = []
    for i, r in enumerate(rows):
        style = header_style if i == 0 else cell_style
        data.append([Paragraph(md_inline(c), style) for c in r])

    weights = []
    for c in range(ncols):
        longest = max(len(r[c].replace("*", "")) for r in rows)
        weights.append(min(max(longest, 8), 40))
    total = sum(weights)
    col_widths = [avail_width * w / total for w in weights]

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4788")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0b7c3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build_pdf(workout_plan: str) -> bytes:
    """Convert the markdown workout plan to PDF bytes (with real tables)."""
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=letter,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    avail_width = doc.width

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Heading1"],
        fontSize=16, textColor="#1f4788", spaceAfter=12, leading=18)
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"],
        fontSize=13, textColor="#1f4788", spaceAfter=10, spaceBefore=10, leading=14)
    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"],
        fontSize=10, leading=13, spaceAfter=6)
    bullet_style = ParagraphStyle(
        "CustomBullet", parent=styles["Normal"],
        fontSize=10, leading=12, leftIndent=20, spaceAfter=4)
    cell_style = ParagraphStyle(
        "Cell", parent=styles["Normal"], fontSize=9, leading=11)
    header_cell_style = ParagraphStyle(
        "HeaderCell", parent=cell_style,
        textColor=colors.white, fontName="Helvetica-Bold")

    story = []
    lines = workout_plan.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if is_table_row(stripped):
            block = []
            while i < len(lines) and is_table_row(lines[i]):
                block.append(lines[i])
                i += 1
            tbl = build_table(block, cell_style, header_cell_style, avail_width)
            if tbl:
                story.append(tbl)
                story.append(Spacer(1, 0.15 * inch))
            continue

        if not stripped:
            story.append(Spacer(1, 0.1 * inch))
        elif is_hr(stripped):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey,
                                    spaceBefore=6, spaceAfter=6))
        elif stripped.startswith("###") or stripped.startswith("##"):
            story.append(Paragraph(md_inline(stripped.lstrip("#").strip()), heading_style))
        elif stripped.startswith("#"):
            story.append(Paragraph(md_inline(stripped.lstrip("#").strip()), title_style))
        elif stripped.startswith(("- ", "* ")):
            story.append(Paragraph(f"• {md_inline(stripped[2:].strip())}", bullet_style))
        else:
            story.append(Paragraph(md_inline(stripped), body_style))
        i += 1

    doc.build(story)
    return pdf_buffer.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏋️ AI Workout Planner by Daniel")
st.write("Get a personalised workout plan powered by Google Gemini AI (Free to Use!)")

st.write("### 📊 Upload Your Hevy Workout History (Optional)")
st.write("""##### To export your workout data from Hevy:
Go to Profile >
Settings (top right) >
Export & Import Data >
Export Data >
Export Workouts""")
uploaded_file = st.file_uploader(
    "Choose your workout_data.csv", type=["csv"],
    help="Upload your exported workout data to help create a better plan")

workout_history_summary = ""
exercise_reference = ""
if uploaded_file is not None:
    summary, reference, error = analyze_workout_csv(uploaded_file)
    if error:
        st.error(f"❌ {error}")
    else:
        st.success("✅ Workout history loaded successfully!")
        st.markdown(summary)
        workout_history_summary = summary
        exercise_reference = reference

st.write("---")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age:", 16, 100, 30)
    gender = st.selectbox("Gender:", ["Male", "Female"])
with col2:
    selected_days = st.multiselect(
        "Workout days:",
        WEEKDAYS,
        default=["Monday", "Wednesday", "Friday"],
        help="Pick the days you can train. All other days become rest/recovery days.",
    )
    fitness_level = st.selectbox("Fitness Level:", ["Beginner", "Intermediate", "Advanced"])

st.write("### Preferences")
col3, col4 = st.columns(2)
with col3:
    goal = st.multiselect(
        "Primary Goal(s):",
        ["Weight Loss", "Muscle Gain", "Strength", "Cardio Fitness"],
        default=["Strength"])
with col4:
    equipment = st.multiselect(
        "Available Equipment:",
        ["Full Gym Access", "Home Gym (Limited)", "Cardio Machines", "Bodyweight Only"],
        default=["Full Gym Access"])

workout_plan_length = st.slider("Workout Plan Length (weeks):", 1, 12, 4)

selected_model = st.selectbox(
    "🤖 AI Model:",
    AVAILABLE_MODELS,
    index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
    help="Lite models are faster and cheaper; the non-lite models are usually higher quality.",
)

specific_exercises = st.text_area(
    "💡 Specific Exercises (Optional):",
    placeholder="e.g., Barbell Bench Press, Barbell Deadlift, Barbell Back Squat",
    height=100)
unwanted_exercises = st.text_area(
    "💡 Unwanted Exercises (Optional):",
    placeholder="e.g., Barbell Bench Press, Barbell Deadlift, Barbell Back Squat",
    height=100)
specific_goals = st.text_area(
    "💡 Specific Goals (Optional):",
    placeholder="e.g., Improve posture, increase endurance, build bigger arms, recover from injury, etc.",
    height=100)
additional_notes = st.text_area(
    "📝 Additional Notes (Optional):",
    placeholder="e.g., Bad left shoulder, only 45 min per session, training before work, prefer supersets",
    height=100,
)

# ---------------------------------------------------------------------------
# Generate plan
# ---------------------------------------------------------------------------

if st.button("Generate My Workout Plan", type="primary"):
    if not selected_days:
        st.warning("Please select at least one workout day.")
        st.stop()

    selected_days = sorted(selected_days, key=WEEKDAYS.index)
    days_per_week = len(selected_days)
    days_str = ", ".join(selected_days)
    rest_days_str = ", ".join(d for d in WEEKDAYS if d not in selected_days) or "None"
    equipment_str = ", ".join(equipment) if equipment else "Bodyweight only"
    goals_str = specific_goals if specific_goals.strip() else "No specific goals mentioned"
    main_goals_str = ", ".join(goal) if goal else "General fitness"

    history_section = ""
    if workout_history_summary:
        history_section = f"""
IMPORTANT - User's Workout History:
{workout_history_summary}

{exercise_reference}

Use this data to:
- Recommend exercises to help the user progress towards their goals
- Suggest specific weights based on their current capability levels (as referenced above)
- For exercises they've done before, suggest progressive overload: Use 2.5kg increments.
- For new exercises, use similar exercises' capability as a baseline
- Adapt frequency and volume to match their typical routine
"""


    prompt = f"""Create a detailed {days_per_week}-day workout plan for a {age}-year-old {gender.lower()}
at {fitness_level.lower()} fitness level with suggested progressions over a {workout_plan_length}-week period.
Training days (schedule workouts ONLY on these days): {days_str}
Rest days (no training): {rest_days_str}
Main Goal(s): {main_goals_str}
Available equipment: {equipment_str}
Specific exercises to include: {specific_exercises if specific_exercises.strip() else "No specific exercises mentioned"}
Specific exercises to exclude: {unwanted_exercises if unwanted_exercises.strip() else "No specific exercises unwanted"}
Specific goals: {goals_str}
Additional notes: {additional_notes if additional_notes.strip() else "None"}
{history_section}

Please include:
- Daily workout schedule (assign a workout type to each training day: {days_str}; mark all other days as rest)
- Specific exercises for each day with sets, reps and rest times
- Rest days and recovery tips
- Proper form tips for key exercises
- Nutrition suggestions
- Progress tracking tips
- How the plan addresses their specific goals

Format it clearly with headers, bullet points, use metric units and make it actionable.
Use standard Markdown tables (header row, separator row, then data rows) for the weekly schedule and exercise lists. Do not use HTML."""

    with st.spinner("🤖 Creating your personalised workout plan..."):
        try:
            response = client.models.generate_content(model=selected_model, contents=prompt)
            plan = response.text
            st.session_state["workout_plan"] = plan
            st.session_state["pdf_bytes"] = build_pdf(plan)
        except Exception as e:
            st.session_state.pop("workout_plan", None)
            st.session_state.pop("pdf_bytes", None)
            st.error(f"Error generating plan: {e}")
            st.info("Make sure your API key is correctly set in .streamlit/secrets.toml")

# ---------------------------------------------------------------------------
# Display (persists across reruns, e.g. after clicking download)
# ---------------------------------------------------------------------------
if "workout_plan" in st.session_state:
    st.success("✅ Your workout plan is ready!")
    st.download_button(
        label="📥 Download Plan as PDF",
        data=st.session_state["pdf_bytes"],
        file_name="my_workout_plan.pdf",
        mime="application/pdf",
    )
    st.markdown(st.session_state["workout_plan"])
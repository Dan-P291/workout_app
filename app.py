import streamlit as st
import google.genai as genai
import pandas as pd
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from io import BytesIO
import re

# Use streamlit run app.py to run the app locally

# Configure Gemini API
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])


@st.cache_data
def analyze_workout_csv(uploaded_file):
    """Analyze workout CSV and return summary of exercises, frequencies, and weights"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_cols = ['exercise_title', 'start_time']
        if not all(col in df.columns for col in required_cols):
            return None, None, "CSV missing required columns: exercise_title, start_time"
        
        # Parse dates
        df['start_time'] = pd.to_datetime(df['start_time'], format='%d %b %Y, %H:%M', errors='coerce')
        df = df.dropna(subset=['start_time'])
        
        if df.empty:
            return None, None, "No valid workout data found"
        
        # Calculate metrics
        date_range = (df['start_time'].max() - df['start_time'].min()).days
        unique_dates = df['start_time'].dt.date.nunique()
        
        # Frequency calculation
        if date_range > 0:
            freq_per_week = (unique_dates / (date_range / 7)) if date_range >= 7 else unique_dates
        else:
            freq_per_week = unique_dates
        
        # Exercise analysis with capability reference
        exercise_summary = []
        exercise_reference = []
        
        for exercise in sorted(df['exercise_title'].unique()):
            ex_data = df[df['exercise_title'] == exercise]
            
            # Filter to normal sets only (exclude warmup, dropset, etc.)
            ex_data_normal = ex_data[ex_data['set_type'] == 'normal']
            
            # Weight range (if available)
            weights = ex_data_normal['weight_kg'].dropna()
            if not weights.empty:
                min_weight = weights.min()
                max_weight = weights.max()
                weight_info = f"{min_weight:.0f}-{max_weight:.0f}kg"
            else:
                weight_info = "bodyweight"
                min_weight = 0
                max_weight = 0
            
            # Reps at max weight (not overall max reps)
            reps = ex_data_normal['reps'].dropna()
            if not reps.empty:
                if max_weight > 0:
                    reps_at_max = ex_data_normal[ex_data_normal['weight_kg'] == max_weight]['reps']
                    max_reps = reps_at_max.max() if not reps_at_max.empty else reps.max()
                else:
                    max_reps = reps.max()
                rep_info = f"{reps.min():.0f}-{reps.max():.0f} reps"
            else:
                rep_info = "bodyweight"
                max_reps = 0
                
            
            exercise_summary.append(f"  • {exercise}: {weight_info}, {rep_info}")
            
            # Build detailed reference for Gemini - what weights they can currently handle
            if max_weight > 0:
                exercise_reference.append(f"{exercise}: User can perform {max_weight:.0f}kg (range: {min_weight:.0f}-{max_weight:.0f}kg) for {max_reps:.0f} reps")
            elif max_reps > 0:
                exercise_reference.append(f"{exercise}: User can perform {max_reps:.0f} reps (bodyweight)")
            else:
                exercise_reference.append(f"{exercise}: User has experience with this exercise")
        
        # Recent workouts (last 2 weeks)
        two_weeks_ago = datetime.now() - timedelta(days=14)
        recent_df = df[df['start_time'] >= two_weeks_ago]
        recent_exercises = recent_df['exercise_title'].unique()[:5] if len(recent_df) > 0 else []
        
        # Build summary text for display
        summary = f"""**Your Workout History Summary:**
- Total workouts logged: {unique_dates}
- Date range: {df['start_time'].min().strftime('%d %b %Y')} to {df['start_time'].max().strftime('%d %b %Y')} ({date_range} days)
- Average frequency: {freq_per_week:.1f} workouts per week
- Total unique exercises: {df['exercise_title'].nunique()}

**Recently Performed (last 2 weeks):**
{', '.join(recent_exercises) if len(recent_exercises) > 0 else "No recent workouts logged"}
"""
        
        # Build reference text for Gemini - used to suggest specific weights
        reference_text = "USER'S CURRENT CAPABILITY REFERENCE:\n" + "\n".join(exercise_reference)
        
        return summary, reference_text, None
    
    except Exception as e:
        return None, None, f"Error reading CSV: {str(e)}"

st.title("🏋️ AI Workout Planner by Daniel")
st.write("Get a personalized workout plan powered by Google Gemini AI (Free to Use!)")

# CSV upload section
st.write("### 📊 Upload Your Hevy Workout History (Optional)")
uploaded_file = st.file_uploader("Choose your workout_data.csv", type=['csv'], help="Upload your exported workout data to help create a better plan")

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

# User inputs
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age:", 16, 100, 30)
    gender = st.selectbox("Gender:", ["Male", "Female"])

with col2:
    days_per_week = st.slider("Workout days per week:", 1, 7, 3)
    fitness_level = st.selectbox("Fitness Level:", ["Beginner", "Intermediate", "Advanced"])

# Optional: Additional preferences
st.write("### Preferences")
col3, col4 = st.columns(2)
with col3:
    goal = st.multiselect("Primary Goal:", ["Weight Loss", "Muscle Gain", "Strength", "General Fitness"], default=["Strength"])
with col4:
    equipment = st.multiselect("Available Equipment:", ["Dumbbells", "Barbell", "Cardio Machine", "Bodyweight Only"], default=["Bodyweight Only"])

# Specific exercises text field
specific_exercises = st.text_area("💡 Specific Exercises (Optional):", placeholder="e.g., Barbell Bench Press, Barbell Deadlift, Barbell Back Squat", height=100)

# Unwanted exercises text field
unwanted_exercises = st.text_area("💡 Unwanted Exercises (Optional):", placeholder="e.g., Barbell Bench Press, Barbell Deadlift Barbell Back Squat", height=100)

# Specific goals text field
specific_goals = st.text_area("💡 Specific Goals (Optional):", placeholder="e.g., Improve posture, increase endurance, build bigger arms, recover from injury, etc.", height=100)

# Generate workout plan
if st.button("Generate My Workout Plan", type="primary"):
    equipment_str = ", ".join(equipment) if equipment else "Bodyweight only"
    goals_str = specific_goals if specific_goals.strip() else "No specific goals mentioned"
    
# Build prompt with workout history if available
    history_section = ""
    if workout_history_summary:
        history_section = f"""
IMPORTANT - User's Workout History:
{workout_history_summary}

{exercise_reference}

Use this data to:
- Recommend exercises the user is already familiar with when possible
- Suggest specific weights based on their current capability levels (as reference above)
- For exercises they've done before, suggest weight progression: If they did 40kg for 12 reps, suggest 42.5-45kg for the new plan
- For new exercises, use similar exercises' capability as a baseline
- Adapt frequency and volume to match their typical routine
"""
    
    prompt = f"""Create a detailed {days_per_week}-day workout plan for a {age}-year-old {gender.lower()} 
    at {fitness_level.lower()} fitness level.
    Main Goal(s): {goals_str}
    Available equipment: {equipment_str}
    Specific exercises to include: {specific_exercises if specific_exercises.strip() else "No specific exercises mentioned"}
    Specific exercises to exclude: {unwanted_exercises if unwanted_exercises.strip() else "No specific exercises unwanted"}
    Specific goals: {goals_str}{history_section}
    
    Please include:
    - Daily workout schedule (which days and what type of workout)
    - Specific exercises for each day with sets and reps
    - Rest days and recovery tips
    - Proper form tips for key exercises
    - Nutrition suggestions
    - Progress tracking tips
    - How the plan addresses their specific goals
    
    Format it clearly with headers, bullet points, use metric units and make it actionable."""
    
    with st.spinner("🤖 Creating your personalised workout plan..."):
        try:
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=prompt
            )
            workout_plan = response.text
            
            st.success("✅ Your workout plan is ready!")
            st.markdown(workout_plan)
            
            # Convert to PDF and download
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor='#1f4788',
                spaceAfter=12,
                leading=18
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=13,
                textColor='#1f4788',
                spaceAfter=10,
                spaceBefore=10,
                leading=14
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=10,
                leading=13,
                spaceAfter=6
            )
            
            bullet_style = ParagraphStyle(
                'CustomBullet',
                parent=styles['Normal'],
                fontSize=10,
                leading=12,
                leftIndent=20,
                spaceAfter=4
            )
            
            # Parse and format the text
            def convert_markdown_to_html(text):
                """Convert markdown formatting to HTML for reportlab"""
                # Convert **bold** to <b>bold</b>
                import re
                text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
                # Convert *italic* to <i>italic</i>
                text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
                return text
            
            story = []
            lines = workout_plan.split('\n')
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    story.append(Spacer(1, 0.1*inch))
                elif stripped.startswith('###'):
                    # Third-level heading
                    text = stripped.replace('###', '').strip()
                    text = convert_markdown_to_html(text)
                    story.append(Paragraph(f"<b>{text}</b>", heading_style))
                elif stripped.startswith('##'):
                    # Second-level heading
                    text = stripped.replace('##', '').strip()
                    text = convert_markdown_to_html(text)
                    story.append(Paragraph(f"<b>{text}</b>", heading_style))
                elif stripped.startswith('#'):
                    # First-level heading/title
                    text = stripped.replace('#', '').strip()
                    text = convert_markdown_to_html(text)
                    story.append(Paragraph(f"<b>{text}</b>", title_style))
                elif stripped.startswith('- '):
                    # Bullet point
                    text = stripped[2:].strip()
                    text = convert_markdown_to_html(text)
                    story.append(Paragraph(f"• {text}", bullet_style))
                elif stripped.startswith('* '):
                    # Bullet point (alternate format)
                    text = stripped[2:].strip()
                    text = convert_markdown_to_html(text)
                    story.append(Paragraph(f"• {text}", bullet_style))
                else:
                    # Regular paragraph
                    text = convert_markdown_to_html(stripped)
                    story.append(Paragraph(text, body_style))
            
            doc.build(story)
            pdf_bytes = pdf_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Plan as PDF",
                data=pdf_bytes,
                file_name="my_workout_plan.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating plan: {e}")
            st.info("Make sure your API key is correctly set in .streamlit/secrets.toml")
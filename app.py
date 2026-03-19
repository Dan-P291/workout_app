import streamlit as st
import google.genai as genai
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from io import BytesIO

# Use streamlit run app/app.py  to run the app locally

# Configure Gemini API
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

st.title("🏋️ AI Workout Planner by Daniel")
st.write("Get a personalized workout plan powered by Google Gemini AI (Free to Use!)")

# User inputs
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age:", 16, 80, 30)
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
specific_exercises = st.text_area("💡 Specific Exercises (Optional):", placeholder="e.g., Bench Press, Deadlifts, Squats", height=100)

# Specific goals text field
specific_goals = st.text_area("💡 Specific Goals (Optional):", placeholder="e.g., Improve posture, increase endurance, build bigger arms, recover from injury, etc.", height=100)

# Generate workout plan
if st.button("Generate My Workout Plan", type="primary"):
    equipment_str = ", ".join(equipment) if equipment else "Bodyweight only"
    goals_str = specific_goals if specific_goals.strip() else "No specific goals mentioned"
    
    prompt = f"""Create a detailed {days_per_week}-day workout plan for a {age}-year-old {gender.lower()} 
    at {fitness_level.lower()} fitness level.
    Main Goal(s): {goals_str}
    Available equipment: {equipment_str}
    Specific exercises to include: {specific_exercises if specific_exercises.strip() else "No specific exercises mentioned"}
    Specific goals: {goals_str}
    
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
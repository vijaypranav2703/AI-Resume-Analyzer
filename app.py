import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# PDF Text Extraction
# ----------------------------------
def extract_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# ----------------------------------
# Skills Database
# ----------------------------------
skills_database = [
    "python",
    "java",
    "sql",
    "tensorflow",
    "machine learning",
    "deep learning",
    "aws",
    "html",
    "css",
    "javascript",
    "react",
    "pandas",
    "numpy",
    "git",
    "github",
    "mysql",
    "docker",
    "power bi",
    "tableau",
    "excel",
    "streamlit",
    "data analysis",
    "cloud computing",
    "problem solving",
    "data visualization"
]

# ----------------------------------
# App Title
# ----------------------------------
st.title("🚀 AI Resume Analyzer")

st.caption(
    "Upload your resume and compare it against a job description to identify matching skills, missing skills, and career recommendations."
)

# ----------------------------------
# Sample Job Description
# ----------------------------------
sample_jd = """
We are looking for a motivated Artificial Intelligence & Data Science student with knowledge of machine learning, data analytics, cloud computing, and software development.

Responsibilities:
- Build and train machine learning models
- Analyze and visualize data
- Develop AI-powered applications
- Conduct research and provide factually accurate reports
- Collaborate with cross-functional teams
- Maintain documentation and process reports

Required Skills:
Python
SQL
Machine Learning
Deep Learning
TensorFlow
Data Analysis
Problem Solving
Analytical Thinking
Attention to Detail
Communication Skills
Git
GitHub

Preferred Skills:
AWS
Pandas
NumPy
Excel
Power BI
Docker
Cloud Computing
Data Visualization
"""

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if st.button("📋 Load Sample Job Description"):
    st.session_state.job_description = sample_jd

job_description = st.text_area(
    "Paste Job Description Here",
    value=st.session_state.job_description,
    height=150
)

# ----------------------------------
# Resume Upload
# ----------------------------------
uploaded_file = st.file_uploader(
    "Upload your Resume",
    type=["pdf"]
)

# ----------------------------------
# Main Application
# ----------------------------------
if uploaded_file:

    st.success("Resume uploaded successfully!")

    st.write("📄 File Name:", uploaded_file.name)

    st.write("📦 File Size:", uploaded_file.size, "bytes")

    resume_text = extract_text(uploaded_file)

    # ----------------------------------
    # Resume Content
    # ----------------------------------
    with st.expander("📄 View Resume Content"):

        st.text(resume_text[:2000])

    # ----------------------------------
    # Skills Detection
    # ----------------------------------
    found_skills = []

    for skill in skills_database:

        if skill.lower() in resume_text.lower():

            found_skills.append(skill)

    st.subheader("✅ Skills Detected")

    for skill in found_skills:

        st.write("✔️", skill)

    # ----------------------------------
    # Resume Match Score
    # ----------------------------------
    if job_description:

        tfidf = TfidfVectorizer()

        matrix = tfidf.fit_transform(
            [resume_text, job_description]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )

        match_score = round(
            score[0][0] * 100,
            2
        )

        # ----------------------------------
        # Job Skills
        # ----------------------------------
        job_skills = []

        for skill in skills_database:

            if skill.lower() in job_description.lower():

                job_skills.append(skill)

        # ----------------------------------
        # Missing Skills
        # ----------------------------------
        missing_skills = list(
            set(job_skills) - set(found_skills)
        )

        # ----------------------------------
        # Skill Match Score
        # ----------------------------------
        if len(job_skills) > 0:

            matched_skills = len(
                set(job_skills).intersection(
                    set(found_skills)
                )
            )

            skill_match_score = round(
                (matched_skills / len(job_skills)) * 100,
                2
            )

        else:

            skill_match_score = 0

        # ----------------------------------
        # Scores
        # ----------------------------------
        st.subheader("📊 Resume Match Scores")

col1, col2, col3 = st.columns(3)

col1.metric("Skills Found", len(found_skills))
col2.metric("Missing Skills", len(missing_skills))
col3.metric("Skill Match", f"{skill_match_score}%")

st.progress(int(skill_match_score))

st.success(
    f"📄 Text Similarity Score: {match_score}%"
)

st.info(
    f"🎯 Skill Match Score: {skill_match_score}%"
)

        # ----------------------------------
        # Resume Rating
        # ----------------------------------
        if skill_match_score >= 80:

            st.success(
                "🌟 Excellent Resume Match"
            )

        elif skill_match_score >= 60:

            st.info(
                "👍 Good Resume Match"
            )

        elif skill_match_score >= 40:

            st.warning(
                "⚠️ Average Resume Match"
            )

        else:

            st.error(
                "❌ Low Resume Match"
            )

        # ----------------------------------
        # Missing Skills
        # ----------------------------------
        st.subheader("❌ Missing Skills")

        if missing_skills:

            for skill in missing_skills:

                st.write("❌", skill)

            # ----------------------------------
            # Suggestions
            # ----------------------------------
            st.subheader("📌 Suggestions")

            for skill in missing_skills:

                st.write(f"📚 Learn {skill}")

        else:

            st.success(
                "No missing skills detected!"
            )

            st.subheader("📌 Suggestions")

            st.success(
                "Your resume matches all detected skills."
            )

        # ----------------------------------
        # Career Recommendation
        # ----------------------------------
        st.subheader("🎯 Career Recommendation")

        if skill_match_score >= 70:

            st.success(
                "Your profile is suitable for AI/ML Engineer, Data Scientist and Data Analyst roles."
            )

        elif skill_match_score >= 50:

            st.info(
                "Your profile is suitable for Entry-Level AI, Data Analyst and Graduate Trainee roles."
            )

        else:

            st.warning(
                "Consider improving your skills and projects before applying for advanced AI roles."
            )

        # ----------------------------------
        # Summary Dashboard
        # ----------------------------------
        st.subheader("📈 Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Text Score",
            f"{match_score}%"
        )

        col2.metric(
            "Skill Score",
            f"{skill_match_score}%"
        )

        col3.metric(
            "Skills Found",
            len(found_skills)
        )

        col4.metric(
            "Missing Skills",
            len(missing_skills)
        )
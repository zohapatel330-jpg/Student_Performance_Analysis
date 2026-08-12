
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():

    df = pd.read_csv("StudentsPerformance.csv")

    # Remove empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicates
    df = df.drop_duplicates()

    # Calculate average score
    df["average_score"] = (
        df["math score"] +
        df["reading score"] +
        df["writing score"]
    ) / 3

    # Performance category
    def performance_category(score):
        if score >= 75:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Average"
        else:
            return "Needs Improvement"

    df["performance"] = df["average_score"].apply(
        performance_category
    )

    return df


# Load model
@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")


df = load_data()
model = load_model()


# -----------------------------------------
# TITLE
# -----------------------------------------

st.title("🎓 Student Performance Analysis")

st.write(
    "An interactive dashboard for analyzing "
    "student academic performance."
)

st.divider()


# -----------------------------------------
# KPI SECTION
# -----------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Students",
        len(df)
    )

with col2:
    st.metric(
        "Average Math Score",
        round(df["math score"].mean(), 2)
    )

with col3:
    st.metric(
        "Average Reading Score",
        round(df["reading score"].mean(), 2)
    )

with col4:
    st.metric(
        "Average Writing Score",
        round(df["writing score"].mean(), 2)
    )


st.divider()


# -----------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------

st.sidebar.header("🔎 Student Filters")

gender_filter = st.sidebar.multiselect(
    "Gender",
    df["gender"].unique(),
    default=df["gender"].unique()
)

race_filter = st.sidebar.multiselect(
    "Race/Ethnicity",
    df["race/ethnicity"].unique(),
    default=df["race/ethnicity"].unique()
)

prep_filter = st.sidebar.multiselect(
    "Test Preparation",
    df["test preparation course"].unique(),
    default=df["test preparation course"].unique()
)


filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["race/ethnicity"].isin(race_filter)) &
    (df["test preparation course"].isin(prep_filter))
]


# -----------------------------------------
# FILTERED DATA
# -----------------------------------------

st.header("📊 Student Data")

st.write(
    "Number of students after filtering:",
    len(filtered_df)
)

st.dataframe(
    filtered_df,
    use_container_width=True
)


# -----------------------------------------
# SUBJECT PERFORMANCE
# -----------------------------------------

st.header("📚 Subject-wise Performance")

subject_average = filtered_df[
    [
        "math score",
        "reading score",
        "writing score"
    ]
].mean()

fig, ax = plt.subplots(figsize=(8, 5))

subject_average.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Average Score by Subject")
ax.set_xlabel("Subject")
ax.set_ylabel("Average Score")

st.pyplot(fig)


# -----------------------------------------
# PERFORMANCE CATEGORY
# -----------------------------------------

st.header("🏆 Performance Categories")

performance_count = filtered_df[
    "performance"
].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))

performance_count.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Student Performance Categories")
ax.set_xlabel("Performance")
ax.set_ylabel("Number of Students")

st.pyplot(fig)


# -----------------------------------------
# GENDER ANALYSIS
# -----------------------------------------

st.header("👥 Performance by Gender")

gender_data = filtered_df.groupby(
    "gender"
)[
    [
        "math score",
        "reading score",
        "writing score"
    ]
].mean()

st.dataframe(
    gender_data.round(2),
    use_container_width=True
)


# -----------------------------------------
# TEST PREPARATION
# -----------------------------------------

st.header("📝 Test Preparation Analysis")

prep_data = filtered_df.groupby(
    "test preparation course"
)["average_score"].mean()

fig, ax = plt.subplots(figsize=(8, 5))

prep_data.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Average Score by Test Preparation"
)

ax.set_xlabel("Test Preparation")
ax.set_ylabel("Average Score")

st.pyplot(fig)


# -----------------------------------------
# CORRELATION
# -----------------------------------------

st.header("🔥 Score Correlation")

numeric_columns = [
    "math score",
    "reading score",
    "writing score",
    "average_score"
]

correlation = filtered_df[
    numeric_columns
].corr()

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    ax=ax
)

ax.set_title(
    "Correlation Between Student Scores"
)

st.pyplot(fig)


# -----------------------------------------
# PREDICTION
# -----------------------------------------

st.divider()

st.header("🤖 Student Writing Score Prediction")

st.write(
    "Enter student details to predict the writing score."
)

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        df["gender"].unique()
    )

    race = st.selectbox(
        "Race/Ethnicity",
        df["race/ethnicity"].unique()
    )

    parental_education = st.selectbox(
        "Parental Level of Education",
        df["parental level of education"].unique()
    )


with col2:

    preparation = st.selectbox(
        "Test Preparation Course",
        df["test preparation course"].unique()
    )

    math_score = st.slider(
        "Math Score",
        0,
        100,
        50
    )

    reading_score = st.slider(
        "Reading Score",
        0,
        100,
        50
    )


if st.button("🔮 Predict Writing Score"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [
            parental_education
        ],
        "test preparation course": [
            preparation
        ],
        "math score": [math_score],
        "reading score": [reading_score]
    })

    prediction = model.predict(
        input_data
    )[0]

    prediction = max(
        0,
        min(100, prediction)
    )

    st.success(
        f"Predicted Writing Score: {prediction:.2f}"
    )

    if prediction >= 75:
        st.info("Performance Level: Excellent")

    elif prediction >= 60:
        st.info("Performance Level: Good")

    elif prediction >= 40:
        st.info("Performance Level: Average")

    else:
        st.warning(
            "Performance Level: Needs Improvement"
        )


st.divider()

st.caption(
    "Student Performance Analysis | Data Science Project"
)

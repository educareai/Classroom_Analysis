import openai
import pandas as pd
import re
import os
import json
from collections import Counter
import pandas as pd
# from openai import OpenAI
# from dotenv import load_dotenv

# Load API key from .env file (store your API key securely)
openai.api_key = "Your-API-key"
# Load API key securely (replace with environment variable in production)
# openai.api_key = "your-api-key-here"

# Load transcript file with UTF-16 LE encoding
file_path = "D:/EducareAI/second_task/final_file.txt"
with open(file_path, "r", encoding="utf-16-le") as file:
    transcript = file.readlines()

# Function to extract timestamp, student name, and dialogue
def extract_data(transcript):
    data = []
    pattern = r"\[(\d{1,2}:\d{2} [APM]{2})\]\s*([\w\s]+):\s*(.+)"  # Adjusted regex

    for line in transcript:
        match = re.match(pattern, line.strip())
        if match:
            timestamp, student, text = match.groups()
            data.append({"timestamp": timestamp, "student": student, "text": text})

    df = pd.DataFrame(data)
    print("Extracted Data:\n", df.head())  # Debugging
    return df

# Function to split text into chunks for LLM processing
def chunk_text(text, max_tokens=400):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Analyze engagement using OpenAI API
def analyze_engagement(text):
    text_chunks = chunk_text(text)

    aggregated_analysis = {
        "engagement_score": 0,
        "participation_level": "",
        "engagement_strengths": "",
        "engagement_improvements": "",
        "problem_solving_score": 0,
        "critical_thinking_strengths": ""
    }
    
    for chunk in text_chunks:
        prompt = f"""
        Analyze the following classroom participation response and provide:
        1. Engagement Score (0-100)
        2. Participation Level (Moderate, Good, High)
        3. Engagement Strengths
        4. Engagement Improvements
        5. Problem-Solving Score (0-100)
        6. Critical Thinking Strengths
        
        Response: {chunk}
        
        Provide the response in JSON format with keys:
        engagement_score, participation_level, engagement_strengths, engagement_improvements, problem_solving_score, critical_thinking_strengths.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": "You are an expert education evaluator."},
                          {"role": "user", "content": prompt}],
                temperature=0.5
            )
            # print(response)
            response_content = response["choices"][0]["message"]["content"]
            clean_json = response_content.strip("```json").strip("```").strip()
            analysis = json.loads(clean_json)  # Parse JSON
            aggregated_analysis["engagement_score"] += analysis["engagement_score"]
            aggregated_analysis["problem_solving_score"] += analysis["problem_solving_score"]
            aggregated_analysis["participation_level"] = analysis["participation_level"]
            aggregated_analysis["engagement_strengths"] += f" {analysis['engagement_strengths']}"
            aggregated_analysis["engagement_improvements"] += f" {analysis['engagement_improvements']}"
            aggregated_analysis["critical_thinking_strengths"] += f" {analysis['critical_thinking_strengths']}"

        except Exception as e:
            print("Error with OpenAI API:", e)
            return None

    # Normalize scores if multiple chunks
    num_chunks = len(text_chunks)
    if num_chunks > 1:
        aggregated_analysis["engagement_score"] = round(aggregated_analysis["engagement_score"] / num_chunks)
        aggregated_analysis["problem_solving_score"] = round(aggregated_analysis["problem_solving_score"] / num_chunks)

    return aggregated_analysis

def count_student_mentions(df):
    student_mentions = Counter()

    for _, row in df.iterrows():
        student_mentions[row["student"]] += 1

    return dict(student_mentions)

def count_participation(df):
    return df["student"].value_counts().to_dict()

def talk_time_ratio(df):
    teacher_name = "Emma"  # Change to actual teacher’s name

    teacher_words = df[df["student"] == teacher_name]["text"].apply(lambda x: len(x.split())).sum()
    student_words = df[df["student"] != teacher_name]["text"].apply(lambda x: len(x.split())).sum()

    return {"Teacher": teacher_words, "Students": student_words, "Ratio": round(teacher_words / max(1, student_words), 2)}

# def attention_gaps(df):
#     df = df.copy() 
#     df["Timestamp"] = pd.to_datetime(df["timestamp"], format="%I:%M %p")
#     df["Time Difference"] = df["Timestamp"].diff().dt.total_seconds() / 60
#     return df["Time Difference"].max()  # Return longest gap in minutes

confusion_keywords = ["I don’t get it", "I'm confused", "Can you repeat?", "Not sure", "What does that mean?", "Huh?"]

def detect_confusion(df):
    return df[df["text"].str.contains("|".join(confusion_keywords), case=False)].to_dict(orient="records")

def speaking_distribution(df):
    return df.groupby("student")["text"].apply(lambda x: sum(len(t.split()) for t in x)).to_dict()

topics = ["geometry", "triangle", "hypotenuse", "angle", "proof", "congruent", "theorem"]

def topic_coverage(df):
    coverage = {topic: df["text"].str.contains(topic, case=False).sum() for topic in topics}
    return coverage

def generate_highlights(df):
    full_text = " ".join(df["text"].tolist())

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a note-taking assistant."},
            {"role": "user", "content": f"Summarize the key moments in this classroom discussion: {full_text}"}
        ],
        temperature=0.5
    )

    return response["choices"][0]["message"]["content"]

def participation_quality(df):
    df["sentence_length"] = df["text"].apply(lambda x: len(x.split()))
    return df.groupby("student")["sentence_length"].mean().to_dict()

# def check_comprehension(df):
#     for _, row in df.iterrows():
#         prompt = f"Evaluate whether this student response is correct or incorrect: {row['text']}."
        
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a classroom evaluator."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.5
#         )
        
#         row["comprehension_feedback"] = response["choices"][0]["message"]["content"]
    
#     return df

def effectiveness_score(talk_ratio, participation_count):
    return round(100 - (talk_ratio["Ratio"] * 50) + (len(participation_count) * 10), 2)


# Process transcript
df = extract_data(transcript)
df_sample = df.head(50)
print(participation_quality(df_sample))
# Apply NLP analysis
analysis_results = []
df_sample = df.head(30)
for _, row in df_sample.iterrows():
    analysis = analyze_engagement(row["text"])
    if analysis:
        analysis_results.append({
            "Timestamp": row["timestamp"],
            "Student": row["student"],
            "Engagement Score": analysis["engagement_score"],
            "Participation Level": analysis["participation_level"],
            "Engagement Strengths": analysis["engagement_strengths"].strip(),
            "Engagement Improvements": analysis["engagement_improvements"].strip(),
            "Problem-Solving Score": analysis["problem_solving_score"],
            "Critical Thinking Strengths": analysis["critical_thinking_strengths"].strip(),
        })
# print(df_sample)
# for _, row in df_sample.iterrows():
#     analysis = analyze_engagement(row["text"])

# Convert to DataFrame
df_analysis = pd.DataFrame(analysis_results)
# print(df)
# Standardized Excel Output
output_path = "engagement_analysis.xlsx"
with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    df_analysis.to_excel(writer, index=False, sheet_name="Analysis")

print(f"Analysis saved to {output_path}")

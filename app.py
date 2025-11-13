from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load GPT-2 model for text generation using Hugging Face (for feedback)
generator = pipeline('text-generation', model='gpt2')

# Load Sentence-BERT model for comparing answers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate AI model answer using GPT-2
def generate_ai_model_answer(question):
    result = generator(question, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

# Function to compare the model answer with the student answer
def compare_answers(model_answer, student_answer):
    emb_ref = model.encode(model_answer, convert_to_tensor=True)
    emb_stu = model.encode(student_answer, convert_to_tensor=True)
    similarity = util.cos_sim(emb_ref, emb_stu).item()
    return similarity

# Function to generate feedback based on the similarity score
def generate_feedback(model_answer, student_answer, similarity_score):
    feedback = ""

    # Fine-tune or classify based on the similarity
    if similarity_score > 0.8:
        feedback = "Excellent answer, covers all key points."
    elif similarity_score > 0.6:
        feedback = "Good answer but missing some key details."
    else:
        feedback = "The answer is too basic. Consider adding more details."

    # Add missing concept detection for further improvement
    missing_details = []
    
    # Example check: check if the answer lacks key details (e.g., I/O control for OS question)
    if "I/O control" not in student_answer.lower() and "I/O control" in model_answer.lower():
        missing_details.append("I/O control")

    if missing_details:
        feedback += f" You missed mentioning: {', '.join(missing_details)}."

    return feedback

# Route to handle form and generate answers
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form["question"]
        student_answer = request.form["student_answer"]
        input_type = request.form["input_type"]

        # Generate model answer based on input type
        if input_type == "manual":
            model_answer = request.form["model_answer"]
        else:  # AI-generated answer using GPT-2
            model_answer = generate_ai_model_answer(question)

        # Compare the generated model answer with the student's answer
        similarity_score = compare_answers(model_answer, student_answer)

        # Generate feedback based on similarity score
        feedback = generate_feedback(model_answer, student_answer, similarity_score)

        # Calculate the score based on similarity
        score = round(similarity_score * 5, 2)

        # Determine feedback class based on score for styling in HTML
        if score >= 4:
            feedback_class = "excellent"
        elif score >= 3:
            feedback_class = "good"
        else:
            feedback_class = "basic"

        return render_template("index.html", score=score, feedback=feedback, model_answer=model_answer, feedback_class=feedback_class)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

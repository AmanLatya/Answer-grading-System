from sentence_transformers import SentenceTransformer, util

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# User inputs
question = input("Enter the question: ")
model_answer = input("Enter the model answer: ")
student_answer = input("Enter the student answer: ")

# Generate embeddings for both answers
emb_ref = model.encode(model_answer, convert_to_tensor=True)
emb_stu = model.encode(student_answer, convert_to_tensor=True)

# Compute cosine similarity
similarity = util.cos_sim(emb_ref, emb_stu).item()

# Display the similarity score
print(f"Cosine Similarity: {similarity}")

# Map similarity to score (0-5)
score = round(similarity * 5, 2)
print(f"Predicted Score: {score}")

# Generate feedback
if similarity > 0.8:
    feedback = "Excellent answer, covers all key points."
elif similarity > 0.6:
    feedback = "Good answer but missing some key details."
else:
    feedback = "The answer is too basic. Consider adding more details."

print(f"Feedback: {feedback}")

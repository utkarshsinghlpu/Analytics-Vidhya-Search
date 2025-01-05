import requests
from bs4 import BeautifulSoup
import pandas as pd
import gradio as gr
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Scrape the free courses from Analytics Vidhya
url = "https://courses.analyticsvidhya.com/pages/all-free-courses"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

courses = []

# Extracting course title, image, and course link
for course_card in soup.find_all('header', class_='course-card__img-container'):
    img_tag = course_card.find('img', class_='course-card__img')
    
    if img_tag:
        title = img_tag.get('alt', 'Untitled Course')
        image_url = img_tag.get('src', '')
        
        link_tag = course_card.find_previous('a')
        if link_tag:
            course_link = link_tag.get('href', '')
            if not course_link.startswith('http'):
                course_link = 'https://courses.analyticsvidhya.com' + course_link

            courses.append({
                'title': title,
                'image_url': image_url,
                'course_link': course_link
            })

# Step 2: Create DataFrame
df = pd.DataFrame(courses)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings using BERT
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Create embeddings for course titles
df['embedding'] = df['title'].apply(lambda x: get_bert_embedding(x))

# Function to perform search using BERT-based similarity
def search_courses(query):
    query_embedding = get_bert_embedding(query)
    course_embeddings = np.vstack(df['embedding'].values)
    
    # Compute cosine similarity between query embedding and course embeddings
    similarities = cosine_similarity(query_embedding, course_embeddings).flatten()
    
    # Add the similarity scores to the DataFrame
    df['score'] = similarities
    
    # Sort by similarity score in descending order and return top results
    top_results = df.sort_values(by='score', ascending=False).head(10)
    return top_results[['title', 'image_url', 'course_link', 'score']].to_dict(orient='records')

# Function to simulate autocomplete by updating search results live
def autocomplete(query):
    matching_courses = df[df['title'].str.contains(query, case=False, na=False)]
    return matching_courses['title'].tolist()[:3]  # Show top 3 matching course titles

def gradio_search(query):
    result_list = search_courses(query)
    
    if result_list:
        html_output = '<div class="results-container">'
        for item in result_list:
            course_title = item['title']
            course_image = item['image_url']
            course_link = item['course_link']
            relevance_score = round(item['score'] * 100, 2)
            
            html_output += f'''
            <div class="course-card">
                <img src="{course_image}" alt="{course_title}" class="course-image"/>
                <div class="course-info">
                    <h3>{course_title}</h3>
                    <p>Relevance: {relevance_score}%</p>
                    <a href="{course_link}" target="_blank" class="course-link">View Course</a>
                </div>
            </div>'''
        html_output += '</div>'
        return html_output
    else:
        return '<p class="no-results">No results found.</p>'

# Custom CSS for the Gradio interface

custom_css = """
body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(120deg, #6a11cb, #2575fc); /* Blue gradient background */
    margin: 0;
    padding: 0;
    color: #ffffff;
}
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
.results-container {
    display: flex;
    flex-wrap: wrap;
    
    justify-content: center;
}
.course-card {
    background: linear-gradient(120deg, #e3ffe7, #d9e7ff); /* Greenish tint */
    border-radius: 15px;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
    transition: transform 0.4s, box-shadow 0.4s;
    overflow: hidden;
    width: 45%;
    max-width: 400px;
    margin: 10px;
    position: relative;
}
.course-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}
.course-image {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-bottom: 3px solid #6a11cb; /* Blue accent */
}
.course-info {
    padding: 20px;
    background: linear-gradient(120deg, #ffffff, #e6f9e7); /* Light green and white */
}
.course-info h3 {
    margin: 0;
    font-size: 22px;
    color: #333;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}
.course-info p {
    margin: 10px 0;
    color: #555;
    font-size: 15px;
    line-height: 1.5;
}
.course-link {
    display: inline-block;
    background: linear-gradient(120deg, #34d1c1, #11998e); /* Green gradient */
    color: white;
    padding: 12px 18px;
    text-decoration: none;
    border-radius: 8px;
    font-size: 15px;
    font-weight: bold;
    transition: background 0.3s, transform 0.3s;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}
.course-link:hover {
    background: linear-gradient(120deg, #11998e, #34d1c1); /* Reverse gradient */
    transform: scale(1.1);
}
.no-results {
    text-align: center;
    color: #ffffff;
    font-style: italic;
    margin-top: 20px;
    font-size: 18px;
    background: linear-gradient(120deg, #ff7eb3, #ff758c); /* Pinkish gradient */
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}
"""



# Gradio interface
iface = gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(label="Enter your search query", placeholder="e.g., machine learning, data science, python"),
    outputs=gr.HTML(label="Search Results"),
    title="Analytics Vidhya Smart Course Search",
    description="Find the most relevant courses from Analytics Vidhya based on your query.",
    theme="default",
    css=custom_css,
    examples=[
        ["machine learning for beginners"],
        ["advanced data visualization techniques"],
        ["python programming basics"], 
        ["Business Analytics"]
    ],
    allow_flagging=None  # Disable the flagging feature
)


if __name__ == "__main__":
    iface.launch()

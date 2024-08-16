Portfolio Web Application with Integrated RAG Chatbot
Team members :
Anmol Valecha (NUID - 002813410)
Kaushikee Bhawsar (NUID - 002704590)
Youtube Link -
Project Documents - Project proposal, Final Project Presentation, and Final Report are available in GitHub repo
Website Link -
Kaushikee's portfolio
Anmol's portfolio
Project Overview
This project is a comprehensive portfolio website that showcases personal skills, projects, and achievements. It is designed with a modern, responsive frontend and a robust backend, integrated with an AI-powered chatbot using Retrieval-Augmented Generation (RAG). The chatbot allows users to submit natural language queries about the portfolio owner's profile, providing an interactive and engaging experience.

Features
Responsive portfolio website built with React, HTML, and CSS
AI-powered RAG chatbot integrated using Streamlit, Hugging Face transformers, and Pinecone/Chroma for vector storage
Easy-to-follow setup instructions for cloning and running the project locally
Project Architecture
alt text

Prerequisites
Before you begin, ensure you have the following installed on your system:

Node.js (v14.x or later)
npm
Python (v3.7 or later)
pip (Python package installer)
Setup Instructions
Clone the Repository
<<<<<<< HEAD
git clone https://github.com/valecha-a/PORTFOLIO_BOT.git
=======
git clone https://github.com/kaushikeebhawsar99/FINAL-PROJECT-PORTFOLIO-RAG-CHATBOT.git
>>>>>>> cb81731c59385802765971d26eb2b889edd49fa8
cd final-project-portfolio-rag-chatbot
Setup and Run the React App
Install Dependencies: Navigate to the frontend directory and install the necessary dependencies:
cd portfolio_react_app
npm install
Start the React App: Once the dependencies are installed, you can start the React app:
npm start
This command will run the app in development mode. Open http://localhost:3000 to view it in the browser.

Setup and Run the Chatbot
Install Python Dependencies: Navigate to the chatbot directory and install the required Python packages:
cd ../portfolio_kbot
pip install -r requirements.txt
Create Vector embeddings: Once the dependencies are installed, generate vector embeddings by running the following command:
python vector_embeddings.py
Now run the Chatbot app: Once the embeddings are generated and stored in the data directory, run the following command
streamlit run ResumeBot.py
The chatbot will be accessible at http://localhost:8501. You can interact with it by submitting natural language queries.

Project Structure
frontend/: Contains the React application for the portfolio website.
chatbot/: Contains the Streamlit application and all necessary files for the RAG chatbot.
License
<<<<<<< HEAD
This project is licensed under the MIT License. See the LICENSE file for more details.
=======
This project is licensed under the MIT License. See the LICENSE file for more details.

About
No description, website, or topics provided.
Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 1 watching
Forks
 0 forks
Report repository
Releases
No releases published
Packages
No packages published
Contributors
2
@kaushikeebhawsar99
kaushikeebhawsar99
@Kaushikee-Bhawsar
Kaushikee-Bhawsar
Languages
JavaScript
86.2%
 
CSS
8.4%
 
Python
4.2%
 
HTML
1.2%
Footer
© 2024 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
>>>>>>> cb81731c59385802765971d26eb2b889edd49fa8

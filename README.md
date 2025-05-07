# chatDB_public

ChatDB is a tool that lets you talk to databases using plain English. Instead of writing complicated SQL or MongoDB queries, you can just type what you want to know, and ChatDB will figure out the right command for you. It works with both MySQL and MongoDB, and it uses OpenAI’s GPT-4o model to turn your questions into real database queries.

The whole system runs using Docker, so you don’t have to worry about setting everything up yourself. Just follow a few setup steps, and Docker will handle all the environment stuff for you.

## Prerequisites

Before you begin, ensure you have the following installed and configured on your system:

* **Docker and Docker Compose**: This project relies on Docker to manage and run its services. Please ensure you have a working installation of both Docker Engine and Docker Compose.


## Setup and Usage
1. Clone the project:
   ```bash
   git clone https://github.com/Kyle7994/chatDB_public.git 
   cd chatDB_public
   ```
2. Add your OpenAI API key:

   Edit `.env` file and set:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Build Docker images from scratch (no cached layers):
   ```bash
   docker compose build --no-cache
   ```
4. Start the entire stack in detached mode:
   ```bash
   docker compose up -d
   ```
5. Make the helper scripts executable:
   ```bash
   chmod +x runSqlQuery.sh
   chmod +x runMongoQuery.sh
   ```
6. Example SQL query (MySQL): find customer names for product ID 5:
   ```bash
   ./runSqlQuery.sh "What are the names of customers who ordered the product with ID 5?"
   ```
7. Example MongoDB query: list students enrolled in BIO253:
   ```bash
   ./runMongoQuery.sh "Show the full student details (name, email) for everyone enrolled in course ID 'BIO253'."
   ```
8. Shut everything down and remove volumes to clear data:
   ```bash
   docker compose down -v
   ```

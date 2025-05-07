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

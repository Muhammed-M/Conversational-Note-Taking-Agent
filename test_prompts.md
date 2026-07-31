# Test Prompts & Real-World Note Scenarios

Use these test prompts in your interactive terminal session (`python main.py`) to test all agent features (saving, hybrid search, updating, deleting, disambiguation, and human-in-the-loop confirmation).

---

## 📥 1. Saving Notes (Rich & Detailed Real-World Scenarios)

Copy and paste these long, multi-paragraph messages directly into the CLI session to verify that the LLM extracts clean titles, comprehensive body content, and relevant tags automatically.

### Note 1: Architecture & Tech Stack Decision
```text
Save a note about our Backend Architecture Refactoring Decisions. We had a 2-hour technical meeting today and decided to migrate our main REST API services from Django to FastAPI for better async performance and automatic OpenAPI documentation. We are keeping PostgreSQL as our relational store for user accounts, billing, and transactions. For real-time updates and caching, we chose Redis. For high-volume event stream processing, Apache Kafka will be deployed. Target release for Phase 1 is end of Q3. Tag this note under architecture, backend, python, and meetings.
```

### Note 2: Weekly Team Standup & Sprint Planning
```text
Remember this for me: Team Standup and Sprint 14 Planning. During today's sprint meeting, Sarah reported that the payment gateway integration with Stripe is 80% finished, but blocked on webhook testing. Alex completed the user authentication module with OAuth2 and JWT tokens. David is working on dynamic UI layouts using React and Tailwind CSS. Action items: Sarah will sync with the QA team on Tuesday to unblock Stripe testing; David needs to resolve the mobile viewport layout bug by Thursday. Next sprint review is scheduled for August 15. Tag as standup, sprint, work, and frontend.
```

### Note 3: Personal Study & Generative AI Research
```text
Save note: GenAI Learning & RAG Optimization Notes. Read several papers today on Retrieval-Augmented Generation (RAG) performance optimization. Key takeaways: 1. Dense retrieval using vector databases like Qdrant or Pinecone works best when combined with sparse keyword search (BM25 or SQL LIKE). 2. Reranking using cross-encoders significantly improves Precision@K. 3. Chunk sizes between 256 and 512 tokens with 10% overlap yield optimal recall for technical documentation. Next step: experiment with hybrid retrieval in our conversational agent project. Tag as ai, rag, study, and research.
```

### Note 4: Grocery & Household Errands
```text
Create a note: Shopping List and Weekly Groceries. Need to buy ingredients for meal prep this Sunday. Produce: organic spinach, avocados, tomatoes, bell peppers, garlic, and fresh basil. Dairy & Protein: chicken breast, salmon fillets, almond milk, Greek yogurt, and eggs. Pantry: olive oil, quinoa, black beans, and whole wheat pasta. Also don't forget dishwasher pods and paper towels from the supermarket.
```

### Note 5: Project Infrastructure & Cloud Deployment
```text
Save a note: Infrastructure Deployment Guidelines for AWS. Here is the checklist for deploying microservices to AWS Elastic Kubernetes Service (EKS). First, ensure Terraform scripts are validated in the staging environment. Second, all Docker images must be scanned for vulnerabilities via AWS ECR before pushing to production. Third, environment variables and secret keys must be loaded from AWS Secrets Manager—never commit `.env` files to git repositories. Tag as devops, aws, cloud, and security.
```

---

## 🔍 2. Searching Notes (Testing the 3 Search Tools)

### Tool A: Keyword Search (SQL LIKE match)
- `Find notes with the word FastAPI`
- `Search for keyword Stripe`
- `Show notes containing Terraform`

### Tool B: Tag Search (Filtered by categories)
- `Show me notes tagged work`
- `Find notes with tag backend`
- `List notes tagged research or ai`

### Tool C: Semantic Search (Qdrant Vector Similarity & RAG)
- `What did we decide about our API backend framework?`
- `What are the action items for Sarah from the team standup?`
- `How should we handle API keys and secrets in AWS deployment?`
- `What did I learn about improving RAG performance with vector databases?`

---

## ✏️ 3. Updating Notes (Human-in-the-Loop & LLM Rewrite)

### Scenario A: Direct Update
1. User prompt: `Update my GenAI learning note to add reranking tests with Cohere API next week.`
2. Agent action: Finds candidate note → LLM generates updated note preview → prompts `Confirm update? (yes / no)`.
3. User prompt: `yes`

### Scenario B: Disambiguation before Update
1. User prompt: `Update the note about API` (Matches both "Backend Architecture" and "Infrastructure Deployment").
2. Agent action: Displays options `[1]` and `[2]` → prompts `Which one did you mean?`.
3. User prompt: `1`
4. Agent action: Previews updated note → prompts confirmation `yes / no`.
5. User prompt: `yes`

---

## 🗑️ 4. Deleting Notes (Disambiguation & Confirmation Safety Gate)

### Scenario A: Explicit Delete with Confirmation
1. User prompt: `Delete the Shopping List and Weekly Groceries note`
2. Agent action: Asks `Are you sure you want to delete 'Shopping List...'? (yes / no)`
3. User prompt: `yes`

### Scenario B: Cancelling a Delete
1. User prompt: `Delete note about Infrastructure`
2. Agent action: Prompts confirmation `(yes / no)`
3. User prompt: `no` or `cancel`
4. Agent result: Action cancelled. Note remains safe in both SQLite and Qdrant.

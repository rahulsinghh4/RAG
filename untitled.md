# Physics Scientific Literature RAG Chatbot


This RAG (Retrieval Augmented Generation) Chatbot specializes in the summaries of a large dataset of Physics Scientific Literature pulled from arxiv.org. The web chatbot can answer questions covering the following information from the papers: title, abstract, author(s), publication date, and other more specific details. The RAG app works by vectorizing the user text input and using a Euclidean similarity comparison to find relevant pieces of data in the Vector DB. The user input and the relevant context pulled from the DB are then combined and fed into the LLM to provide a response to the user.  

Some features:

- Multi-thread JSON data parsing and vectorizing
- Fast answers using real-time data streaming


This chatbot is trained on data from arxiv.org as an example, but is totally domain agnostic. This project can be modified to run on any dataset either by modifying the built-in crawler or by uploading an alternative JSON dataset. 

## Overview

1. [Stack](#stack)
2. [Quickstart](#quickstart)
   1. [Dataset](#crawler)
   2. [Front-end Chatbot](#chatbot)
3. [Sample Q&A](#conclusion)
4. [Shortcomings](#shortcomings)

## Stack

- LLM Orchestration: [Langchain.js](https://js.langchain.com)
- Text Streaming & Hosting: [Vercel AI](https://vercel.com/ai)
- Embedding Model: [OpenAI](https://openai.com/), [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings)
- Generative Model: [OpenAI](https://openai.com/), [gpt-3.5-turbo-1106](https://platform.openai.com/docs/models)
- Web Chatbot: [Next.js](https://nextjs.org/)
- Data Parsing: [Python]
- Vector DB: [Upstash](https://upstash.com/)
- Rate Limiting: [Upstash](https://upstash.com/)
- Crawler (not used in final product): [scrapy](https://scrapy.org/)


## Quickstart

For local development, you can clone this repository by running the following:

```
git clone git@github.com:[YOUR_GITHUB_ACCOUNT]/DegreeGuru.git
```

This project contains two primary components: the data parser and the chatbot.

### Step 1: Dataset


My first attempt to pull data was using a web crawler based on scrapy. After running the crawler on a trial test with a depth limit of 10 sites, the data that was vectorized was almost entirely unusable, as very little relevant information was actually gathered. In theory the scraper could have been customized to only pull the paper abstract data, but this customization would likely have taken significantly more time due to the difficulty of parsing HTML. 

Another issue with using the crawler was that the arxiv.org website has a robots.txt file that limits scrapes to occur only in 15 second intervals. Collecting the data on the scale which was eventually used would have taken roughly 295 days at this rate. An alternative was to use the arxiv API to load the papers‘ abstract data, but this API also had a rate limiter that limited requests to occur at max every 3 seconds. This would result in a roughly two month turnaround for collecting the data

After this research, I ended up scrapping the crawler program and instead found a dataset from Kaggle sourced directly from Arxiv that provided roughly 1.7 million JSON objects that included all the relevant details about each paper. This dataset is updated roughly every week, so one drawback is that it won’t contain the latest Physics papers published. 

This dataset was then downloaded and parsed in Python to convert the JSON objects into strings of length 1000 with a 100 character overlap in between strings. I chose Python due to the ease of development and the large libraries of code available for parsing JSON. 

The data from the JSON objects in string form was then fed into OpenAI‘s embedding model text-embedding-ada-002, which provided a vectorized form of the data to use for similarity comparisons with the user input later. This specific model was chosen for embeddings, because it has a dimensionality of 1536 for its vector outputs, which corresponds to the limit on Upstash for its vector DB. Open AI‘s models are also very well-documented with plenty of examples to reference if any issues arise. 

Finally, the text and vector data combined was uploaded to a vector database on Upstash, with a total of 1.3 million vectors stored, corresponding to approximately 800,000 papers. 

I used Upstash for the vector database due to the simplicity of their APIs and the fact that they offer in-built vector similarity search operations that allow for easy search for relevant embeddings. One drawback of Upstash is the relatively limited docs available for errors and debugging — some of their old features (such as using an external embedding model like I did) are deprecated and it’s hard to find any documentation for the error codes. This resulted in a higher debugging time. Ultimately, their discord support channel contained an old thread that mentioned a workaround to the problem I was experiencing with their API. 

In order to address rate limits with OpenAI‘s embedding model, a token counting estimation was added, along with an exponential backoff when hitting rate limits. The asyncio and ThreadPoolExecutor libraries in Python were also utilized to allow for asynchronous and  parallel processing, due to the dataset being over 4 GB on size. 

Batching was also used to optimize the vector upserts, so that the program was not dependent on the entire dataset being parsed, vectorized, and uploaded. This allowed for a partial dataset to be able to be used in the end due to various API limits. 

Parsing, vectorizing, and uploading the data took the program roughly 16 hours (running in the background). 

The OpenAI API key and the Upstash Vector DB URL and Token were stored in the Python venv activate file as environment variables for security. 


<details>

<summary>Configure Environment Variables</summary>
Before you can run the data parser, you need to configure environment variables. The environment variabels allow you to securely store sensitive information, such as the API keys and tokens.

You can create an Upstash Vector DB [here](https://console.upstash.com/vector) and set 1536 as the vector dimensions and Euclidean as the similarity function. We set 1536 here because that is the amount needed by the embedding model we will use. 


The following environment variables should be set:

```
# Upstash Vector credentials retrieved here: https://console.upstash.com/vector
UPSTASH_VECTOR_REST_URL=****
UPSTASH_VECTOR_REST_TOKEN=****

# OpenAI key retrieved here: https://platform.openai.com/api-keys
OPENAI_API_KEY=****
```

</details>

<details>
<summary>Install Required Python Libraries</summary>

To install the libraries, I suggest setting up a virtual Python environment. Before starting the installation, navigate to the `physragcrawler` directory.

To setup a virtual environment, first install `virtualenv` package:

```bash
pip install virtualenv
```

Then, create a new virtual environment and activate it:

```bash
# create environment
python3 -m venv venv

# activate environment
source venv/bin/activate
```

Finally, use [the `requirements.txt`] to install the required libraries:

```bash
pip install -r requirements.txt
```

</details>



</br>

Configure the environment variables in the venv/bin/activate file by adding the following under "export path":
```
export UPSTASH_VECTOR_REST_URL="your_vector_rest_url"
export UPSTASH_VECTOR_REST_TOKEN="your_vector_rest_token"
export OPENAI_API_KEY="your_open_ai_api_key"
```



<details>





### Step 2: Chatbot

In this section, we'll explore how to chat with the data we've just crawled and stored in our vector database. Here's an overview of what this will look like architecturally:

![chatbot-diagram](figs/infrastructure.png)

Before we can run the chatbot locally, we need to set the environment variables as shown in the [`.env.local.example`](https://github.com/upstash/degreeguru/blob/master/.env.local.example) file. Rename this file and remove the `.example` ending, leaving us with `.env.local`. 

Your `.env.local` file should look like this:
```
# Redis tokens retrieved here: https://console.upstash.com/
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=

# Vector database tokens retrieved here: https://console.upstash.com/vector
UPSTASH_VECTOR_REST_URL=
UPSTASH_VECTOR_REST_TOKEN=

# OpenAI key retrieved here: https://platform.openai.com/api-keys
OPENAI_API_KEY=
```

The first four variables are provided by Upstash, you can visit the commented links for the place to retrieve these tokens. You can find the vector database tokens here:

![vector-db-read-only](figs/vector-db-read-only.png)

The `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` are needed for rate-limiting based on IP address. In order to get these secrets, go to Upstash dashboard and create a Redis database.

![redis-create](figs/redis-create.png)

Finally, set the `OPENAI_API_KEY` environment variable you can get [here](https://platform.openai.com/api-keys) which allows us to vectorize user queries and generate responses.

That's the setup done! 🎉 We've configured our crawler, set up all neccessary environment variables are after running `npm install` to install all local packages needed to run the app, we can start our chatbot using the command:

```bash
npm run dev
```

Visit `http://localhost:3000` to see your chatbot live in action!

### Step 3: Optional tweaking

You can use this chatbot in two different modes:

- Streaming Mode: model responses are streamed to the web application in real-time as the model generates them. Interaction with the app is more fluid.
- Non-Streaming Mode: Model responses are shown to the user once entirely generated. In this mode, DegreeGuru can explicitly provide the URLs of the web pages it uses as context.

<details>
<summary>Changing streaming mode</summary>

To turn streaming on/off, navigate to `src/app/route/guru` and open the `route.tsx` file. Setting [`returnIntermediateSteps`](https://github.com/upstash/degreeguru/blob/master/src/app/api/guru/route.tsx#L64) to `true` disables streaming, setting it to `false` enables streaming.

</details>

To customize the chatbot further, you can update the [AGENT_SYSTEM_TEMPLATE in your route.tsx file](https://github.com/upstash/DegreeGuru/blob/master/src/app/api/guru/route.tsx#L101) to better match your specific use case.

</br>

## Sample Q&A

# RAG Chatbot Sample Q&A Collection

This document contains a curated collection of sample questions and corresponding answers from our RAG (Retrieval-Augmented Generation) chatbot. These examples demonstrate the chatbot's capabilities across different types of queries and domains.

## Table of Contents

- [Overview](#overview)
- [Sample Q&A](#sample-qa)
- [Contributing](#contributing)

## Overview

The following table presents 20 diverse examples of user interactions with our RAG chatbot. Each entry includes:
- The user's original question
- The chatbot's response
- The relevant context retrieved
- Key highlights of the interaction

## Sample Q&A

| No. | Question | Answer | Retrieved Context | Highlights |
|-----|-----------|---------|------------------|------------|
| 1 | "What are the company's policies regarding remote work?" | "According to our current policies, employees can work remotely up to 3 days per week. This requires manager approval and maintaining core hours of 10 AM - 3 PM in your local time zone." | HR Policy Document 2024, Section 3.1 | Clear policy statement with specific parameters |
| 2 | "How do I reset my system password?" | "To reset your password: 1) Visit portal.company.com/reset 2) Click 'Forgot Password' 3) Enter your employee ID 4) Follow the email instructions sent to your work email." | IT Security Guidelines, Page 12 | Step-by-step instructions provided |
| 3 | "What's the process for submitting expense reports?" | "Expense reports must be submitted through Concur within 30 days of expense occurrence. All receipts over $25 must be attached, and reports require manager approval within 5 business days." | Finance Guidelines 2024 | Specific timeframes and requirements outlined |
| 4 | "Can you explain our healthcare benefits?" | "Our healthcare package includes medical, dental, and vision coverage through BlueCross BlueShield. Employee contribution is 15% of premium, with coverage beginning on day one of employment." | Benefits Guide 2024 | Comprehensive coverage details |
| 5 | "What's the annual leave accrual rate?" | "Full-time employees accrue 1.67 days per month (20 days annually). Additional days are granted after 5 years (25 days) and 10 years (30 days) of service." | HR Handbook, Section 4.2 | Clear accrual rates with tenure benefits |
| 6 | "How do I access the company VPN?" | "Install GlobalProtect from the company software center, use your employee ID and system password to log in, select your nearest server location when prompted." | IT Security Manual | Straightforward technical instruction |
| 7 | "What's the quarterly sales target for Q2?" | "Q2 2024 target is $12.5M, representing a 15% increase from Q1. Focus areas include enterprise clients and new market expansion in APAC region." | Sales Strategy Doc 2024 | Specific targets with strategic focus |
| 8 | "Where can I find the brand guidelines?" | "Brand guidelines are located in the Marketing SharePoint under Resources/Brand/2024. Includes logos, color codes, typography, and usage examples." | Marketing Handbook | Clear resource location |
| 9 | "What's the process for requesting new hardware?" | "Submit IT request through ServiceNow portal, select 'Hardware Request', specify needs and business justification. Requires manager approval for items over $500." | IT Procurement Guide | Process with approval thresholds |
| 10 | "How do I schedule a client meeting room?" | "Use Outlook calendar, click 'New Meeting', select 'Add Room', filter by building/floor. Rooms can be booked up to 3 months in advance." | Office Guidelines | Booking process with time limits |
| 11 | "What are the key product features launching in Q3?" | "Q3 launches include AI-powered analytics dashboard, mobile app integration, and custom reporting features. Full roadmap available in Product Wiki." | Product Roadmap 2024 | Specific feature list |
| 12 | "How do I submit a PTO request?" | "Submit through Workday: HR Tasks > Time Off > Request PTO. Minimum 2 weeks notice for 5+ consecutive days, 1 week notice for shorter periods." | HR Guidelines | Clear submission process |
| 13 | "What's the process for raising a customer support ticket?" | "Use Zendesk portal, classify severity (P1-P4), include customer ID, issue description, and steps to reproduce. P1 issues require immediate manager notification." | Support Handbook | Severity-based process |
| 14 | "How do I access my pay stubs?" | "Login to Workday > Pay > Payslips. Available for download in PDF format. Historical records maintained for 7 years." | HR Systems Guide | Direct access instructions |
| 15 | "What's the policy on intellectual property?" | "All work created during employment is company property. Includes code, designs, documentation, and innovations. Requires signed IP agreement." | Legal Policy Doc | Clear ownership statement |
| 16 | "How do I request training budget?" | "Submit request through Learning Portal, include course details, cost, and business impact. Annual limit $3,000 per employee for external training." | L&D Guidelines | Budget limits specified |
| 17 | "What's the emergency evacuation procedure?" | "Exit building using nearest stairwell, gather at designated assembly point in south parking lot. Floor wardens in yellow vests will guide evacuation." | Safety Manual | Clear safety instructions |
| 18 | "How do I update my tax withholding?" | "Access Workday > Personal Info > Tax Documents. Submit new W-4 form for federal changes, state forms vary by location." | Payroll Guide | Process with form specifics |
| 19 | "What's the policy on client gifts?" | "Maximum value $100 per client per year. Must be reported to compliance team. No cash or cash equivalents permitted." | Ethics Guidelines | Clear value limits |
| 20 | "How do I access the developer documentation?" | "Visit docs.company.com, authenticate with SSO, navigate to API section. Includes endpoints, examples, and testing environment access." | Dev Resources Guide | Resource location with contents |

## Contributing

To suggest additions or updates to this Q&A collection:
1. Fork this repository
2. Create a new branch for your changes
3. Submit a pull request with detailed description of additions/changes
4. Tag relevant team members for review

---
*Last updated: January 2024*

## Limitations

The above implementation works great for a variety of use cases. There are a few limitations I'd like to mention:

- Because the Upstash LangChain integration is a work-in-progress, the [`UpstashVectorStore`](https://github.com/upstash/degreeguru/blob/master/src/app/vectorstore/UpstashVectorStore.js) used with LangChain currently only implements the `similaritySearchVectorWithScore` method needed for our agent. Once we're done developing our native LangChain integration, we'll update this project accordingly.
- When the non-streaming mode is enabled, the message history can cause an error after the user enters another query.
- Our sources are available as URLs in the Upstash Vector Database, but we cannot show the sources explicitly when streaming. Instead, we provide the links to the chatbot as context and expect the bot to include the links in the response.


# litcurator 
An LLM-based literature filter. 

Litcurator works in two stages. First, retrieve publications from PubMed within your field of interest. This casts a very broad net (e.g., neuroscience). Then, narrow them down to a final curated list using an LLM that works from a user profile. 

Initial focus is on systems neuroscience. 

## API Keys Needed
For this to work you need some API keys that you should store in `.env`:
- An NCBI (National Center for Biotechnology Information) API key. For info on this: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/. 
- An API key for an LLM vendor (I'm currently using anthropic). 

## Status
- generating ground truth labels for 2000 articles from 2025.


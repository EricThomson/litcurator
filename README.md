
# litcurator 

An LLM-based literature filter. 

Litcurator works in two stages. First, retrieve publications from PubMed within your field of interest. This casts a very broad net (e.g., neuroscience). Then, narrow them down to a final curated list using an LLM that builds up a profile of your interests. It uses [streamlit](https://streamlit.io/) to display articles in the browser.

Initial focus is on systems neuroscience. 

## API Keys Needed
For this to work you need some API keys that you should store in `.env`:
- An NCBI (National Center for Biotechnology Information) API key. For info on this: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/. 
- An API key for an LLM vendor (I'm currently using anthropic). 



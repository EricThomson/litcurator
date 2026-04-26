# litcurator 
Literature curator: utilities to retrieve publications from pubmed each month (or whatever time range you care about), and narrow them down to a final list using an LLM that has learned your interests (and continues to learn them each month). 

Initial focus is on systems neuroscience. :brain:

## Info
For this to work you need an NCBI (National Center for Biotechnology Information) API key. For info on this: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/. You also need an API key for an LLM (currently I am building uding anthropic). Put both API keys in local `.env` file. 


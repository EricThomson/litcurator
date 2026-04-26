# litcurator 
<img src="https://raw.githubusercontent.com/EricThomson/litcurator/main/src/litcurator/assets/litcurator_logo.svg" alt="litcurator logo" align="right" width="250">

Literature curator: utilities to retrieve publications from pubmed each month (or whatever time range you care about) within your field. Then, narrow them down to a final curated list using an LLM that has learned your interests (and continues to learn them each month). 

Initial focus is on systems neuroscience. :brain:

## API Keys Needed
For this to work you need some API keys that you should store in `.env`:
- An NCBI (National Center for Biotechnology Information) API key. For info on this: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/. 
- An API key for an LLM vendor (currently I am building uding anthropic). 


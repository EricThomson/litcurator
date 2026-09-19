
# litcurator 
An LLM-based literature filter. 

Litcurator works in two stages. First, retrieve publications from PubMed within your field of interest. Then, use an LLM (armed with a description of your specific interests) to narrow them down to a final curated list. 

Initial focus is on systems neuroscience. 

## API Keys Needed
For this to work you need some API keys that you should store in `.env`:
- An NCBI (National Center for Biotechnology Information) API key. For info on this: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/. 
- An API key for an LLM vendor (I'm currently using anthropic). 

## Status
- Major refactor with error analysis.  
- Run 2025 simulation with labeled data to see how it works.  


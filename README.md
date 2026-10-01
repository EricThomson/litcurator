
# litcurator 
An LLM-based literature filter. 

After retrieving publications from PubMed within your field of interest (e.g., neuroscience), it applies a two-stage LLM-based filter. A *coarse* filter coarse filter narrows things down within your field (e.g., systems neuroscience). The second finer-grained filter uses a specific user profile to extract papers with higher precision (e.g., somatosensory processing). 

Initial focus is on systems neuroscience, but litcurator should work for any field (with a little kneading). 

## API Keys Needed
For this to work you need some API keys that you should store in `.env`:
- An NCBI (National Center for Biotechnology Information) API key. For info on this: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/. 
- An API key for an LLM vendor (I'm currently using anthropic). 

## Status
- Running 2025 benchmark trial to compare results to ground truth labeled dataset. 
- Then will go live to see how we do. 



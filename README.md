# litcurator README
Literature curator: utilities to retrieve publications from pubmed, and narrow them down to a final list using an LLM. Initial focus is on systems neuroscience. :brain:

## Info
For this to work you need an NCBI (National Center for Biotechnology Information) API key. For info on this: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/api/api-keys/.

## To Do
- [ ] `retrieve.py` :  extract initial candidate articles from pubmed using esearch and efetch
- [ ] Figure how basic architecture: how should result be stored/compared across searches, bits be eliminated: likely a db, but what will db schema be?
- [ ] `rank.py` : narrow down to final list using LLM

## Done


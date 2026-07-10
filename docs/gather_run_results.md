# Gather experimental results and draw conclusions

## Goal

Given a repo and many experimental results, summarize or expand the main findings. Provide details, link supporting files, and prepare a short report / summary.

## Assumption

You will be give the following documents and information.

1. A local repo directory or a link to a Github repo. The repo contains basic information about the project, usually summarized in a readme.md file.
2. A local directory that contains many run subfolders. Each run subfolder contains relevant info about the run history, config files, and results. The results can be csv files, json files, figures, etc.
3. An initial document that outlines the skeleton of the report. The document serves as a starting point of your writing. from which you expand, revise, 
4. Optionally, you will be directed to read planning related documenets (e.g., plan.md).

Raise questions if you don't have access to all the required documents or links.


## Brief summary

You should first quickly scan the repo, readme.md, the structure of the run folders, etc. Use your words to provide a brief summary to the goal of the project, the scope of the repo, and what experiments have been already implemented.

- Write your summary in a few sentences, starting with a clear marker such as [Auto summary of experiments]
- If possible, write a few key hypothesis in concise languages: what the repo and run experiments aim to verify. Use bullet points to write each hypothsis.

## Expand and support existing conclusions 

When you read the initial document, one task for you is to expand existing summary.

1. You will look for existing summary / conclusions from the document. The summary is either numbered or starts with something like [Finding 1], **Finding 2**, etc.
2. Find supporting evidence in the run folders to support the conclusions. Usually the conclusion summarizes an empirical phenemenon from various experiments, or confirm a hypothesis. Please check the experiment setup / configs in each run folders, look into the csv, json, txt, or figures files. 
3. Use bullet points to state the supporting evidence you find. Each bullet point should be concise and describe one key aspect of the evidence. Then, use a nested bullet points to refer to specific path or filenames in the run folder, e.g., see row 10--12 of `{run_name}/tables/ablations/summary.csv`, see example.records.key in the json file `{run_name}/tables/results.jsonl`. It is recommended to use markdown-legitimate format to refer to a path or file whenever it is possible.
4. If the conclusion is weak or not well supported by the run folders, state the issue or objections in a new bullet point, starting with something like [Objection 1], **Objection 2**, etc.

## Propose new conclusions 

You also need to inspect the run folders carefully and come up with your own findings to complement existing summary / conclusions in the initial documents.

1. Start with something like [Finding 3, new], **Finding 2, new**, etc., if you propose new findings. Remember to put "new" to distinguish the initial findings in the document.
2. Similar to before, present evidence with concise statements in bullet points, followed by further links / explanations in nested bullet points.
3. You also need to write things like [Objection 1, new] if the conclusion can be refutued.

## To-do list

Check whether the run experiments support or disapprove the hypotheses. Add a list of possible experiments to do in the future, e.g., consider new config settings, try a differnet model family, etc. Use bullet points.

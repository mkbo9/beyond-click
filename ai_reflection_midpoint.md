# AI Reflection: Midpoint

**AI-aided pair:** Mihail mkbo9 (k12342321) & Friend First Friend Last (k11111111)

## Experience with GenAI
[Write your actual reflection text here... We already doing well]

What went well:
- An AI audit of the first plan and code found a silent bug: image features were matched to videos by row order only. Reordering the table dropped the test AUC from 0.63 to 0.37 with no error. Files are now matched by video id.
- A synthetic-data run lets me test the whole pipeline without an API key or model downloads. The shuffled-label control gives AUC ≈ 0.5, as it should.
- Shared project documents (plan, decisions, progress) keep separate chats consistent.

What was harder:
- A very long chat became unreliable, so I split the work into short chats with fixed rules.
- The AI sometimes assumed things that were wrong or explained too little. I had to ask for explicit reasons and check every file before pushing.


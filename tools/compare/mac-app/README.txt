Kera vs Ollama
==============

A small Mac app that races Kera against Ollama on this Mac, live, with one button.

Needs: an Apple-silicon Mac with macOS 14 or later, 64 GB of memory or more (the model is
loaded twice, once in each engine), about 25 GB free on the disk, and the internet the first
time (the model is downloaded once).

Opening it the first time
-------------------------
1. Drag "Kera vs Ollama" onto the Applications folder.
2. Open it from Applications. macOS says it can't check it (the app isn't signed with a paid
   Apple developer account): click Done.
3. Open System Settings > Privacy & Security, scroll down to "Kera vs Ollama was blocked",
   click "Open Anyway", and confirm with your password or Touch ID.
It opens normally from then on. (Or, in Terminal, once:
xattr -dr com.apple.quarantine "/Applications/Kera vs Ollama.app")

What "Start test" does
----------------------
1. Finds Ollama: the one running on this Mac if there is one, else the installed program,
   else it downloads Ollama into the app's own folder (nothing is installed).
2. Downloads the chosen model through Ollama, if it isn't there already. Both models offered
   are mixtures of experts (30B in all, about 3B for each token).
3. Starts Kera (inside the app) on Ollama's own copy of the model, so both run the very same
   file.
4. For 10 minutes (or 3 or 20), asks both the same messages, one engine at a time, taking
   turns at going first: a chat message, a piece of code, a sum to work out (the answer is
   checked), a 16K-token log with a question about it (checked) and a follow-up question on
   the same log (checked). Answers are greedy, with thinking off and capped lengths, and
   each engine keeps a 32K-token memory, so the speeds shown are for greedy answers, not
   for the models' usual settings. No message is asked twice: Kera's guessing ahead could
   copy its own earlier answer to it. When the chats, code and sums run out (after 12 of
   each), the test goes on with new logs only.
5. Shows each result as it comes in: writing speed, the wait for the first token, right
   answers, two charts, and the answer being written right now.

A token is about three quarters of a word. Results are saved as JSON in
~/Library/Application Support/KeraVsOllama (the "Show the results file" link), with both
servers' logs. Quitting the app stops the servers it started; an Ollama you were already
running is left as it was.


## 2026-05-29 — Verify the BUILD line, not just the install Success
**Mistake:** ran `assembleDebug | tail -4` then `adb install` chained with `&&`-less
sequencing; `adb install` printed "Success" because it installed the STALE prior
APK. I nearly reported a failed build as verified.
**Rules:**
- After any build, assert on the actual result: capture exit code or grep for
  `BUILD SUCCESSFUL`/`BUILD FAILED`. Never infer build success from a later
  `adb install` "Success" — install happily reuses the old APK.
- Check the APK timestamp (`ls --time-style`) to confirm it was rebuilt.
- Kotlin: augmented assignment on an array index (`arr[i] /= x`, `arr[i] += x`
  in some Float/Int mixes) can fail with "No set method providing array access".
  Use explicit `arr[i] = arr[i] / x`.

## 2026-05-29 (RECURRENCE) — Stale-APK trap bit me AGAIN
Despite logging the lesson earlier the SAME session, I chained
`assembleDebug | tail` + `adb install` and reported "verified" when the build had
FAILED and install reused the old APK. Screenshots were the stale app.
**Hard rule now:** after a build, the VERY NEXT thing is to assert
`gradle_exit=0` (capture `$?` right after gradlew, or grep `BUILD SUCCESSFUL`).
Do NOT run `adb install` until that assertion passes. And confirm APK mtime
advanced. Treating "install Success" as build success is banned.

## 2026-05-29 — Edit anchors must be verified against real file content
Two XML Edits silently failed because I guessed anchor strings (`cancel`,
`readout_placeholder`, a `backgroundTint` block) that weren't in the file; the
Read output was also garbled (zero-width chars, duplicate line numbers). When
Edit/Read are unreliable, patch via a Python script that searches for robust
anchors (e.g. `rfind('</FrameLayout>')`) and prints explicit success markers.

## 2026-05-29 — Measure before claiming a perf number
On the FPS task I predicted BlazeFace would ~double fps to ~40. Actual: 21→25
(+20%). And I guessed CPU>GPU for the tiny model; measured GPU>CPU. Both guesses
wrong. The PERF instrumentation caught it. Rule: never state an expected perf
multiplier as if it's a result; label predictions as predictions, then measure.
Also: on weak mobile GPUs (Mali-T830), GPU-delegate overhead can dominate for
small models — but TEST it, don't assume either direction.

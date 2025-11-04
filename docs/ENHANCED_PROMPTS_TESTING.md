# Enhanced Prompts Testing Documentation

## Date
2025-11-04

## Changes Made

### System Prompt (`py/reason/prompts/system_prompt.txt`)

**Before:**
- Simple instruction: "Answer only with information present in the provided context."
- No guidance on using metadata

**After:**
- Clear explanation of available resources (context entries + metadata)
- Explicit permission to use metadata for game/viewer/title/status questions
- Allows reasonable inferences when question is relevant but exact answer isn't in context
- More nuanced "I don't have enough information" guidance

### User Prompt (`py/reason/prompts/user_prompt.txt`)

**Before:**
- Basic instructions with strict "use only context" rule
- No examples
- No differentiation between question types

**After:**
- Expanded instructions covering all question types
- Few-shot examples for:
  1. METADATA QUESTIONS (5 examples)
  2. EVENT QUESTIONS (2 examples)
  3. TRANSCRIPT QUESTIONS (2 examples)
  4. INFERENCE QUESTIONS (2 examples)
  5. NO INFORMATION (1 example)
- Guidance on when to use citations vs when metadata is sufficient
- Encourages conversational, helpful responses
- Allows inference when question is stream-relevant

## Key Improvements

1. **Metadata Usage**: LLM can now answer "What game are they playing?" using system prompt metadata
2. **Inference**: LLM can make reasonable inferences from context even if exact answer isn't present
3. **Few-Shot Examples**: Clear examples show expected behavior for each question type
4. **Better Guidance**: More nuanced rules about when to say "I don't have enough information"

## Test Questions

### Original JCB-22 Questions
1. "What game are they playing?" (METADATA)
2. "Who raided?" (EVENT)
3. "When did the stream start?" (EVENT)
4. "What did they say about the boss?" (TRANSCRIPT)
5. "Is there a prediction active?" (EVENT)

### Additional Test Questions

#### Metadata Questions
- "How many viewers are watching?" 
- "What's the stream title?"
- "Are they live right now?"

#### Event Questions
- "Has anyone raided recently?"
- "Any subscriptions today?"

#### Transcript Questions
- "What are they talking about?"
- "What's the current topic of discussion?"
- "What game are they discussing?"

#### Inference Questions
- "What's happening right now?"
- "Are they winning?"
- "How's the stream going?"
- "What's the mood of the stream?"

#### Edge Cases
- "What's their favorite color?" (should say no info)
- "What's the weather like?" (should say no info)

## Expected Behavior Changes

### Before Enhanced Prompts
- "What game are they playing?" → "I don't have enough information" (even though metadata has it)
- Strict adherence to context only
- No inference allowed

### After Enhanced Prompts
- "What game are they playing?" → "They're currently playing Just Chatting." (from metadata)
- "What's happening right now?" → Combines metadata + context for comprehensive answer
- "Are they winning?" → Makes reasonable inference from context about game state
- Better contextual responses even when exact answer isn't in context

## Testing Instructions

1. **Start the API server:**
   ```bash
   uvicorn py.main:app --reload --port 8000
   ```

2. **Run the test script:**
   ```bash
   python scripts/test_enhanced_prompts.py
   ```

3. **Manual testing via API:**
   ```powershell
   $body = @{channel='672238954';question='What game are they playing?'} | ConvertTo-Json
   Invoke-RestMethod -Uri 'http://localhost:8000/rag/answer' -Method Post -Body $body -ContentType 'application/json'
   ```

4. **Evaluate results:**
   - Check if metadata questions now use system prompt metadata
   - Verify inference questions provide helpful answers
   - Ensure edge cases still correctly say "I don't have enough information"
   - Confirm citations are included for transcript/event references

## Success Criteria

- ✅ Metadata questions answered using system prompt (no citations needed)
- ✅ Event questions answered with event context when available
- ✅ Transcript questions answered with transcript context + citations
- ✅ Inference questions provide helpful answers based on available info
- ✅ Edge cases correctly return "I don't have enough information"
- ✅ More conversational, helpful tone in responses
- ✅ Better use of combined metadata + context for comprehensive answers

## Iteration Notes

If results aren't satisfactory, consider:
1. Adding more few-shot examples for problematic question types
2. Adjusting inference guidance (more/less permissive)
3. Clarifying citation requirements
4. Fine-tuning metadata usage instructions

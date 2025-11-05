# Multi-User Parallel Testing Documentation

## Overview

This document describes the multi-user parallel testing suite for Phase 5.5, which verifies that the bot can handle multiple concurrent users correctly, maintaining context isolation and providing personalized responses.

## Test Scenarios

### 1. Concurrent Different Questions

**Purpose**: Verify system can handle 5+ users asking different questions simultaneously.

**Test Steps**:
1. 5 users send different questions at the same time
2. Wait for responses
3. Verify each user received a response
4. Measure latency

**Expected Results**:
- All users receive responses
- Average latency < 5s
- No errors or timeouts

**Acceptance Criteria**:
- ✅ All users get responses
- ✅ Latency stays within acceptable range

### 2. Follow-up Questions

**Purpose**: Verify follow-up questions work correctly using session history.

**Test Steps**:
1. User asks initial question
2. Wait for response
3. User asks follow-up question referencing previous answer
4. Verify response is contextually aware

**Expected Results**:
- Follow-up response references or builds on previous answer
- Session history is correctly retrieved and used
- Response is personalized to user's conversation

**Acceptance Criteria**:
- ✅ Follow-up questions answered correctly
- ✅ Context from previous Q&A included
- ✅ Response is contextually aware

### 3. Context Isolation

**Purpose**: Verify users don't see each other's conversation history.

**Test Steps**:
1. User A asks question about topic X
2. User B asks question about topic Y
3. Verify responses are different and appropriate

**Expected Results**:
- Each user gets a response relevant to their question
- No context bleeding between users
- Responses are isolated per user

**Acceptance Criteria**:
- ✅ No context mixing between users
- ✅ Each user gets personalized response
- ✅ Session isolation maintained

### 4. Rate Limiting with Concurrent Users

**Purpose**: Verify rate limiting works correctly with multiple users.

**Test Steps**:
1. Single user sends two questions rapidly
2. Verify only one is processed (rate limited)
3. Verify other users aren't affected

**Expected Results**:
- Rate limit enforced per-user
- Other users not affected
- Second question blocked or delayed

**Acceptance Criteria**:
- ✅ Rate limiting works per-user
- ✅ No interference between users
- ✅ Limits enforced correctly

### 5. Session Persistence

**Purpose**: Verify session history persists across multiple questions.

**Test Steps**:
1. User asks first question
2. Wait for rate limit to expire
3. User asks second question
4. Verify session persisted

**Expected Results**:
- Session history maintained
- Second question can reference first
- Redis session stored correctly

**Acceptance Criteria**:
- ✅ Session persists across questions
- ✅ History available for follow-ups
- ✅ Redis storage working

### 6. Mixed Context Questions

**Purpose**: Verify system handles mix of in-context and out-of-context questions.

**Test Steps**:
1. Multiple users ask different types of questions
2. Mix of transcript-related and general questions
3. Verify all processed correctly

**Expected Results**:
- All questions processed
- Appropriate responses for each
- No errors with mixed contexts

**Acceptance Criteria**:
- ✅ All questions processed
- ✅ Appropriate responses
- ✅ System handles variety

## Performance Benchmarks

### Target Metrics

- **Response Latency**: < 5s per response (average)
- **P95 Latency**: < 8s
- **P99 Latency**: < 10s
- **Concurrent Users**: Support 5+ simultaneous users
- **Error Rate**: < 1%

### Measuring Performance

The test suite tracks:
- Average response time per user
- P95/P99 latency percentiles
- Total requests processed
- Error count
- Per-user latency breakdown

## Running Tests

### Prerequisites

1. Python service running on `http://localhost:8000`
2. Redis running and accessible
3. PostgreSQL with vector store populated (for realistic responses)
4. All dependencies installed

### Command

```bash
python scripts/test_multi_user_parallel.py
```

### Environment Variables

No special environment variables required for testing, but ensure:
- `REDIS_HOST` and `REDIS_PORT` are set correctly
- `OPENAI_API_KEY` is set (for RAG responses)
- Service is accessible at configured URL

## Interpreting Results

### Test Results

- **[OK] [PASS]**: Test passed successfully
- **[X] [FAIL]**: Test failed, check details

### Performance Metrics

- **Average Latency**: Should be < 5s
- **P95/P99**: Higher percentiles, indicates tail latency
- **Errors**: Should be 0 or minimal
- **Per-User Latency**: Helps identify user-specific issues

### Common Issues

1. **High Latency**: Check LLM API response times, Redis performance
2. **Missing Responses**: Check rate limiting, service logs
3. **Context Bleeding**: Verify session isolation in Redis
4. **Follow-up Failures**: Check session history retrieval

## Known Limitations

1. **Test Data**: Uses mock users and questions - may not reflect real-world patterns
2. **Timing**: Tests rely on fixed delays - may need adjustment for slower systems
3. **Load**: Tests simulate 5 users - not representative of high-traffic scenarios
4. **Context**: Requires vector store to have relevant transcripts for realistic responses

## Future Enhancements

1. **Load Testing**: Test with 10+ concurrent users
2. **Stress Testing**: Test system limits and failure modes
3. **Realistic Patterns**: Use actual Twitch chat patterns
4. **Monitoring**: Integrate with performance monitoring tools
5. **Continuous Testing**: Add to CI/CD pipeline

## Troubleshooting

### Service Not Available

**Error**: `Service is not available`

**Solution**:
- Ensure Python service is running: `uvicorn py.main:app --reload --port 8000`
- Check service URL in test script
- Verify port is not blocked

### No Responses Received

**Error**: Users not receiving responses

**Solution**:
- Check RAG service is initialized
- Verify OpenAI API key is set
- Check Redis connection
- Review service logs for errors

### High Latency

**Error**: Average latency > 5s

**Solution**:
- Check OpenAI API response times
- Verify Redis performance
- Check network latency
- Review database query performance

### Context Bleeding

**Error**: Users seeing each other's context

**Solution**:
- Verify Redis session keys are user-specific
- Check session retrieval logic
- Ensure user_id is passed correctly to RAG service

## Related Documentation

- [Rate Limiting Test Results](../docs/RATE_LIMITING_TEST_RESULTS.md)
- [Architecture Documentation](../docs/ARCHITECTURE.md)
- [Redis Session Management](../py/memory/redis_session.py)


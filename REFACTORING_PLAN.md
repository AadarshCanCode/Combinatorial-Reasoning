# CRQUBO Refactoring Plan

## Overview
This document tracks the systematic refactoring of CRQUBO based on issues identified in `DEEP_ANALYSIS.md`.

## Iteration Status

### Iteration 1: Foundation & Critical Fixes ✅ PLANNED
**Scope**: Old API usage, error handling basics, logging infrastructure
- [ ] Migrate from deprecated OpenAI API to new client (v1.0+)
- [ ] Create custom exception hierarchy
- [ ] Implement structured logging throughout
- [ ] Add basic resource limits (timeouts, max retries)

### Iteration 2: Configuration & Security 🔄 TODO
**Scope**: Config management, API key security
- [ ] Remove lru_cache from load_config
- [ ] Implement Pydantic config models with validation
- [ ] Centralize API key management
- [ ] Add environment-based configuration profiles

### Iteration 3: Type Safety & Contracts 🔄 TODO
**Scope**: Type hints, data models, interfaces
- [ ] Replace Dict[str, Any] with Pydantic models
- [ ] Add strict type checking
- [ ] Define clear module interfaces
- [ ] Version data contracts

### Iteration 4: Dependency Injection 🔄 TODO
**Scope**: Decouple modules, improve testability
- [ ] Implement constructor-based dependency injection
- [ ] Add factory patterns for module creation
- [ ] Create abstract base classes for all modules
- [ ] Add module registry

### Iteration 5: Observability & Monitoring 🔄 TODO
**Scope**: Metrics, tracing, debugging
- [ ] Add OpenTelemetry instrumentation
- [ ] Implement performance metrics
- [ ] Add distributed tracing
- [ ] Create debugging utilities

### Iteration 6: Resource Management 🔄 TODO
**Scope**: Memory, caching, async support
- [ ] Implement caching strategy
- [ ] Add memory limits
- [ ] Begin async/await migration
- [ ] Add connection pooling

### Iteration 7: Testing Infrastructure 🔄 TODO
**Scope**: Comprehensive test coverage
- [ ] Add unit tests with mocks
- [ ] Create integration test suite
- [ ] Add property-based tests
- [ ] Implement test fixtures

### Iteration 8: Documentation & Polish 🔄 TODO
**Scope**: Code documentation, examples
- [ ] Add comprehensive docstrings
- [ ] Create architecture diagrams
- [ ] Update examples
- [ ] Write migration guides

## Iteration Details

### ITERATION 1: Foundation & Critical Fixes

#### Problem 1: Old OpenAI API Usage
**Severity**: HIGH  
**Files**: `crqubo/modules/reason_sampler.py`, `crqubo/modules/final_inference.py`

**Current Issues**:
- Using deprecated `openai.ChatCompletion.create()`
- No async support
- Poor error handling

**Solution**:
1. Migrate to OpenAI v1.0+ client
2. Use new client initialization pattern
3. Add proper error handling for new exception types
4. Prepare for async support

**Changes Required**:
- Update OpenAISampler class to use new client
- Update OpenAIInferenceEngine class
- Add retry logic with exponential backoff
- Handle rate limiting properly

#### Problem 2: Error Handling
**Severity**: HIGH  
**Files**: All modules

**Current Issues**:
- Mix of print(), exceptions, and silent failures
- No structured error reporting
- Generic Exception catching

**Solution**:
1. Create custom exception hierarchy
2. Replace print() with logging
3. Add error context and recovery hints
4. Implement error aggregation

**Changes Required**:
- Create `crqubo/exceptions.py` with custom exceptions
- Update all modules to use new exceptions
- Add logging configuration
- Replace print statements with logging calls

#### Problem 3: Logging Infrastructure
**Severity**: HIGH  
**Files**: All modules

**Current Issues**:
- Basic logging only
- No structured logging
- Inconsistent log levels

**Solution**:
1. Implement structured logging
2. Add log levels consistently
3. Include context in logs
4. Add performance logging

**Changes Required**:
- Create logging configuration module
- Add context managers for operation logging
- Include timing information
- Add debug utilities

#### Problem 4: Resource Limits
**Severity**: HIGH  
**Files**: `crqubo/core.py`, module files

**Current Issues**:
- No timeouts on operations
- Unlimited retries possible
- No circuit breakers

**Solution**:
1. Add configurable timeouts
2. Implement retry limits
3. Add circuit breaker pattern
4. Track resource usage

**Changes Required**:
- Add timeout decorators
- Implement retry logic with limits
- Add circuit breaker for API calls
- Track and log resource usage

## Testing Strategy Per Iteration
- Unit tests for each changed module
- Integration tests for pipeline
- Backward compatibility tests
- Performance regression tests

## Rollback Plan
- Each iteration on separate branch
- Full test suite must pass before merge
- Keep deprecated code with warnings initially
- Document breaking changes

## Success Criteria
- All tests pass
- No functionality loss
- Improved error messages
- Better debugging capabilities
- Reduced technical debt

---

**Current Iteration**: 1  
**Last Updated**: 2024  
**Status**: In Progress

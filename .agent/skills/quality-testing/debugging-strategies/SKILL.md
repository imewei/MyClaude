---
name: debugging-strategies
version: "1.0.7"
maturity: "5-Expert"
specialization: Systematic Debugging
description: Systematic debugging with scientific method, profiling, RCA across any stack. Use for runtime errors, performance issues, memory leaks, flaky bugs, production debugging with Chrome DevTools, VS Code, pdb/ipdb, Delve, git bisect.
---

# Debugging Strategies

## Scientific Method
1. Observe: Actual behavior?
2. Hypothesize: Possible causes?
3. Experiment: Test hypothesis
4. Analyze: Proved/disproved?
5. Repeat: Until root cause found

## Process

| Phase | Actions |
|-------|---------|
| Reproduce | Minimal repro, exact steps |
| Info | Stack trace, env, recent changes |
| Hypothesize | What changed? Where fail? |
| Test | Binary search, logging, isolate |
| Fix | Root cause, not symptoms |

## Tools

### JavaScript/TypeScript
```typescript
debugger;  // Pause here
console.log('Value:', v);
console.table(arr);
console.time('op'); /* code */ console.timeEnd('op');
console.trace();  // Stack
performance.mark('start'); /* op */ performance.mark('end');
performance.measure('op', 'start', 'end');
```

VS Code launch.json:
```json
{"type":"node","request":"launch","name":"Debug","program":"${workspaceFolder}/src/index.ts","outFiles":["${workspaceFolder}/dist/**/*.js"]}
```

### Python
```python
import pdb; pdb.set_trace()  # or breakpoint()
try: risky()
except: import pdb; pdb.post_mortem()
import cProfile; cProfile.run('slow()', 'stats')
```

### Go
```go
import "runtime/debug"
debug.PrintStack()
defer func() { if r := recover(); r != nil { debug.PrintStack() }}()
// pprof: http://localhost:6060/debug/pprof/
import _ "net/http/pprof"
```

### Git Bisect
```bash
git bisect start
git bisect bad            # Current broken
git bisect good v1.0.0    # Known working
git bisect good|bad       # Repeat
git bisect reset
```

## Differential Debugging

| Aspect | Working | Broken |
|--------|---------|--------|
| Env | Dev | Prod |
| Data | Empty | 1M records |
| User | Admin | Regular |
| Browser | Chrome | Safari |
| Time | Day | Midnight |

→ Form hypothesis from differences

## By Issue Type

**Intermittent/Flaky**: Extensive logging + timing, check race conditions, async ordering, stress test
**Performance**: Profile first, check N+1 queries, unnecessary renders, sync I/O. Tools: DevTools, cProfile, clinic.js
**Memory Leaks**:
```javascript
if (process.memoryUsage().heapUsed > 500*1024*1024) require('v8').writeHeapSnapshot();
```
**Production**: Gather evidence, reproduce locally with prod data, never change prod directly

## Best Practices

| Practice | Why |
|----------|-----|
| Reproduce first | Can't fix what you can't reproduce |
| Isolate | Remove complexity to minimal case |
| Read errors | Usually helpful |
| Check recent | Most bugs are recent |
| Take breaks | Fresh eyes see better |
| Fix root | Not symptoms |

## Common Mistakes

| Mistake | Problem |
|---------|---------|
| Multiple changes | Can't ID what fixed |
| Ignore stack trace | Miss obvious clues |
| Debug logs in prod | Security, perf risk |
| Give up soon | Persistence pays |
| Assume complex | Often simple |

## Quick Checklist

- [ ] Typos, case sensitivity
- [ ] Null/undefined
- [ ] Array off-by-one
- [ ] Async/race conditions
- [ ] Type mismatches
- [ ] Environment vars
- [ ] Cache issues
- [ ] Absolute vs relative paths

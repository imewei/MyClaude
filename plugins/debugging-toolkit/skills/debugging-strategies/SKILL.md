---
name: debugging-strategies
version: "1.0.5"
maturity: "5-Expert"
specialization: Systematic Debugging
description: Apply systematic debugging with scientific method, profiling tools, and root cause analysis across any stack. Use for runtime errors, performance issues, memory leaks, flaky bugs, and production debugging with Chrome DevTools, VS Code, pdb/ipdb, Delve, and git bisect.
---

# Debugging Strategies

Transform debugging from guesswork into systematic problem-solving.

---

## The Scientific Method

1. **Observe**: What's the actual behavior?
2. **Hypothesize**: What could cause it?
3. **Experiment**: Test your hypothesis
4. **Analyze**: Did it prove/disprove theory?
5. **Repeat**: Until root cause found

---

## Systematic Process

| Phase | Actions |
|-------|---------|
| 1. Reproduce | Create minimal reproduction, document exact steps |
| 2. Gather Info | Full stack trace, environment, recent changes |
| 3. Hypothesize | What changed? What's different? Where could this fail? |
| 4. Test | Binary search, logging, isolate components |
| 5. Fix | Address root cause, not symptoms |

---

## JavaScript/TypeScript

```typescript
// Breakpoint
debugger;  // Execution pauses here

// Console techniques
console.log('Value:', value);
console.table(arrayOfObjects);
console.time('op'); /* code */ console.timeEnd('op');
console.trace();  // Stack trace

// Performance
performance.mark('start');
// ... operation
performance.mark('end');
performance.measure('operation', 'start', 'end');
```

### VS Code launch.json
```json
{
    "type": "node",
    "request": "launch",
    "name": "Debug",
    "program": "${workspaceFolder}/src/index.ts",
    "outFiles": ["${workspaceFolder}/dist/**/*.js"]
}
```

---

## Python

```python
# Built-in debugger
import pdb
pdb.set_trace()  # or: breakpoint() (Python 3.7+)

# Post-mortem debugging
try:
    risky_operation()
except Exception:
    import pdb; pdb.post_mortem()

# Profile performance
import cProfile
cProfile.run('slow_function()', 'profile_stats')
```

---

## Go

```go
import "runtime/debug"

// Print stack trace
debug.PrintStack()

// Panic recovery
defer func() {
    if r := recover(); r != nil {
        debug.PrintStack()
    }
}()

// pprof: http://localhost:6060/debug/pprof/
import _ "net/http/pprof"
```

---

## Git Bisect (Binary Search)

```bash
git bisect start
git bisect bad                # Current is broken
git bisect good v1.0.0        # Known working version
# Git checks out middle, you test, then:
git bisect good   # or
git bisect bad
# Repeat until bug found
git bisect reset
```

---

## Differential Debugging

| Aspect | Working | Broken |
|--------|---------|--------|
| Environment | Development | Production |
| Data | Empty DB | 1M records |
| User | Admin | Regular user |
| Browser | Chrome | Safari |
| Time | During day | After midnight |

→ Form hypothesis from differences

---

## Debugging by Issue Type

### Intermittent/Flaky Bugs
- Add extensive logging with timing
- Look for race conditions
- Check async operation ordering
- Stress test repeatedly

### Performance Issues
- Profile before optimizing
- Check: N+1 queries, unnecessary renders, synchronous I/O
- Tools: DevTools Performance, cProfile, clinic.js

### Memory Leaks
```javascript
// Node.js heap snapshot
if (process.memoryUsage().heapUsed > 500 * 1024 * 1024) {
    require('v8').writeHeapSnapshot();
}
```

### Production Bugs
- Gather evidence (error tracking, logs, metrics)
- Reproduce locally with production data
- Never change production directly

---

## Best Practices

| Practice | Why |
|----------|-----|
| Reproduce first | Can't fix what you can't reproduce |
| Isolate the problem | Remove complexity until minimal case |
| Read error messages | They're usually helpful |
| Check recent changes | Most bugs are recent |
| Take breaks | Fresh eyes see better |
| Fix root cause | Not just symptoms |

---

## Common Mistakes

| Mistake | Problem |
|---------|---------|
| Multiple changes at once | Can't identify what fixed it |
| Not reading stack trace | Missing obvious clues |
| Debug logging in prod | Security risk, performance |
| Giving up too soon | Persistence pays off |
| Assuming it's complex | Often it's simple |

---

## Quick Checklist

When stuck, check:
- [ ] Typos (variable names, case sensitivity)
- [ ] Null/undefined values
- [ ] Array index off-by-one
- [ ] Async timing / race conditions
- [ ] Type mismatches
- [ ] Environment variables
- [ ] Cache issues
- [ ] File paths (absolute vs relative)

---

**Version**: 1.0.5

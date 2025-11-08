# Session Management Guide

**Version**: 1.0.3
**Category**: codebase-cleanup
**Purpose**: State tracking, progress management, and session continuity for long-running cleanup operations

## Session State Structure

```typescript
interface SessionState {
    version: string;
    sessionId: string;
    created: string;
    updated: string;
    totalItems: number;
    completed: number;
    remaining: number;
    currentIndex: number;
    status: 'in_progress' | 'completed' | 'paused' | 'failed';
    decisions: Record<string, any>;
    gitCheckpoint?: string;
}
```

## Progress Tracking

```typescript
class ProgressTracker {
    private state: SessionState;

    constructor(private sessionDir: string) {
        this.state = this.loadOrCreate();
    }

    update(updates: Partial<SessionState>): void {
        this.state = {
            ...this.state,
            ...updates,
            updated: new Date().toISOString()
        };
        this.save();
    }

    markCompleted(index: number): void {
        this.state.completed++;
        this.state.remaining--;
        this.state.currentIndex = index + 1;
        this.save();
    }

    getProgress(): { percentage: number; eta: string } {
        const percentage = (this.state.completed / this.state.totalItems) * 100;
        const avgTimePerItem = this.calculateAvgTime();
        const remainingTime = avgTimePerItem * this.state.remaining;

        return {
            percentage: Math.round(percentage),
            eta: this.formatDuration(remainingTime)
        };
    }
}
```

## Decision Tracking

```typescript
interface DecisionRecord {
    questionId: string;
    timestamp: string;
    choice: string;
    appliedTo: string[];
    confidence: number;
}

class DecisionManager {
    recordDecision(question: string, choice: string, context: any): void {
        const decision: DecisionRecord = {
            questionId: this.hashQuestion(question),
            timestamp: new Date().toISOString(),
            choice,
            appliedTo: [context.file],
            confidence: 1.0
        };

        this.saveDecision(decision);
    }

    findSimilarDecision(question: string): DecisionRecord | null {
        const questionId = this.hashQuestion(question);
        return this.loadDecision(questionId);
    }

    applyConsistentChoice(similarItems: any[]): void {
        // Apply same decision to similar scenarios
        for (const item of similarItems) {
            this.applyDecision(item);
        }
    }
}
```

## Resume Capability

```typescript
class SessionResume {
    canResume(sessionDir: string): boolean {
        return fs.existsSync(path.join(sessionDir, 'state.json'));
    }

    async resume(sessionDir: string): Promise<SessionState> {
        const state = this.loadState(sessionDir);

        console.log(`
╔══════════════════════════════════════╗
║   RESUMING SESSION                   ║
╠══════════════════════════════════════╣
║ Total Items:    ${state.totalItems.toString().padEnd(20)}║
║ Completed:      ${state.completed.toString().padEnd(20)}║
║ Remaining:      ${state.remaining.toString().padEnd(20)}║
║ Progress:       ${((state.completed/state.totalItems)*100).toFixed(1)}%${' '.repeat(17)}║
╚══════════════════════════════════════╝
        `);

        return state;
    }
}
```

## Checkpoint Management

```bash
#!/bin/bash
# Create git checkpoint before making changes

create_checkpoint() {
    local checkpoint_name="cleanup-checkpoint-$(date +%Y%m%d-%H%M%S)"

    # Stash all changes
    git add -A
    git stash push -m "$checkpoint_name"

    # Save stash hash
    local stash_hash=$(git rev-parse stash@{0})
    echo $stash_hash > .cleanup-session/git-checkpoint.txt

    echo "✓ Checkpoint created: $checkpoint_name"
    echo "  Hash: $stash_hash"
}

restore_checkpoint() {
    local stash_hash=$(cat .cleanup-session/git-checkpoint.txt)

    echo "Restoring from checkpoint: $stash_hash"
    git stash apply $stash_hash

    echo "✓ Checkpoint restored"
}
```

## Session Archiving

```typescript
function archiveSession(sessionDir: string): void {
    const timestamp = new Date().toISOString().replace(/:/g, '-');
    const archiveDir = path.join(sessionDir, 'archive', timestamp);

    fs.mkdirSync(archiveDir, { recursive: true });
    fs.copyFileSync(
        path.join(sessionDir, 'state.json'),
        path.join(archiveDir, 'state.json')
    );
    fs.copyFileSync(
        path.join(sessionDir, 'plan.md'),
        path.join(archiveDir, 'plan.md')
    );

    console.log(`Session archived to: ${archiveDir}`);
}
```

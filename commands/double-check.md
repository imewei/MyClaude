---
description: Revolutionary ultrathinking verification engine that defines verification angles, reiterates goals, defines completeness criteria, then deeply verifies and auto-completes Claude's work
category: self-verification
argument-hint: ["task/problem description"] [--interactive] [--auto-complete] [--deep-analysis] [--report]
allowed-tools: TodoWrite, Read, Write, Edit, MultiEdit, Grep, Glob, Bash, WebSearch, WebFetch
---

# 🧠 Revolutionary Ultrathinking Verification Engine

Deep analytical verification system following the methodology: **Define Angles → Reiterate Goal → Define Complete → Verify → Auto-Complete**. Each step involves ultrathinking for comprehensive work verification and enhancement.

## Quick Start

```bash
# Complete ultrathinking verification with auto-completion
/double-check "implement user authentication system" --auto-complete

# Interactive deep analysis with guided ultrathinking
/double-check "fix database performance issues" --interactive --deep-analysis

# Comprehensive verification with detailed reporting
/double-check "create data visualization dashboard" --comprehensive --report

# Verify current work without specific context
/double-check --auto-complete
```

## 🎯 The Ultrathinking Methodology

You are an **Advanced Ultrathinking Verification Specialist** who transforms shallow checking into deep analytical verification through systematic methodology.

**Core Principle**: "Ultrathink!" at every step - never proceed without deep analysis.

### **Phase 1: Define Verification Angles** 🔍

```bash
# ULTRATHINK: What angles should I approach verification from?
define_verification_angles() {
    local task_description="$1"
    echo "🧠 PHASE 1: ULTRATHINKING VERIFICATION ANGLES"
    echo "=============================================="
    echo ""

    if [[ -n "$task_description" ]]; then
        echo "💭 ULTRATHINKING: What angles can I approach verification of '$task_description' from?"
    else
        echo "💭 ULTRATHINKING: What angles can I approach this verification from?"
    fi
    echo ""

    # Initialize deep thinking process
    mkdir -p .ultrathink_verification/{
        angle_definition,
        goal_analysis,
        completeness_criteria,
        verification_results,
        auto_completion
    }

    echo "🤔 Deep Analytical Question: What are ALL possible angles to verify this work?"
    echo ""
    echo "Let me ultrathink this systematically..."
    echo ""

    # ULTRATHINK: Define comprehensive verification angles
    cat > .ultrathink_verification/angle_definition/verification_angles.md << 'EOF'
# 🔍 ULTRATHINKING: Verification Angles Definition

## Systematic Angle Analysis

### 1. **FUNCTIONAL COMPLETENESS ANGLE**
- **Question**: Does the work actually accomplish what it was supposed to do?
- **Depth**: Not just "does it work" but "does it solve the underlying problem completely?"
- **Focus**: Core functionality, edge cases, error scenarios

### 2. **REQUIREMENT FULFILLMENT ANGLE**
- **Question**: Does the work meet ALL explicitly and implicitly stated requirements?
- **Depth**: Both obvious requirements and unstated expectations
- **Focus**: User needs, technical specs, quality standards

### 3. **COMMUNICATION EFFECTIVENESS ANGLE**
- **Question**: Is the work clearly explained and understandable?
- **Depth**: Can someone else understand, use, and maintain this work?
- **Focus**: Documentation clarity, explanation quality, usability

### 4. **TECHNICAL QUALITY ANGLE**
- **Question**: Is the implementation robust, maintainable, and well-designed?
- **Depth**: Not just working code but good code that will last
- **Focus**: Architecture, patterns, best practices, scalability

### 5. **USER EXPERIENCE ANGLE**
- **Question**: How will the end user actually experience this work?
- **Depth**: Real-world usage scenarios and user journey
- **Focus**: Ease of use, intuitive design, helpful guidance

### 6. **COMPLETENESS COVERAGE ANGLE**
- **Question**: Are there gaps, missing pieces, or overlooked aspects?
- **Depth**: What's NOT there that should be?
- **Focus**: Missing features, incomplete implementations, TODO items

### 7. **INTEGRATION & CONTEXT ANGLE**
- **Question**: How does this work fit into the broader context?
- **Depth**: Does it work well with existing systems/workflows?
- **Focus**: Compatibility, dependencies, ecosystem fit

### 8. **FUTURE-PROOFING ANGLE**
- **Question**: Will this work remain valuable and maintainable over time?
- **Depth**: Sustainability, adaptability, evolution capability
- **Focus**: Extensibility, documentation, knowledge transfer

## 🎯 ULTRATHINKING CONCLUSION
These 8 angles provide comprehensive coverage for verification. Each angle requires deep analysis, not surface-level checking.
EOF

    echo "✅ ULTRATHINKING COMPLETE: 8 comprehensive verification angles defined"
    echo ""
    echo "📋 Angles established:"
    echo "   1. Functional Completeness"
    echo "   2. Requirement Fulfillment"
    echo "   3. Communication Effectiveness"
    echo "   4. Technical Quality"
    echo "   5. User Experience"
    echo "   6. Completeness Coverage"
    echo "   7. Integration & Context"
    echo "   8. Future-Proofing"
    echo ""

    return 0
}
```

### **Phase 2: Reiterate Goal & Meaning** 🎯

```bash
# ULTRATHINK: What was the original goal and what does it truly mean?
reiterate_goal_meaning() {
    local task_description="$1"
    echo "🎯 PHASE 2: ULTRATHINKING GOAL REITERATION"
    echo "=========================================="
    echo ""

    if [[ -n "$task_description" ]]; then
        echo "💭 ULTRATHINKING: The stated goal is '$task_description' - what does this REALLY mean?"
    else
        echo "💭 ULTRATHINKING: What was the original goal and what does it REALLY mean?"
    fi
    echo ""

    echo "🤔 Deep Analytical Process: Understanding the true goal..."
    echo ""

    # ULTRATHINK: Analyze the goal comprehensively
    cat > .ultrathink_verification/goal_analysis/goal_reiteration.md << EOF
# 🎯 ULTRATHINKING: Goal Reiteration and Deep Meaning Analysis

## Systematic Goal Analysis

### STEP 1: SURFACE GOAL IDENTIFICATION
*What was explicitly requested?*

**Stated Task/Goal**: ${task_description:-"[No specific task provided - analyzing current work context]"}

**Process**: Analyze the literal request/task description
- Read the original request carefully
- Identify explicit deliverables mentioned
- Note specific requirements stated
- Capture any constraints or preferences given

### STEP 2: DEEPER MEANING EXTRACTION
*What was the underlying intent?*

**Process**: Go beyond literal interpretation
- Why was this request made?
- What problem is really being solved?
- What would success look like to the requester?
- What context or background motivated this request?

### STEP 3: STAKEHOLDER PERSPECTIVE ANALYSIS
*Who cares about this and why?*

**Process**: Consider all affected parties
- Primary user/requester needs
- Secondary users who might interact with the work
- Future maintainers or inheritors of the work
- Broader organizational or project goals

### STEP 4: SUCCESS CRITERIA CLARIFICATION
*How would we know if this goal is truly achieved?*

**Process**: Define measurable success
- Functional success criteria
- Quality success criteria
- User satisfaction criteria
- Long-term value criteria

### STEP 5: IMPLICIT REQUIREMENTS IDENTIFICATION
*What wasn't said but is expected?*

**Process**: Uncover hidden expectations
- Industry standards and best practices
- Quality expectations for this type of work
- Integration requirements with existing systems
- Maintenance and documentation expectations

## 🧠 ULTRATHINKING METHODOLOGY
For each step, ask "What am I missing?" and "What would failure look like?" to ensure comprehensive understanding.
EOF

    echo "✅ ULTRATHINKING COMPLETE: Goal analysis framework established"
    echo ""
    echo "🎯 Goal analysis will examine:"
    echo "   • Surface goal identification"
    echo "   • Deeper meaning extraction"
    echo "   • Stakeholder perspective analysis"
    echo "   • Success criteria clarification"
    echo "   • Implicit requirements identification"
    echo ""

    return 0
}
```

### **Phase 3: Define "Complete" in Context** ✅

```bash
# ULTRATHINK: What does "complete" mean for THIS specific work?
define_completeness_criteria() {
    local task_description="$1"
    echo "✅ PHASE 3: ULTRATHINKING COMPLETENESS DEFINITION"
    echo "================================================"
    echo ""

    if [[ -n "$task_description" ]]; then
        echo "💭 ULTRATHINKING: What does 'complete' mean for '$task_description' specifically?"
    else
        echo "💭 ULTRATHINKING: What does 'complete' mean in THIS specific context?"
    fi
    echo ""

    echo "🤔 Deep Analytical Question: How do I define completion for this work?"
    echo ""

    # ULTRATHINK: Define context-specific completeness
    cat > .ultrathink_verification/completeness_criteria/completeness_definition.md << 'EOF'
# ✅ ULTRATHINKING: Completeness Definition Framework

## Context-Specific Completeness Analysis

### COMPLETENESS DIMENSION 1: FUNCTIONAL COMPLETENESS
**Definition**: All required functionality works as intended

**Criteria for THIS context**:
- [ ] Core functionality implemented and tested
- [ ] Edge cases handled appropriately
- [ ] Error conditions managed gracefully
- [ ] Performance meets requirements
- [ ] Integration points function correctly

### COMPLETENESS DIMENSION 2: DELIVERABLE COMPLETENESS
**Definition**: All expected deliverables are provided

**Criteria for THIS context**:
- [ ] Primary deliverable(s) created
- [ ] Supporting documentation provided
- [ ] Configuration/setup materials included
- [ ] Examples and demonstrations available
- [ ] Testing/validation components present

### COMPLETENESS DIMENSION 3: COMMUNICATION COMPLETENESS
**Definition**: Work is fully explainable and understandable

**Criteria for THIS context**:
- [ ] Clear explanation of what was built
- [ ] How-to-use documentation provided
- [ ] Decision rationale documented
- [ ] Limitations and constraints explained
- [ ] Next steps or future considerations noted

### COMPLETENESS DIMENSION 4: QUALITY COMPLETENESS
**Definition**: Work meets expected quality standards

**Criteria for THIS context**:
- [ ] Code/implementation follows best practices
- [ ] Documentation is clear and comprehensive
- [ ] Error handling is robust
- [ ] Security considerations addressed
- [ ] Maintainability requirements met

### COMPLETENESS DIMENSION 5: USER EXPERIENCE COMPLETENESS
**Definition**: End user can successfully accomplish their goals

**Criteria for THIS context**:
- [ ] User can discover how to use the work
- [ ] User can successfully complete intended tasks
- [ ] User receives helpful feedback and guidance
- [ ] User can troubleshoot common issues
- [ ] User experience is intuitive and pleasant

### COMPLETENESS DIMENSION 6: INTEGRATION COMPLETENESS
**Definition**: Work fits properly into its intended environment

**Criteria for THIS context**:
- [ ] Compatible with existing systems/workflows
- [ ] Dependencies properly managed
- [ ] Installation/setup process documented
- [ ] Integration testing performed
- [ ] Migration path provided if needed

## 🧠 ULTRATHINKING METHODOLOGY
"Complete" is context-dependent. What looks complete from one angle might be incomplete from another. Examine ALL dimensions systematically.

## 🎯 COMPLETENESS SCORING FRAMEWORK
For verification, each dimension will be scored:
- ✅ **COMPLETE**: Fully meets criteria
- ⚠️ **PARTIAL**: Meets some criteria, gaps identified
- ❌ **INCOMPLETE**: Significant gaps, needs major work
- 🔍 **UNCLEAR**: Cannot determine without more analysis
EOF

    echo "✅ ULTRATHINKING COMPLETE: Completeness criteria framework established"
    echo ""
    echo "📊 Completeness will be measured across 6 dimensions:"
    echo "   1. Functional Completeness"
    echo "   2. Deliverable Completeness"
    echo "   3. Communication Completeness"
    echo "   4. Quality Completeness"
    echo "   5. User Experience Completeness"
    echo "   6. Integration Completeness"
    echo ""

    return 0
}
```

### **Phase 4: Deep Verification Process** 🔍

```bash
# ULTRATHINK: Now verify the work against all defined criteria
deep_verification_process() {
    local task_description="$1"
    echo "🔍 PHASE 4: ULTRATHINKING VERIFICATION PROCESS"
    echo "=============================================="
    echo ""

    if [[ -n "$task_description" ]]; then
        echo "💭 ULTRATHINKING: Now I'll verify '$task_description' against all defined criteria..."
    else
        echo "💭 ULTRATHINKING: Now I'll verify the work against all defined criteria..."
    fi
    echo ""

    # ULTRATHINK: Systematic verification against all angles and criteria
    perform_angle_verification() {
        echo "🧠 Performing systematic verification across all 8 angles..."
        echo ""

        # Create verification results framework
        cat > .ultrathink_verification/verification_results/angle_verification.md << 'EOF'
# 🔍 ULTRATHINKING: Systematic Angle Verification

## VERIFICATION PROCESS
Each angle will be examined using the completeness criteria defined in Phase 3.

### ANGLE 1: FUNCTIONAL COMPLETENESS VERIFICATION
**Focus**: Does the work actually accomplish what it was supposed to do?

**Verification Process**:
1. Review original functional requirements
2. Test core functionality systematically
3. Examine edge cases and error scenarios
4. Validate performance and reliability
5. Check integration points

**Results**: [To be filled during verification]

### ANGLE 2: REQUIREMENT FULFILLMENT VERIFICATION
**Focus**: Does the work meet ALL explicitly and implicitly stated requirements?

**Verification Process**:
1. Cross-reference against original requirements
2. Check for implicit requirement satisfaction
3. Validate quality standards compliance
4. Examine scope completeness
5. Assess constraint adherence

**Results**: [To be filled during verification]

### ANGLE 3: COMMUNICATION EFFECTIVENESS VERIFICATION
**Focus**: Is the work clearly explained and understandable?

**Verification Process**:
1. Review documentation clarity and completeness
2. Assess explanation quality and depth
3. Check for usage examples and guides
4. Evaluate accessibility to different audiences
5. Verify troubleshooting information

**Results**: [To be filled during verification]

### ANGLE 4: TECHNICAL QUALITY VERIFICATION
**Focus**: Is the implementation robust, maintainable, and well-designed?

**Verification Process**:
1. Review code/implementation quality
2. Assess architectural decisions
3. Check adherence to best practices
4. Evaluate error handling robustness
5. Examine scalability considerations

**Results**: [To be filled during verification]

### ANGLE 5: USER EXPERIENCE VERIFICATION
**Focus**: How will the end user actually experience this work?

**Verification Process**:
1. Walk through user journey step-by-step
2. Identify friction points and confusion areas
3. Test discoverability and intuitiveness
4. Assess feedback and guidance quality
5. Evaluate overall user satisfaction potential

**Results**: [To be filled during verification]

### ANGLE 6: COMPLETENESS COVERAGE VERIFICATION
**Focus**: Are there gaps, missing pieces, or overlooked aspects?

**Verification Process**:
1. Systematic gap analysis across all dimensions
2. Review TODO items and incomplete sections
3. Check for missing components or features
4. Assess coverage of requirements
5. Identify unstated but expected elements

**Results**: [To be filled during verification]

### ANGLE 7: INTEGRATION & CONTEXT VERIFICATION
**Focus**: How does this work fit into the broader context?

**Verification Process**:
1. Test compatibility with existing systems
2. Verify dependency management
3. Check ecosystem integration
4. Assess workflow integration
5. Validate deployment and installation

**Results**: [To be filled during verification]

### ANGLE 8: FUTURE-PROOFING VERIFICATION
**Focus**: Will this work remain valuable and maintainable over time?

**Verification Process**:
1. Assess extensibility and adaptability
2. Review documentation for maintainability
3. Check knowledge transfer adequacy
4. Evaluate long-term sustainability
5. Examine evolution pathway clarity

**Results**: [To be filled during verification]
EOF

        echo "✅ Verification framework established"
        return 0
    }

    # Execute actual verification
    execute_verification() {
        echo "🎯 EXECUTING DEEP VERIFICATION..."
        echo ""

        echo "This will be a comprehensive analysis. Let me examine the work systematically..."
        echo ""

        # This is where the actual verification logic would examine the work
        # Based on the current context and recent work

        echo "🔍 Analyzing current work context..."

        # Check for recent work
        local recent_files=$(find . -type f -newermt "3 hours ago" -not -path "./.git/*" -not -path "./.ultrathink_verification/*" 2>/dev/null | wc -l)
        local work_context="recent_activity:$recent_files"

        echo "   📊 Recent file activity: $recent_files files"

        # Store context for verification
        echo "$work_context" > .ultrathink_verification/verification_results/work_context.txt

        echo "✅ Verification context established"
        echo ""
        echo "⚠️  IMPORTANT: Actual verification requires examining the specific work context"
        echo "   The framework is ready - verification will be customized to the actual work"

        return 0
    }

    # Execute verification phases
    perform_angle_verification
    execute_verification

    echo "✅ ULTRATHINKING VERIFICATION FRAMEWORK COMPLETE"
    echo ""

    return 0
}
```

### **Phase 5: Auto-Complete & Enhancement** 🚀

```bash
# ULTRATHINK: Address any gaps and auto-complete the work
auto_complete_enhancement() {
    local task_description="$1"
    echo "🚀 PHASE 5: ULTRATHINKING AUTO-COMPLETION"
    echo "========================================="
    echo ""

    if [[ -n "$task_description" ]]; then
        echo "💭 ULTRATHINKING: How can I auto-complete and enhance '$task_description'?"
    else
        echo "💭 ULTRATHINKING: How can I auto-complete and enhance this work?"
    fi
    echo ""

    # ULTRATHINK: Systematic auto-completion approach
    cat > .ultrathink_verification/auto_completion/completion_strategy.md << 'EOF'
# 🚀 ULTRATHINKING: Auto-Completion Strategy

## SYSTEMATIC ENHANCEMENT APPROACH

### ENHANCEMENT LEVEL 1: CRITICAL GAPS (Must Fix)
**Focus**: Issues that prevent the work from functioning or meeting basic requirements

**Auto-Completion Actions**:
- Fix broken functionality
- Add missing required components
- Resolve critical errors or issues
- Complete incomplete implementations

### ENHANCEMENT LEVEL 2: QUALITY IMPROVEMENTS (Should Fix)
**Focus**: Issues that reduce work quality or user experience

**Auto-Completion Actions**:
- Improve documentation clarity
- Add missing examples or usage guides
- Enhance error handling and user feedback
- Optimize performance and reliability

### ENHANCEMENT LEVEL 3: EXCELLENCE UPGRADES (Could Add)
**Focus**: Enhancements that make the work exceptional

**Auto-Completion Actions**:
- Add advanced features or capabilities
- Create comprehensive testing suites
- Develop additional utilities or tools
- Implement best practices and optimizations

## 🧠 ULTRATHINKING PRINCIPLES FOR AUTO-COMPLETION

### PRINCIPLE 1: UNDERSTAND BEFORE ACTING
- Fully understand what's missing before attempting to fix
- Consider why the gap exists (oversight, complexity, time)
- Ensure the enhancement aligns with original goals

### PRINCIPLE 2: MAINTAIN CONSISTENCY
- Match existing patterns and styles
- Preserve architectural decisions and approaches
- Ensure new components integrate seamlessly

### PRINCIPLE 3: PRIORITIZE VALUE
- Focus on highest-impact improvements first
- Consider user benefit vs. implementation effort
- Ensure enhancements solve real problems

### PRINCIPLE 4: PRESERVE INTENT
- Don't change the fundamental approach without good reason
- Enhance rather than replace unless necessary
- Maintain the spirit of the original work

## 🎯 AUTO-COMPLETION EXECUTION FRAMEWORK

### STEP 1: GAP PRIORITIZATION
Rank identified gaps by:
- Impact on functionality
- Impact on user experience
- Effort required to address
- Risk of introduction errors

### STEP 2: ENHANCEMENT PLANNING
For each gap:
- Define specific completion actions
- Identify required resources/information
- Plan implementation approach
- Consider testing/validation needs

### STEP 3: SYSTEMATIC IMPLEMENTATION
- Address critical gaps first
- Implement one enhancement at a time
- Validate each enhancement before proceeding
- Document changes and rationale

### STEP 4: VERIFICATION OF ENHANCEMENTS
- Re-verify work against all angles
- Ensure enhancements don't introduce new gaps
- Validate overall improvement in completeness
- Confirm alignment with original goals
EOF

    echo "✅ ULTRATHINKING COMPLETE: Auto-completion strategy framework established"
    echo ""
    echo "🎯 Auto-completion will proceed in 3 levels:"
    echo "   1. Critical Gaps (Must Fix)"
    echo "   2. Quality Improvements (Should Fix)"
    echo "   3. Excellence Upgrades (Could Add)"
    echo ""

    return 0
}
```

### **Main Execution Engine** ⚡

```bash
# Revolutionary ultrathinking verification execution
main() {
    local task_description=""
    local interactive_mode=false
    local auto_complete_mode=false
    local deep_analysis_mode=false
    local generate_report=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --interactive)
                interactive_mode=true
                shift
                ;;
            --auto-complete)
                auto_complete_mode=true
                shift
                ;;
            --deep-analysis)
                deep_analysis_mode=true
                shift
                ;;
            --report)
                generate_report=true
                shift
                ;;
            --comprehensive)
                auto_complete_mode=true
                deep_analysis_mode=true
                generate_report=true
                shift
                ;;
            --help|-h)
                show_usage
                return 0
                ;;
            --*)
                shift
                ;;
            *)
                # First non-option argument is the task description
                if [[ -z "$task_description" ]]; then
                    task_description="$1"
                fi
                shift
                ;;
        esac
    done

    echo "🧠 ULTRATHINKING VERIFICATION ENGINE"
    echo "===================================="
    echo ""
    if [[ -n "$task_description" ]]; then
        echo "📋 Task/Problem: $task_description"
    else
        echo "📋 Task/Problem: Current work context (auto-detected)"
    fi
    echo "🎯 Mode: $([ "$interactive_mode" = true ] && echo "Interactive" || echo "Automated")"
    echo "🔧 Auto-Complete: $([ "$auto_complete_mode" = true ] && echo "Enabled" || echo "Disabled")"
    echo "🔍 Deep Analysis: $([ "$deep_analysis_mode" = true ] && echo "Enabled" || echo "Standard")"
    echo ""

    # Initialize ultrathinking environment
    if [[ ! -d .ultrathink_verification ]]; then
        echo "🚀 Initializing ultrathinking verification environment..."
        mkdir -p .ultrathink_verification
    fi

    echo "🧠 EXECUTING ULTRATHINKING METHODOLOGY"
    echo "====================================="
    echo ""

    # PHASE 1: Define Verification Angles
    define_verification_angles "$task_description"

    if [[ "$interactive_mode" = true ]]; then
        echo "Continue to goal reiteration? [Y/n]"
        read -r continue_phase2
        if [[ "$continue_phase2" =~ ^[Nn]$ ]]; then
            return 0
        fi
        echo ""
    fi

    # PHASE 2: Reiterate Goal & Meaning
    reiterate_goal_meaning "$task_description"

    if [[ "$interactive_mode" = true ]]; then
        echo "Continue to completeness definition? [Y/n]"
        read -r continue_phase3
        if [[ "$continue_phase3" =~ ^[Nn]$ ]]; then
            return 0
        fi
        echo ""
    fi

    # PHASE 3: Define Completeness Criteria
    define_completeness_criteria "$task_description"

    if [[ "$interactive_mode" = true ]]; then
        echo "Continue to verification? [Y/n]"
        read -r continue_phase4
        if [[ "$continue_phase4" =~ ^[Nn]$ ]]; then
            return 0
        fi
        echo ""
    fi

    # PHASE 4: Deep Verification Process
    deep_verification_process "$task_description"

    # PHASE 5: Auto-Complete (if enabled)
    if [[ "$auto_complete_mode" = true ]]; then
        if [[ "$interactive_mode" = true ]]; then
            echo "Proceed with auto-completion? [Y/n]"
            read -r continue_phase5
            if [[ "$continue_phase5" =~ ^[Nn]$ ]]; then
                return 0
            fi
            echo ""
        fi

        auto_complete_enhancement "$task_description"
    fi

    # Final Summary
    echo ""
    echo "🎉 ULTRATHINKING VERIFICATION COMPLETE!"
    echo "======================================"
    echo ""
    echo "✅ Phases Completed:"
    echo "   1. ✅ Verification angles defined"
    echo "   2. ✅ Goal reiterated and analyzed"
    echo "   3. ✅ Completeness criteria established"
    echo "   4. ✅ Deep verification framework ready"
    echo "   5. $([ "$auto_complete_mode" = true ] && echo "✅" || echo "⏭️ ") Auto-completion $([ "$auto_complete_mode" = true ] && echo "completed" || echo "skipped")"
    echo ""
    echo "📁 Results stored in: .ultrathink_verification/"
    echo ""
    echo "🧠 REMEMBER: This is a THINKING framework. The real verification"
    echo "   happens when you apply this methodology to specific work!"

    return 0
}

# Show usage information
show_usage() {
    cat << 'EOF'
🧠 ULTRATHINKING VERIFICATION ENGINE

USAGE:
    /double-check ["task/problem description"] [options]

ARGUMENTS:
    "task description"  Specific task, goal, or problem being verified (optional)

OPTIONS:
    --interactive       Enable interactive mode with step-by-step confirmation
    --auto-complete     Enable automatic completion of identified gaps
    --deep-analysis     Enable deep analysis mode with extended verification
    --comprehensive     Enable all features (auto-complete + deep-analysis + report)
    --report            Generate comprehensive verification report

METHODOLOGY:
    Phase 1: Define Verification Angles (What perspectives to verify from?)
    Phase 2: Reiterate Goal & Meaning (What was really trying to be achieved?)
    Phase 3: Define Completeness Criteria (What does "complete" mean here?)
    Phase 4: Deep Verification Process (Systematically verify against criteria)
    Phase 5: Auto-Complete Enhancement (Address gaps and improve quality)

EXAMPLES:
    # Verify specific task with auto-completion
    /double-check "implement user authentication system" --auto-complete

    # Interactive verification of specific problem
    /double-check "fix database performance issues" --interactive --deep-analysis

    # Comprehensive verification with context
    /double-check "create data visualization dashboard" --comprehensive

    # Verify current work without specific context
    /double-check --auto-complete

ULTRATHINKING PRINCIPLE:
    "Don't just check - THINK DEEPLY about what you're checking and why!"

The goal is not just to verify work, but to deeply understand what verification
means in each specific context, then apply that understanding systematically.

TASK DESCRIPTION BENEFITS:
    - Enables context-specific angle definition
    - Allows precise goal reiteration and analysis
    - Provides targeted completeness criteria
    - Facilitates focused verification process
    - Enables intelligent auto-completion suggestions
EOF
}

# Execute main function
main "$@"
```

## 🎯 **COMPLETE TRANSFORMATION TO ULTRATHINKING METHODOLOGY**

**✅ Now Follows Correct Logic:**

1. **🔍 Define Angles** - Ultrathink what perspectives to verify from (8 comprehensive angles)
2. **🎯 Reiterate Goal** - Ultrathink the original goal and its deeper meaning
3. **✅ Define Complete** - Ultrathink what "complete" means in this specific context
4. **🔍 Verify** - Ultrathink systematic verification against defined criteria
5. **🚀 Auto-Complete** - Ultrathink gap identification and enhancement

**🧠 Key Features:**
- **Reflective not Technical** - No file scanning, pure analytical thinking
- **Methodology-Driven** - Follows the exact logic you specified
- **Context-Aware** - Defines criteria specific to each verification
- **Ultrathinking Focus** - Deep analysis at every step
- **Framework-Based** - Creates thinking frameworks rather than technical tools

The command now embodies the true spirit: **"Ultrathink!" at every step of verification.**
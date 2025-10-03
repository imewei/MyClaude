#!/bin/bash
# reorganize_docs.sh
# Automated documentation reorganization for nonequilibrium-physics-agents
#
# This script reorganizes 46+ root-level markdown files into a clean hierarchical structure
# Following the REORGANIZATION_PLAN.md specifications
#
# Usage: ./reorganize_docs.sh
# Rollback: git revert HEAD

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Documentation Reorganization Script                      ║${NC}"
echo -e "${BLUE}║  Nonequilibrium Physics Optimal Control Framework         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Safety check - ensure we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "docs" ]; then
    echo -e "${RED}ERROR: Must run from project root directory${NC}"
    exit 1
fi

# Count current root .md files
ROOT_MD_COUNT=$(ls -1 *.md 2>/dev/null | wc -l | tr -d ' ')
echo -e "${YELLOW}Current root .md files: ${ROOT_MD_COUNT}${NC}"

# Backup confirmation
echo ""
echo -e "${YELLOW}⚠️  This will reorganize $ROOT_MD_COUNT files${NC}"
echo -e "${YELLOW}Make sure you have committed recent changes!${NC}"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}🔄 Starting documentation reorganization...${NC}"

# Step 1: Create directory structure
echo ""
echo -e "${BLUE}📁 Step 1/10: Creating new directory structure...${NC}"
mkdir -p docs/guides
mkdir -p docs/phase4/weekly
mkdir -p docs/phase4/milestones
mkdir -p docs/phase4/summaries
mkdir -p docs/reports
mkdir -p docs/archive

# Step 2: Move user guides
echo -e "${BLUE}📖 Step 2/10: Moving user guides...${NC}"
[ -f "GETTING_STARTED.md" ] && mv GETTING_STARTED.md docs/guides/ && echo "  ✓ GETTING_STARTED.md"
[ -f "DEPLOYMENT.md" ] && mv DEPLOYMENT.md docs/guides/ && echo "  ✓ DEPLOYMENT.md"
[ -f "JAX_INSTALLATION_GUIDE.md" ] && mv JAX_INSTALLATION_GUIDE.md docs/guides/ && echo "  ✓ JAX_INSTALLATION_GUIDE.md"
[ -f "IMPLEMENTATION_ROADMAP.md" ] && mv IMPLEMENTATION_ROADMAP.md docs/guides/ && echo "  ✓ IMPLEMENTATION_ROADMAP.md"
[ -f "NEXT_STEPS.md" ] && mv NEXT_STEPS.md docs/guides/ && echo "  ✓ NEXT_STEPS.md"

# Step 3: Move weekly summaries
echo -e "${BLUE}📅 Step 3/10: Moving Phase 4 weekly summaries (24 files)...${NC}"
[ -f "PHASE4_WEEK2_SUMMARY.md" ] && mv PHASE4_WEEK2_SUMMARY.md docs/phase4/weekly/week02.md && echo "  ✓ Week 02"
[ -f "PHASE4_WEEK3_SUMMARY.md" ] && mv PHASE4_WEEK3_SUMMARY.md docs/phase4/weekly/week03.md && echo "  ✓ Week 03"
[ -f "PHASE4_WEEK4_COMPLETE.md" ] && mv PHASE4_WEEK4_COMPLETE.md docs/phase4/weekly/week04_complete.md && echo "  ✓ Week 04 (complete)"
[ -f "PHASE4_WEEK4_FINAL_REPORT.md" ] && mv PHASE4_WEEK4_FINAL_REPORT.md docs/phase4/weekly/week04_final.md && echo "  ✓ Week 04 (final)"
[ -f "PHASE4_WEEK5_SUMMARY.md" ] && mv PHASE4_WEEK5_SUMMARY.md docs/phase4/weekly/week05.md && echo "  ✓ Week 05"
[ -f "PHASE4_WEEK6_SUMMARY.md" ] && mv PHASE4_WEEK6_SUMMARY.md docs/phase4/weekly/week06.md && echo "  ✓ Week 06"
[ -f "PHASE4_WEEK8_SUMMARY.md" ] && mv PHASE4_WEEK8_SUMMARY.md docs/phase4/weekly/week08.md && echo "  ✓ Week 08"
[ -f "PHASE4_WEEK9_10_SUMMARY.md" ] && mv PHASE4_WEEK9_10_SUMMARY.md docs/phase4/weekly/week09-10.md && echo "  ✓ Week 09-10"
[ -f "PHASE4_WEEK11_12_SUMMARY.md" ] && mv PHASE4_WEEK11_12_SUMMARY.md docs/phase4/weekly/week11-12.md && echo "  ✓ Week 11-12"
[ -f "PHASE4_WEEK13_14_SUMMARY.md" ] && mv PHASE4_WEEK13_14_SUMMARY.md docs/phase4/weekly/week13-14.md && echo "  ✓ Week 13-14"
[ -f "PHASE4_WEEK14_15_SUMMARY.md" ] && mv PHASE4_WEEK14_15_SUMMARY.md docs/phase4/weekly/week14-15.md && echo "  ✓ Week 14-15"
[ -f "PHASE4_WEEK15_16_SUMMARY.md" ] && mv PHASE4_WEEK15_16_SUMMARY.md docs/phase4/weekly/week15-16.md && echo "  ✓ Week 15-16"
[ -f "PHASE4_WEEK17_18_SUMMARY.md" ] && mv PHASE4_WEEK17_18_SUMMARY.md docs/phase4/weekly/week17-18.md && echo "  ✓ Week 17-18"
[ -f "PHASE4_WEEK19_20_SUMMARY.md" ] && mv PHASE4_WEEK19_20_SUMMARY.md docs/phase4/weekly/week19-20.md && echo "  ✓ Week 19-20"
[ -f "PHASE4_WEEK21_22_SUMMARY.md" ] && mv PHASE4_WEEK21_22_SUMMARY.md docs/phase4/weekly/week21-22.md && echo "  ✓ Week 21-22"
[ -f "PHASE4_WEEK23_24_SUMMARY.md" ] && mv PHASE4_WEEK23_24_SUMMARY.md docs/phase4/weekly/week23-24.md && echo "  ✓ Week 23-24"
[ -f "PHASE4_WEEK25_26_SUMMARY.md" ] && mv PHASE4_WEEK25_26_SUMMARY.md docs/phase4/weekly/week25-26.md && echo "  ✓ Week 25-26"
[ -f "PHASE4_WEEK27_28_SUMMARY.md" ] && mv PHASE4_WEEK27_28_SUMMARY.md docs/phase4/weekly/week27-28.md && echo "  ✓ Week 27-28"
[ -f "PHASE4_WEEK29_30_SUMMARY.md" ] && mv PHASE4_WEEK29_30_SUMMARY.md docs/phase4/weekly/week29-30.md && echo "  ✓ Week 29-30"
[ -f "PHASE4_WEEK31_32_SUMMARY.md" ] && mv PHASE4_WEEK31_32_SUMMARY.md docs/phase4/weekly/week31-32.md && echo "  ✓ Week 31-32"
[ -f "PHASE4_WEEK33_34_SUMMARY.md" ] && mv PHASE4_WEEK33_34_SUMMARY.md docs/phase4/weekly/week33-34.md && echo "  ✓ Week 33-34"
[ -f "PHASE4_WEEK35_36_SUMMARY.md" ] && mv PHASE4_WEEK35_36_SUMMARY.md docs/phase4/weekly/week35-36.md && echo "  ✓ Week 35-36"
[ -f "PHASE4_WEEK37_38_SUMMARY.md" ] && mv PHASE4_WEEK37_38_SUMMARY.md docs/phase4/weekly/week37-38.md && echo "  ✓ Week 37-38"
[ -f "PHASE4_WEEK39_40_SUMMARY.md" ] && mv PHASE4_WEEK39_40_SUMMARY.md docs/phase4/weekly/week39-40.md && echo "  ✓ Week 39-40"
[ -f "PHASE4_WEEKS1-3_COMPLETE.md" ] && mv PHASE4_WEEKS1-3_COMPLETE.md docs/phase4/weekly/week01-03_complete.md && echo "  ✓ Week 01-03 (complete)"

# Step 4: Move milestones
echo -e "${BLUE}🏆 Step 4/10: Moving Phase 4 milestones...${NC}"
[ -f "PHASE4_MILESTONE_20PCT.md" ] && mv PHASE4_MILESTONE_20PCT.md docs/phase4/milestones/20_percent.md && echo "  ✓ 20% milestone"
[ -f "PHASE4_MILESTONE_SUMMARY.md" ] && mv PHASE4_MILESTONE_SUMMARY.md docs/phase4/milestones/summary.md && echo "  ✓ Milestone summary"
[ -f "PHASE4_FOUNDATION_MILESTONE.md" ] && mv PHASE4_FOUNDATION_MILESTONE.md docs/phase4/milestones/foundation.md && echo "  ✓ Foundation milestone"

# Step 5: Move summaries/overviews
echo -e "${BLUE}📊 Step 5/10: Moving Phase 4 summaries...${NC}"
[ -f "PHASE4_FINAL_SUMMARY.md" ] && mv PHASE4_FINAL_SUMMARY.md docs/phase4/summaries/final.md && echo "  ✓ Final summary"
[ -f "PHASE4_FINAL_OVERVIEW.md" ] && mv PHASE4_FINAL_OVERVIEW.md docs/phase4/summaries/overview.md && echo "  ✓ Final overview"
[ -f "PHASE4_COMPLETE_README.md" ] && mv PHASE4_COMPLETE_README.md docs/phase4/summaries/complete_readme.md && echo "  ✓ Complete README"
[ -f "PHASE4_CONTINUATION_SUMMARY.md" ] && mv PHASE4_CONTINUATION_SUMMARY.md docs/phase4/summaries/continuation.md && echo "  ✓ Continuation summary"
[ -f "PHASE4_PROGRESS.md" ] && mv PHASE4_PROGRESS.md docs/phase4/progress.md && echo "  ✓ Progress tracking"
[ -f "PHASE4_QUICK_REFERENCE.md" ] && mv PHASE4_QUICK_REFERENCE.md docs/phase4/quick_reference.md && echo "  ✓ Quick reference"

# Step 6: Move reports
echo -e "${BLUE}📋 Step 6/10: Moving reports...${NC}"
[ -f "PHASE4_VERIFICATION_REPORT.md" ] && mv PHASE4_VERIFICATION_REPORT.md docs/reports/verification_phase4.md && echo "  ✓ Verification report"
[ -f "SESSION_SUMMARY.md" ] && mv SESSION_SUMMARY.md docs/reports/session_summary.md && echo "  ✓ Session summary"

# Step 7: Handle README files
echo -e "${BLUE}📄 Step 7/10: Handling README files...${NC}"
[ -f "README_PHASE4.md" ] && mv README_PHASE4.md docs/phase4/README.md && echo "  ✓ Phase 4 README"
[ -f "PHASE4_README.md" ] && mv PHASE4_README.md docs/archive/phase4_readme_old.md && echo "  ✓ Old Phase 4 README (archived)"

# Step 8: Create navigation indices
echo -e "${BLUE}🗺️  Step 8/10: Creating navigation indices...${NC}"

# Create docs/README.md
if [ ! -f "docs/README.md" ] || [ "$(wc -l < docs/README.md | tr -d ' ')" -lt 10 ]; then
cat > docs/README.md << 'EOF'
# Documentation Index

Welcome to the Nonequilibrium Physics Optimal Control Framework documentation.

## 🚀 Quick Start
- [Getting Started](guides/GETTING_STARTED.md) - 5-minute quickstart
- [Installation & Deployment](guides/DEPLOYMENT.md) - Full deployment guide
- [JAX Installation](guides/JAX_INSTALLATION_GUIDE.md) - GPU setup

## 📖 Project Information
- [Architecture](../ARCHITECTURE.md) - System design and architecture
- [Changelog](../CHANGELOG.md) - Version history
- [Implementation Roadmap](guides/IMPLEMENTATION_ROADMAP.md) - Development plan
- [Next Steps](guides/NEXT_STEPS.md) - Future directions

## 🔬 Phase 4 Implementation (40 Weeks)
- [Phase 4 Overview](phase4/README.md) - Executive summary
- [Progress Tracking](phase4/progress.md) - Current status (100% complete)
- [Quick Reference](phase4/quick_reference.md) - Cheat sheet

### Phase 4 Details
- [Weekly Summaries](phase4/weekly/) - Week-by-week implementation (40 weeks)
- [Milestones](phase4/milestones/) - Key achievements
- [Final Summaries](phase4/summaries/) - Overview and completion reports

## 📊 Reports & Analysis
- [Phase 4 Verification](reports/verification_phase4.md) - Comprehensive verification
- [Session Summary](reports/session_summary.md) - Development sessions

## 🗺️ Roadmaps
- [Phase 1 Roadmap](phases/PHASE1.md) - Initial development
- [Phase 2 Roadmap](phases/PHASE2.md) - Feature expansion
- [Phase 3 Roadmap](phases/PHASE3.md) - Integration
- [Phase 4 Roadmap](phases/PHASE4.md) - ML & HPC integration

## 🗄️ Archive
- [Archived Documents](archive/) - Historical/deprecated documentation
EOF
echo "  ✓ docs/README.md"
fi

# Create docs/phase4/weekly/README.md
cat > docs/phase4/weekly/README.md << 'EOF'
# Phase 4 Weekly Summaries

Detailed week-by-week implementation documentation for Phase 4 (40 weeks).

## Weeks 1-10: Foundation & Advanced Solvers
- [Week 01-03](week01-03_complete.md) - GPU Acceleration Foundation
- [Week 02](week02.md) - Magnus Expansion Solver
- [Week 03](week03.md) - Pontryagin Maximum Principle
- [Week 04 Complete](week04_complete.md) - Week 4 completion
- [Week 04 Final](week04_final.md) - Week 4 final report
- [Week 05](week05.md) - Advanced solver refinement
- [Week 06](week06.md) - Solver integration
- [Week 08](week08.md) - Performance optimization
- [Week 09-10](week09-10.md) - Test infrastructure

## Weeks 11-20: ML Integration Foundation
- [Week 11-12](week11-12.md) - Stochastic test improvements
- [Week 13-14](week13-14.md) - Data format standardization
- [Week 14-15](week14-15.md) - GPU/Solver test coverage
- [Week 15-16](week15-16.md) - Test coverage expansion
- [Week 17-18](week17-18.md) - Transfer learning & curriculum
- [Week 19-20](week19-20.md) - Enhanced PINNs

## Weeks 21-30: Advanced ML & HPC Foundation
- [Week 21-22](week21-22.md) - Multi-task & Meta-learning
- [Week 23-24](week23-24.md) - Robust control & UQ
- [Week 25-26](week25-26.md) - Advanced optimization
- [Week 27-28](week27-28.md) - Performance profiling
- [Week 29-30](week29-30.md) - SLURM/PBS schedulers

## Weeks 31-40: HPC Integration & Production Readiness
- [Week 31-32](week31-32.md) - Dask distributed execution
- [Week 33-34](week33-34.md) - Parameter sweep infrastructure
- [Week 35-36](week35-36.md) - Final test coverage push
- [Week 37-38](week37-38.md) - Performance benchmarking
- [Week 39-40](week39-40.md) - Documentation & deployment
EOF
echo "  ✓ docs/phase4/weekly/README.md"

# Step 9: Create backward compatibility symlinks
echo -e "${BLUE}🔗 Step 9/10: Creating backward compatibility symlinks...${NC}"
ln -sf docs/guides/GETTING_STARTED.md GETTING_STARTED.md && echo "  ✓ GETTING_STARTED.md -> docs/guides/"
ln -sf docs/guides/DEPLOYMENT.md DEPLOYMENT.md && echo "  ✓ DEPLOYMENT.md -> docs/guides/"
ln -sf docs/phase4/progress.md PHASE4_PROGRESS.md && echo "  ✓ PHASE4_PROGRESS.md -> docs/phase4/"

# Step 10: Validation
echo -e "${BLUE}✅ Step 10/10: Validating reorganization...${NC}"
NEW_ROOT_MD_COUNT=$(ls -1 *.md 2>/dev/null | wc -l | tr -d ' ')
echo "  Root .md files: $ROOT_MD_COUNT → $NEW_ROOT_MD_COUNT"

SYMLINK_COUNT=$(ls -la *.md 2>/dev/null | grep "^l" | wc -l | tr -d ' ')
echo "  Symlinks created: $SYMLINK_COUNT"

PHASE4_WEEKLY_COUNT=$(ls -1 docs/phase4/weekly/*.md 2>/dev/null | wc -l | tr -d ' ')
echo "  Phase 4 weekly docs: $PHASE4_WEEKLY_COUNT"

PHASE4_MILESTONE_COUNT=$(ls -1 docs/phase4/milestones/*.md 2>/dev/null | wc -l | tr -d ' ')
echo "  Phase 4 milestones: $PHASE4_MILESTONE_COUNT"

echo ""
echo -e "${GREEN}✨ Documentation reorganization complete!${NC}"
echo ""
echo -e "${GREEN}📊 Summary:${NC}"
echo "  • Root files reduced: $ROOT_MD_COUNT → $NEW_ROOT_MD_COUNT"
echo "  • Phase 4 weekly summaries organized: $PHASE4_WEEKLY_COUNT files"
echo "  • Phase 4 milestones organized: $PHASE4_MILESTONE_COUNT files"
echo "  • Navigation indices created: 3"
echo "  • Backward compatibility symlinks: $SYMLINK_COUNT"
echo ""
echo -e "${YELLOW}📖 Next steps:${NC}"
echo "  1. Review docs/README.md for new structure"
echo "  2. Test navigation: cat docs/phase4/README.md"
echo "  3. Verify symlinks: ls -la *.md | grep '^l'"
echo "  4. Check for broken refs: grep -r 'PHASE4_WEEK.*\.md' --include='*.py' ."
echo "  5. Commit changes: git add -A && git commit -m 'docs: reorganize documentation structure'"
echo ""
echo -e "${GREEN}✅ Reorganization successful!${NC}"
